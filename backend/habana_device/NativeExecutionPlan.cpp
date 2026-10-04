// SPDX-License-Identifier: Apache-2.0
#include "NativeExecutionPlan.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <utility>
#include "backend/helpers/collective_kernel_info.h"

namespace at::hpu {
namespace {

VecOfIValPtrSh snapshot(const VecOfIValPtrSh& values) {
  VecOfIValPtrSh copy;
  copy.reserve(values.size());
  for (const auto& value : values) {
    copy.push_back(value ? std::make_shared<habana_torch::jit::IValue>(*value)
                         : nullptr);
  }
  return copy;
}

struct TensorLease {
  at::Tensor tensor;
  const c10::StorageImpl* storage;
  void* address;
  int64_t offset;
  at::ScalarType dtype;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  explicit TensorLease(const at::Tensor& value)
      : tensor(value),
        storage(value.storage().unsafeGetStorageImpl()),
        address(value.storage().data_ptr().get()),
        offset(value.storage_offset()),
        dtype(value.scalar_type()),
        sizes(value.sizes().vec()),
        strides(value.strides().vec()) {}

  bool matches(const at::Tensor& value) const {
    return value.defined() && value.has_storage() &&
        value.storage().unsafeGetStorageImpl() == storage &&
        value.storage().data_ptr().get() == address &&
        value.storage_offset() == offset && value.scalar_type() == dtype &&
        value.sizes().equals(sizes) && value.strides().equals(strides);
  }
};

struct RecipeCommand {
  std::shared_ptr<habana::RecipeLauncher> launcher;
  synapse_helpers::hpuStream_t stream;
  std::vector<habana_torch::jit::IValue> inputs;
  std::shared_ptr<VecOfIValPtrSh> intermediates;
  VecOfIValPtrSh outputs;
  VecOfIValPtrSh dma_inputs;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> bindings;
  std::vector<synLaunchTensorInfo> live_bindings;
  std::vector<size_t> external_events;
};

bool static_recipe(synRecipeHandle recipe) {
  uint32_t count = 0;
  if (synTensorRetrieveLaunchAmount(recipe, &count) != synSuccess || !count ||
      count > 65536) {
    return false;
  }
  std::vector<uint64_t> ids(count);
  std::vector<synRetrievedLaunchTensorInfo> info(count);
  if (synTensorRetrieveLaunchIds(recipe, ids.data(), count) != synSuccess) {
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    info[i].tensorId = ids[i];
  }
  if (synTensorRetrieveLaunchInfoById(recipe, count, info.data()) !=
      synSuccess) {
    return false;
  }
  for (const auto& tensor : info) {
    if (tensor.tensorType != DATA_TENSOR || !tensor.tensorDims ||
        tensor.tensorDims > HABANA_DIM_MAX) {
      return false;
    }
    for (uint32_t d = 0; d < tensor.tensorDims; ++d) {
      if (!tensor.tensorMaxSize[d] ||
          tensor.tensorMaxSize[d] != tensor.tensorMinSize[d]) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

struct NativeExecutionPlan::Impl {
  mutable std::recursive_mutex mutex;
  std::vector<RecipeCommand> commands;
  std::vector<TensorLease> leases;
  std::unordered_multimap<const c10::StorageImpl*, size_t> leases_by_storage;
  std::vector<size_t> boundary_leases;
  NativeReplayStats stats;

  bool owns(const at::Tensor& value) const {
    if (!value.defined() || !value.has_storage()) {
      return false;
    }
    const auto range = leases_by_storage.equal_range(
        value.storage().unsafeGetStorageImpl());
    for (auto it = range.first; it != range.second; ++it) {
      if (leases[it->second].matches(value)) {
        return true;
      }
    }
    return false;
  }

  bool keep(const habana_torch::jit::IValue& value) {
    if (!value.isTensor()) {
      // Containers may hide tensors with mutable bindings. Admit only frozen
      // scalar values until there is a structured binding adapter.
      return value.isNone() || value.isInt() || value.isDouble() ||
          value.isBool() || value.isString();
    }
    auto tensor = value.toTensor();
    if (!tensor.defined() || !tensor.has_storage() ||
        tensor.device().type() != c10::DeviceType::HPU ||
        tensor.requires_grad() || !tensor.is_contiguous()) {
      return false;
    }
    if (!owns(tensor)) {
      leases_by_storage.emplace(
          tensor.storage().unsafeGetStorageImpl(), leases.size());
      leases.emplace_back(tensor);
    }
    return true;
  }
};

NativeExecutionPlan::NativeExecutionPlan() : impl_(std::make_unique<Impl>()) {}
NativeExecutionPlan::~NativeExecutionPlan() = default;

void NativeExecutionPlan::reject(std::string reason) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->stats.reason.empty()) {
    impl_->stats.reason = std::move(reason);
  }
  impl_->stats.ready = false;
  // A rejected plan must not pin an unused copy of the graph's activation
  // storage. In-flight launches retain their own bridge resource holders.
  impl_->commands.clear();
  impl_->leases.clear();
  impl_->leases_by_storage.clear();
  impl_->boundary_leases.clear();
  impl_->stats.commands = 0;
}

void NativeExecutionPlan::append(
    const habana::RecipeLauncher& launcher,
    synapse_helpers::hpuStream_t stream,
    at::ArrayRef<habana_torch::jit::IValue> inputs,
    const std::shared_ptr<VecOfIValPtrSh>& intermediates,
    const VecOfIValPtrSh& outputs,
    const std::vector<synLaunchTensorInfo>& bindings,
    const std::vector<size_t>& external_events,
    const VecOfIValPtrSh& dma_inputs) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (!impl_->stats.reason.empty()) {
    return;
  }
  if (!launcher.recipe_ || !launcher.collective_kernels_info_ ||
      !launcher.collective_kernels_info_->Empty() ||
      !external_events.empty() || !dma_inputs.empty() ||
      !static_recipe(launcher.recipe_->syn_recipe_handle_)) {
    reject("initial subset requires static compute-only recipes");
    return;
  }
  if (!impl_->commands.empty() && impl_->commands.front().stream != stream) {
    reject("multiple recipe streams");
    return;
  }
  for (const auto& input : inputs) {
    if (!impl_->keep(input)) {
      reject("unsupported input tensor lease");
      return;
    }
  }
  const std::array<const VecOfIValPtrSh*, 2> tensor_lists{
      intermediates.get(), &outputs};
  for (const auto* values : tensor_lists) {
    if (!values) {
      continue;
    }
    for (const auto& value : *values) {
      if (value && !impl_->keep(*value)) {
        reject("unsupported output/intermediate tensor lease");
        return;
      }
    }
  }
  RecipeCommand command;
  command.launcher = std::make_shared<habana::RecipeLauncher>(launcher);
  command.launcher->time_slot_.reset();
  command.stream = stream;
  command.inputs.assign(inputs.begin(), inputs.end());
  command.intermediates = std::make_shared<VecOfIValPtrSh>(
      intermediates ? snapshot(*intermediates) : VecOfIValPtrSh{});
  command.outputs = snapshot(outputs);
  command.bindings = bindings;
  command.names.reserve(bindings.size());
  for (const auto& binding : bindings) {
    command.names.emplace_back(binding.tensorName ? binding.tensorName : "");
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    command.bindings[i].tensorName = command.names[i].c_str();
  }
  impl_->commands.push_back(std::move(command));
  impl_->stats.commands = impl_->commands.size();
}

void NativeExecutionPlan::seal(size_t subgraphs) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->stats.reason.empty() && !impl_->commands.empty() &&
      impl_->commands.size() == subgraphs) {
    impl_->stats.ready = true;
  } else if (impl_->stats.reason.empty()) {
    reject("capture does not provide one compiled recipe per subgraph");
  }
}

bool NativeExecutionPlan::capture_boundary(const at::Tensor& tensor) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (!impl_->stats.reason.empty() || !tensor.defined() ||
      !tensor.has_storage()) {
    return false;
  }
  const auto range = impl_->leases_by_storage.equal_range(
      tensor.storage().unsafeGetStorageImpl());
  for (auto it = range.first; it != range.second; ++it) {
    if (impl_->leases[it->second].matches(tensor)) {
      impl_->boundary_leases.push_back(it->second);
      return true;
    }
  }
  reject("capture boundary has no matching compiled tensor lease");
  return false;
}

bool NativeExecutionPlan::matches_boundary(
    size_t slot, const at::Tensor& tensor) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return slot < impl_->boundary_leases.size() &&
      impl_->leases[impl_->boundary_leases[slot]].matches(tensor);
}

bool NativeExecutionPlan::replay() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  TORCH_CHECK(!impl_->stats.failed, "Native HPUGraph plan failed after submission");
  if (!impl_->stats.ready) {
    return false;
  }
  for (const auto& lease : impl_->leases) {
    if (!lease.matches(lease.tensor)) {
      reject("retained tensor storage/layout changed before replay");
      return false;
    }
  }
  try {
    for (auto& command : impl_->commands) {
      // graph::launch mutates addresses to physical locked addresses. Reset the
      // abstract bindings each time; never reuse a physical pointer from capture.
      command.live_bindings = command.bindings;
      command.launcher->Launch(
          command.stream, command.inputs, command.intermediates,
          command.outputs, command.live_bindings, command.external_events,
          command.dma_inputs);
    }
  } catch (...) {
    impl_->stats.failed = true;
    impl_->stats.ready = false;
    throw;
  }
  ++impl_->stats.replays;
  return true;
}

NativeReplayStats NativeExecutionPlan::stats() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->stats;
}

} // namespace at::hpu
