// SPDX-License-Identifier: Apache-2.0
#include "NativeExecutionPlan.h"
#include "RetainedTensorBinding.h"
#include "NativeDependencyPlan.h"
#include "backend/habana_device/HPUDevice.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "backend/helpers/collective_kernel_info.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "pytorch_helpers/habana_helpers/logging.h"

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


struct RecipeCommand {
  std::shared_ptr<habana::RecipeLauncher> launcher;
  synapse_helpers::hpuStream_t stream;
  std::vector<habana_torch::jit::IValue> inputs;
  std::shared_ptr<VecOfIValPtrSh> intermediates;
  VecOfIValPtrSh outputs;
  VecOfIValPtrSh dma_inputs;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> live_bindings;
  std::vector<size_t> external_events;
  NativeCommandAccess access;
  std::vector<std::string> access_issues;
};

bool static_recipe(
    synRecipeHandle recipe,
    std::vector<synRetrievedLaunchTensorInfo>& info) {
  uint32_t count = 0;
  if (synTensorRetrieveLaunchAmount(recipe, &count) != synSuccess || !count ||
      count > 65536) {
    return false;
  }
  std::vector<uint64_t> ids(count);
  info.resize(count);
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
  std::vector<RetainedTensorBinding> leases;
  std::unordered_multimap<const c10::StorageImpl*, size_t> leases_by_storage;
  std::vector<size_t> boundary_leases;
  NativeReplayStats stats;
  NativeDependencyPlan dependencies;
  struct StreamRegion {
    size_t first;
    size_t last;
    std::shared_ptr<habana::HcclCommunicator> communicator;
  };
  std::vector<StreamRegion> regions;

  uint64_t allocation_key(uint64_t address, bool& complete) const {
    if (!address) {
      return 0;
    }
    uint64_t key = 0;
    for (const auto& lease : leases) {
      auto base = reinterpret_cast<uint64_t>(lease.address);
      if (address >= base && address - base < lease.bytes) {
        if (key && key != base) {
          // Independently owned overlapping allocations need an explicit
          // alias adapter; retain the ordinary launch path for this command.
          complete = false;
        }
        key = base;
      }
    }
    if (!key) {
      complete = false;
    }
    return key;
  }

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
  impl_->dependencies = {};
  impl_->regions.clear();
  impl_->stats.commands = 0;
  impl_->stats.collectives = 0;
  impl_->stats.raw_edges = 0;
  impl_->stats.war_edges = 0;
  impl_->stats.waw_edges = 0;
  impl_->stats.collective_order_edges = 0;
  impl_->stats.serial_collectives = 0;
  impl_->stats.external_events = 0;
  impl_->stats.incomplete_access_commands = 0;
  impl_->stats.coalesced_collectives = 0;
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
  if (!launcher.collective_kernels_info_ || !dma_inputs.empty()) {
    reject("missing collective metadata or unsupported DMA inputs");
    return;
  }
  if (!habana::HPUDeviceContext::get_device().IsStreamASyncEnabled()) {
    reject("synchronous stream mode requires the original adapter");
    return;
  }
  auto collectives = launcher.collective_kernels_info_->SnapshotForReplay();
  if (!collectives) {
    reject("unsupported collective kind or dtype");
    return;
  }
  std::vector<synRetrievedLaunchTensorInfo> recipe_tensors;
  if ((!launcher.recipe_ && collectives->Empty()) ||
      (launcher.recipe_ &&
       !static_recipe(launcher.recipe_->syn_recipe_handle_, recipe_tensors))) {
    reject("missing or non-static compute recipe");
    return;
  }
  if (!launcher.recipe_ && !external_events.empty()) {
    reject("external tensor events without a compute recipe");
    return;
  }
  for (auto index : external_events) {
    if (index >= bindings.size()) {
      reject("external tensor event has no binding");
      return;
    }
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
  command.launcher->collective_kernels_info_ = std::move(collectives);
  command.external_events = external_events;
  impl_->stats.collectives += command.launcher->collective_kernels_info_->Count();
  command.stream = stream;
  command.inputs.assign(inputs.begin(), inputs.end());
  command.intermediates = std::make_shared<VecOfIValPtrSh>(
      intermediates ? snapshot(*intermediates) : VecOfIValPtrSh{});
  command.outputs = snapshot(outputs);
  command.access.compute = static_cast<bool>(launcher.recipe_);
  command.access.collectives = command.launcher->collective_kernels_info_->Count();
  command.access.external_events = external_events.size();
  // Recipe metadata is authoritative for direction. The existing producers
  // vector also contains read inputs for WAR protection; it is not a write set.
  std::unordered_map<uint64_t, bool> directions;
  for (const auto& tensor : recipe_tensors) {
    directions.emplace(tensor.tensorId, tensor.isInput != 0);
  }
  std::unordered_set<uint64_t> active_bindings;
  auto resolve = [&](uint64_t address, std::string label) {
    bool complete = true;
    auto key = impl_->allocation_key(address, complete);
    command.access.complete &= complete;
    if (!complete) {
      command.access_issues.push_back(std::move(label) + ": no unique retained allocation");
    }
    return key;
  };
  for (const auto& binding : bindings) {
    if (!launcher.recipe_ || !binding.pTensorAddress) {
      continue;
    }
    // patch_launch_info retains entries whose tensors disappeared during
    // compilation. RetrieveLaunchIds is the authoritative live launch set;
    // the invalid sentinel is not a Tensor read or write. Keep the original
    // SDK bindings unchanged, and do not exempt any unknown *valid* ID.
    if (binding.tensorId == synapse_helpers::INVALID_SYN_TENSOR_ID) {
      continue;
    }
    auto direction = directions.find(binding.tensorId);
    if (direction == directions.end()) {
      command.access.complete = false;
      command.access_issues.push_back(
          std::string(binding.tensorName ? binding.tensorName : "") +
          ": missing recipe tensor ID " + std::to_string(binding.tensorId));
      continue;
    }
    active_bindings.insert(binding.tensorId);
    auto key = resolve(binding.pTensorAddress,
                       binding.tensorName ? binding.tensorName : "");
    (direction->second ? command.access.reads : command.access.writes)
        .push_back(key);
  }
  for (const auto& tensor : recipe_tensors) {
    if (!active_bindings.count(tensor.tensorId)) {
      command.access.complete = false;
      command.access_issues.push_back(
          "live recipe tensor has no nonzero binding: " + std::to_string(tensor.tensorId));
    }
  }
  std::vector<PtTensorInfoShared> collective_reads;
  std::vector<PtTensorInfoShared> collective_writes;
  if (!command.launcher->collective_kernels_info_->ReplayAccesses(
          collective_reads, collective_writes)) {
    command.access.complete = false;
    command.access_issues.emplace_back("unsupported collective access description");
  }
  auto collect = [&](const auto& values, auto& keys) {
    for (const auto& tensor : values) {
      if (tensor->get_numel()) {
        keys.push_back(resolve(tensor->get_buffer_start_syn(), tensor->get_syn_name()));
      }
    }
  };
  collect(collective_reads, command.access.reads);
  collect(collective_writes, command.access.writes);
  command.launcher->PrepareTensorBindings(
      command.inputs, command.intermediates, command.outputs, bindings,
      external_events);
  command.live_bindings = bindings;
  command.names.reserve(bindings.size());
  for (const auto& binding : bindings) {
    command.names.emplace_back(binding.tensorName ? binding.tensorName : "");
  }
  for (size_t i = 0; i < bindings.size(); ++i) {
    command.live_bindings[i].tensorName = command.names[i].c_str();
  }
  command.launcher->prepared_bindings_->launch->initialize(command.live_bindings);
  impl_->commands.push_back(std::move(command));
  impl_->stats.commands = impl_->commands.size();
}

void NativeExecutionPlan::seal(size_t subgraphs) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->stats.reason.empty() && !impl_->commands.empty() &&
      impl_->commands.size() == subgraphs) {
    std::vector<NativeCommandAccess> accesses;
    accesses.reserve(impl_->commands.size());
    for (const auto& command : impl_->commands) {
      accesses.push_back(command.access);
      impl_->stats.external_events += command.access.external_events;
      impl_->stats.incomplete_access_commands += !command.access.complete;
    }
    impl_->dependencies = NativeDependencyPlan::compile(accesses);
    for (const auto& edge : impl_->dependencies.edges) {
      switch (edge.kind) {
        case NativeDependencyKind::Raw: ++impl_->stats.raw_edges; break;
        case NativeDependencyKind::War: ++impl_->stats.war_edges; break;
        case NativeDependencyKind::Waw: ++impl_->stats.waw_edges; break;
        case NativeDependencyKind::CollectiveOrder:
          ++impl_->stats.collective_order_edges; break;
      }
    }
    impl_->stats.serial_collectives = impl_->dependencies.serial_collectives.size();
    // Compile maximal proven regions. Unknown accesses, partial readiness,
    // mixed compute/collective commands and communicator changes are barriers.
    // A region needs more than one collective to amortize its two FIFO guards.
    if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SFG)) {
      std::vector<uint64_t> communicator_keys;
      std::vector<std::shared_ptr<habana::HcclCommunicator>> communicators;
      for (const auto& command : impl_->commands) {
        auto communicator = command.launcher->collective_kernels_info_
                                ->SingleReplayCommunicator();
        communicator_keys.push_back(reinterpret_cast<uint64_t>(communicator.get()));
        communicators.push_back(std::move(communicator));
      }
      for (const auto& region : impl_->dependencies.serial_regions(accesses, communicator_keys)) {
        impl_->regions.push_back({region.collectives.front(), region.collectives.back(),
                                  communicators[region.collectives.front()]});
        impl_->stats.coalesced_collectives += region.collectives.size();
      }
    }
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
  PT_BRIDGE_LAMBDA_BEGIN("NativePlanReplay");
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  TORCH_CHECK(!impl_->stats.failed, "Native HPUGraph plan failed after submission");
  if (!impl_->stats.ready) {
    return false;
  }
  {
    PT_BRIDGE_LAMBDA_BEGIN("NativePlanBindingPreflight");
    for (const auto& lease : impl_->leases) {
      if (!lease.matches(lease.tensor)) {
        reject("retained tensor storage/layout changed before replay");
        return false;
      }
    }
  }
  try {
    auto submit = [&]() {
      PT_BRIDGE_LAMBDA_BEGIN("NativePlanSubmit");
      std::shared_ptr<habana::HcclCommunicator> communicator;
      size_t region_index = 0;
      auto& device = habana::HPUDeviceContext::get_device();
      auto& compute = device.get_stream(impl_->commands.front().stream);
      synapse_helpers::stream* communication = nullptr;
      bool entered = false;
      auto leave = [&]() {
        if (entered) {
          // Publish the mapped collective sequence back to the communicator's
          // original FIFO before another ordinary HCCL job may run there.
          device.record_and_wait_for_event(compute, *communication,
                                           [communicator]() {});
          entered = false;
        }
      };
      try {
        for (size_t index = 0; index < impl_->commands.size(); ++index) {
          auto& command = impl_->commands[index];
          if (region_index < impl_->regions.size() &&
              index == impl_->regions[region_index].first) {
            communicator = impl_->regions[region_index].communicator;
            communication = &communicator->getDeviceCtxt()->get_stream_fromhandle(
                communicator->getCommStream());
            device.record_and_wait_for_event(*communication, compute,
                                             [communicator]() {});
            entered = true;
          }
          if (entered && command.access.collectives) {
            command.launcher->collective_kernels_info_->SetReplayStream(&compute);
          }
          // graph::launch mutates addresses to physical locked addresses. Reset the
          // abstract bindings each time; never reuse a physical pointer from capture.
          command.launcher->prepared_bindings_->launch->restore_addresses(
              command.live_bindings);
          command.launcher->Launch(
              command.stream, command.inputs, command.intermediates,
              command.outputs, command.live_bindings, command.external_events,
              command.dma_inputs);
          if (entered && index == impl_->regions[region_index].last) {
            leave();
            ++region_index;
          }
        }
      } catch (...) {
        // A submitted prefix cannot be retried. Attempt to restore the stream
        // order, then preserve the original error and poison the graph below.
        try { leave(); } catch (...) {}
        throw;
      }
    };
    if (impl_->stats.collectives) {
      // Queue the whole chain once. The shared HCCL queue preserves ordering
      // against ordinary collectives; snapshots submit inline inside that job.
      habana::SubmitCollectivePlan(std::move(submit));
      ++impl_->stats.queued_replays;
    } else {
      submit();
    }
  } catch (...) {
    impl_->stats.failed = true;
    impl_->stats.ready = false;
    throw;
  }
  ++impl_->stats.replays;
  impl_->stats.coalesced_replays += !impl_->regions.empty();
  return true;
}

NativeReplayStats NativeExecutionPlan::stats() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->stats;
}

std::vector<NativeAccessDiagnostic>
NativeExecutionPlan::dependency_diagnostics() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  std::vector<NativeAccessDiagnostic> result;
  result.reserve(impl_->commands.size());
  for (size_t i = 0; i < impl_->commands.size(); ++i) {
    const auto& command = impl_->commands[i];
    const auto& access = command.access;
    result.push_back({i, access.compute, access.collectives, access.external_events,
                      access.complete, access.reads.size(), access.writes.size(),
                      command.access_issues});
  }
  return result;
}

} // namespace at::hpu
