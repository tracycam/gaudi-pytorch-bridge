// SPDX-License-Identifier: Apache-2.0
#include "NativeExecutionPlan.h"
#include "RetainedTensorBinding.h"
#include "NativeDependencyPlan.h"
#include "backend/habana_device/HPUDevice.h"

#include <algorithm>
#include <array>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "backend/helpers/collective_kernel_info.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/hccl_communicator.h"
#include "backend/synapse_helpers/replay_event_batch.h"
#include "backend/synapse_helpers/devmem_logger.h"
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
  std::vector<RetainedTensorBinding> captured_inputs;
  std::shared_ptr<VecOfIValPtrSh> intermediates;
  VecOfIValPtrSh outputs;
  VecOfIValPtrSh dma_inputs;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> live_bindings;
  std::vector<size_t> external_events;
  NativeCommandAccess access;
  std::vector<std::string> access_issues;
};

// Lower an already-proven serial region, not a second graph executor. Address
// translation and completion still belong to the bridge allocator/event map.
struct PreparedRegionProgram : std::enable_shared_from_this<PreparedRegionProgram> {
  struct Call {
    std::shared_ptr<habana::RecipeLauncher> owner;
    std::shared_ptr<habana::PreparedCollectiveCall> collective;
    std::vector<synLaunchTensorInfoExt> tensors;
    std::vector<std::string> names;
    std::vector<size_t> slots;
    size_t binding_offset = 0;
  };
  std::vector<Call> calls;
  size_t binding_count = 0;
  std::vector<at::Tensor> retained;
  std::vector<c10::Storage> storage_owners;
  std::vector<synapse_helpers::device_ptr> addresses, dependencies;
  std::unique_ptr<synapse_helpers::device_memory::prepared_addresses> address_plan;
  size_t workspace_size = 0;

  static std::shared_ptr<PreparedRegionProgram> compile(
      const std::vector<RecipeCommand>& commands, size_t first, size_t last,
      const std::vector<RetainedTensorBinding>& leases, std::string& rejection) {
    auto result = std::make_shared<PreparedRegionProgram>();
    result->calls.reserve(last-first+1);
    std::vector<std::vector<uint64_t>> bindings;
    for (size_t i=first; i<=last; ++i) {
      auto unsupported = [&](std::string reason) -> std::shared_ptr<PreparedRegionProgram> {
        rejection = "commands " + std::to_string(first) + ".." + std::to_string(last) +
            ": command " + std::to_string(i) + ": " + std::move(reason);
        return nullptr;
      };
      const auto& command = commands[i];
      Call call;
      call.owner = command.launcher;
      std::vector<uint64_t> pointers;
      if (command.access.compute && !command.access.collectives && command.external_events.empty()) {
        if (!call.owner->recipe_ || call.owner->recipe_->graph_is_empty_)
          return unsupported("missing or empty compute recipe");
        // patch_launch_info also keeps compiler-eliminated shape entries.
        // They have no live recipe Tensor ID and are ignored by Synapse.
        // Apply the same live-binding rule as dependency admission above;
        // never let dead metadata reject an otherwise static serial region.
        std::vector<synLaunchTensorInfo> source;
        for (const auto& tensor : command.live_bindings) {
          if (tensor.tensorId != synapse_helpers::INVALID_SYN_TENSOR_ID)
            source.push_back(tensor);
        }
        call.tensors.resize(source.size());
        call.names.reserve(source.size());
        for (const auto& tensor : source) {
          if (tensor.tensorType != DATA_TENSOR)
            return unsupported("non-data binding type=" + std::to_string(tensor.tensorType) +
                " id=" + std::to_string(tensor.tensorId) + " name=" +
                (tensor.tensorName ? tensor.tensorName : ""));
          call.names.emplace_back(tensor.tensorName ? tensor.tensorName : "");
          pointers.push_back(tensor.pTensorAddress);
        }
        for (size_t n=0; n<source.size(); ++n) {
          auto& target = call.tensors[n];
          target.tensorName = call.names[n].c_str();
          target.tensorId = source[n].tensorId;
          target.tensorType = source[n].tensorType;
          std::copy(std::begin(source[n].tensorSize), std::end(source[n].tensorSize), target.tensorSize);
        }
        result->workspace_size = std::max(result->workspace_size, size_t(call.owner->workspace_size_));
      } else if (!command.access.compute && command.access.collectives == 1) {
        call.collective = call.owner->collective_kernels_info_->PrepareReplayCall();
        if (!call.collective) return unsupported("collective has no prepared adapter");
        pointers = {call.collective->input, call.collective->output};
      } else return unsupported("mixed, external-event or unsupported command");
      result->addresses.insert(result->addresses.end(), pointers.begin(), pointers.end());
      result->dependencies.insert(result->dependencies.end(), command.access.reads.begin(), command.access.reads.end());
      result->dependencies.insert(result->dependencies.end(), command.access.writes.begin(), command.access.writes.end());
      bindings.push_back(std::move(pointers));
      call.binding_offset = result->binding_count;
      result->binding_count += call.tensors.size();
      result->calls.push_back(std::move(call));
    }
    auto unique = [](auto& values) {
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
    };
    unique(result->addresses);
    unique(result->dependencies);
    result->address_plan = std::make_unique<synapse_helpers::device_memory::prepared_addresses>(result->addresses);
    for (size_t i=0; i<bindings.size(); ++i)
      for (auto address : bindings[i])
        result->calls[i].slots.push_back(std::lower_bound(result->addresses.begin(), result->addresses.end(), address)-result->addresses.begin());
    for (const auto& lease : leases) {
      result->retained.push_back(lease.tensor);
      // TensorImpl can be rebound after enqueue. Keep the captured Storage
      // independently alive until the region's completion event fires.
      result->storage_owners.push_back(lease.owner);
    }
    return result;
  }

  // Physical bindings belong to an invocation, never to the compiled program.
  // Retain one frame across repeated entries, and publish its completion only
  // after the invocation's last submitted command. Stream dependency checks
  // still run at each entry; reuse does not weaken external stream ordering.
  struct Frame {
    std::shared_ptr<const PreparedRegionProgram> program;
    std::unique_ptr<synapse_helpers::device_ptr_lock> lock;
    std::vector<synLaunchTensorInfoExt> bindings;
    synapse_helpers::device_ptr workspace = 0;
    synapse_helpers::active_recipe_counter* counter = nullptr;
    bool needs_publication = false;
    ~Frame() {
      lock.reset();
      program.reset();
      if (counter) counter->decrease_and_notify();
    }
  };

  std::shared_ptr<Frame> instantiate(
      synapse_helpers::device& device, synapse_helpers::device_ptr workspace) {
    auto frame = std::make_shared<Frame>();
    frame->program = shared_from_this();
    frame->workspace = workspace;
    frame->lock = std::make_unique<synapse_helpers::device_ptr_lock>(
        device.get_device_memory().lock_addresses(*address_plan));
    frame->bindings.reserve(binding_count);
    for (const auto& call : calls) {
      frame->bindings.insert(frame->bindings.end(), call.tensors.begin(), call.tensors.end());
      for (size_t n = 0; n < call.tensors.size(); ++n) {
        frame->bindings[call.binding_offset + n].pTensorAddress = frame->lock->at(call.slots[n]);
      }
    }
    device.get_active_recipe_counter().increase();
    frame->counter = &device.get_active_recipe_counter();
    return frame;
  }

  void replay(
      synapse_helpers::device& device, synapse_helpers::stream& stream,
      Frame& frame) const {
    PT_BRIDGE_LAMBDA_BEGIN("PreparedRegionSubmit");
    device.add_wait_events_on_stream(dependencies, stream);
    frame.needs_publication = true;
    for (const auto& call : calls) {
      if (call.collective) {
        call.collective->submit(
            frame.lock->at(call.slots[0]), frame.lock->at(call.slots[1]), stream);
      } else {
        const auto* tensors = call.tensors.empty()
            ? nullptr : frame.bindings.data() + call.binding_offset;
        auto status = synLaunchWithExternalEventsExt(
            stream, tensors, call.tensors.size(),
            call.owner->workspace_size_ ? frame.workspace : 0,
            call.owner->recipe_->syn_recipe_handle_, nullptr, 0, 0);
        HABANA_ASSERT(status == synSuccess, "Prepared recipe launch failed: ", status);
      }
    }
  }

  static void publish(
      synapse_helpers::device& device, synapse_helpers::stream& stream,
      const std::shared_ptr<Frame>& frame) {
    if (!frame->needs_publication) return;
    try {
      device.register_producer_on_stream(
          std::vector<synapse_helpers::device_ptr>(frame->program->dependencies),
          stream, [frame]() mutable {});
      frame->needs_publication = false;
    } catch (...) {
      // A partially submitted invocation must drain before its owners vanish.
      try { stream.synchronize(); } catch (...) { std::terminate(); }
      frame->needs_publication = false;
      throw;
    }
  }

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
  NativePlanState state = NativePlanState::Capturing;
  std::string rejection;
  struct Policy {
    bool compiled_regions = true;
    bool event_batching = false;
  } policy;
  NativeDependencyPlan dependencies;
  struct StreamRegion {
    size_t first;
    size_t last;
    std::shared_ptr<habana::HcclCommunicator> communicator;
    std::shared_ptr<synapse_helpers::ReplayEventBatch::Footprint> footprint;
    size_t workspace_size;
    std::shared_ptr<PreparedRegionProgram> program;
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

NativeExecutionPlan::NativeExecutionPlan() : impl_(std::make_unique<Impl>()) {
  // Prepared adapters run only in proven serial regions. Unsupported regions
  // retain their ordinary launchers; diagnostic switches can still disable it.
  impl_->policy.compiled_regions = true;
}
NativeExecutionPlan::~NativeExecutionPlan() = default;

void NativeExecutionPlan::reject(std::string reason) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->state == NativePlanState::Failed) return;
  if (impl_->rejection.empty()) {
    impl_->rejection = std::move(reason);
  }
  impl_->state = NativePlanState::Rejected;
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
  if (impl_->state != NativePlanState::Capturing) {
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
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      command.captured_inputs.emplace_back(input.toTensor());
    }
  }
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

bool NativeExecutionPlan::matches_captured_inputs(
    size_t command, at::ArrayRef<const at::Tensor*> tensors) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (command >= impl_->commands.size()) {
    return false;
  }
  const auto& inputs = impl_->commands[command].captured_inputs;
  if (inputs.size() != tensors.size()) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!tensors[i] || !inputs[i].matches(*tensors[i])) {
      return false;
    }
  }
  return true;
}

void NativeExecutionPlan::seal(size_t subgraphs) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->state == NativePlanState::Capturing && !impl_->commands.empty() &&
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
        std::string rejection;
        std::vector<synapse_helpers::device_ptr> addresses;
        size_t workspace = 0;
        for (size_t i=region.collectives.front(); i<=region.collectives.back(); ++i) {
          addresses.insert(addresses.end(), accesses[i].reads.begin(), accesses[i].reads.end());
          addresses.insert(addresses.end(), accesses[i].writes.begin(), accesses[i].writes.end());
          workspace = std::max(workspace, size_t(impl_->commands[i].launcher->workspace_size_));
        }
        impl_->regions.push_back({region.collectives.front(), region.collectives.back(),
                                  communicators[region.collectives.front()],
                                  std::make_shared<synapse_helpers::ReplayEventBatch::Footprint>(std::move(addresses)),
                                  workspace, PreparedRegionProgram::compile(impl_->commands,
                                      region.collectives.front(), region.collectives.back(), impl_->leases, rejection)});
        impl_->stats.prepared_regions += bool(impl_->regions.back().program);
        if (impl_->regions.back().program) {
          for (const auto& call : impl_->regions.back().program->calls) {
            ++impl_->stats.prepared_commands;
            impl_->stats.prepared_compute += !call.collective;
            impl_->stats.prepared_collectives += bool(call.collective);
          }
        } else {
          impl_->stats.prepared_rejections.push_back(std::move(rejection));
        }
        impl_->stats.coalesced_collectives += region.collectives.size();
      }
    }
    // Input snapshots only establish capture-time admission. Runtime guards
    // use the canonical retained leases, so release duplicate snapshot owners.
    for (auto& command : impl_->commands) {
      std::vector<RetainedTensorBinding>().swap(command.captured_inputs);
    }
    impl_->state = NativePlanState::Ready;
  } else if (impl_->state == NativePlanState::Capturing) {
    reject("capture does not provide one compiled recipe per subgraph");
  }
}

bool NativeExecutionPlan::capture_boundary(const at::Tensor& tensor) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->state != NativePlanState::Capturing || !tensor.defined() ||
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

bool NativeExecutionPlan::matches_boundaries(
    at::ArrayRef<const at::Tensor*> tensors) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (tensors.size() != impl_->boundary_leases.size()) return false;
  for (size_t slot=0; slot<tensors.size(); ++slot) {
    const auto& lease = impl_->leases[impl_->boundary_leases[slot]];
    // Replay still checks every retained Tensor's complete metadata before
    // submission. Here we only have to detect a replaced boundary Tensor.
    if (!tensors[slot] ||
        (tensors[slot]->unsafeGetTensorImpl() != lease.tensor.unsafeGetTensorImpl() &&
         !lease.matches(*tensors[slot]))) return false;
  }
  return true;
}

bool NativeExecutionPlan::replay(size_t repetitions) {
  PT_BRIDGE_LAMBDA_BEGIN("NativePlanReplay");
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  TORCH_CHECK(impl_->state != NativePlanState::Failed, "Native HPUGraph plan failed after submission");
  TORCH_CHECK(repetitions >= 1, "Replay repetitions must be positive");
  if (impl_->state != NativePlanState::Ready) {
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
    auto& device = habana::HPUDeviceContext::get_device();
    auto& compute = device.get_stream(impl_->commands.front().stream);
    const char* sync_launch = std::getenv("PT_HPU_SYNC_LAUNCH");
    const bool prepared_allowed = impl_->policy.compiled_regions &&
        !GET_ENV_FLAG_NEW(PT_COMPILE_ONLY_MODE) &&
        !synapse_helpers::memory_reporter_enable() &&
        (!sync_launch || std::atoi(sync_launch) == 0);
    auto submit_one = [&](auto& frames, synapse_helpers::device_ptr workspace) {
      PT_BRIDGE_LAMBDA_BEGIN("NativePlanSubmit");
      std::shared_ptr<habana::HcclCommunicator> communicator;
      size_t region_index = 0;
      synapse_helpers::stream* communication = nullptr;
      std::unique_ptr<synapse_helpers::ReplayEventBatch> event_batch;
      bool entered = false;
      auto leave = [&]() {
        if (event_batch) {
          event_batch->finish();
          ++impl_->stats.event_batches;
          impl_->stats.batched_publications += event_batch->publications();
          impl_->stats.interrupted_event_batches += event_batch->interrupted();
          event_batch.reset();
        }
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
            if (prepared_allowed && impl_->regions[region_index].program) {
              auto& frame = frames[region_index];
              if (!frame) {
                frame = impl_->regions[region_index].program->instantiate(device, workspace);
                ++impl_->stats.prepared_frames;
              }
              impl_->regions[region_index].program->replay(device, compute, *frame);
              ++impl_->stats.prepared_region_replays;
              impl_->stats.prepared_replayed_commands += impl_->regions[region_index].program->calls.size();
              index = impl_->regions[region_index].last;
              leave();
              ++region_index;
              continue;
            }
            if (impl_->policy.event_batching) {
              // Reserve the region's largest workspace before extending any
              // per-command lifetime to the region boundary.
              device.get_workspace_buffer(impl_->regions[region_index].workspace_size);
              event_batch = std::make_unique<synapse_helpers::ReplayEventBatch>(
                  device, compute, *impl_->regions[region_index].footprint);
            }
          }
          // An ordinary launcher may dispatch a collective on another stream.
          // Publish preceding writes before handing control to that adapter;
          // only a fully prepared interval can defer publication safely.
          for (const auto& frame : frames) {
            if (frame && frame->needs_publication) {
              PreparedRegionProgram::publish(device, compute, frame);
              ++impl_->stats.prepared_publications;
            }
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
    // Reuse one captured program and one HCCL queue job. The existing wait
    // keeps this invocation's borrowed references alive through SDK submission;
    // device completion remains asynchronous and retains its storage leases.
    auto submit = [&]() {
      // Reserve the largest workspace before retaining any in-flight frame.
      // Ordinary prefix/tail launches must not grow it between iterations.
      size_t workspace_size = 0;
      if (prepared_allowed) {
        for (const auto& command : impl_->commands) {
          workspace_size = std::max(workspace_size, size_t(command.launcher->workspace_size_));
        }
      }
      const auto workspace = prepared_allowed
          ? device.get_workspace_buffer(workspace_size) : 0;
      std::vector<std::shared_ptr<PreparedRegionProgram::Frame>> frames(impl_->regions.size());
      auto publish_frames = [&] {
        std::exception_ptr error;
        for (const auto& frame : frames) {
          if (!frame || !frame->needs_publication) continue;
          try {
            PreparedRegionProgram::publish(device, compute, frame);
            ++impl_->stats.prepared_publications;
          } catch (...) {
            if (!error) error = std::current_exception();
          }
        }
        if (error) std::rethrow_exception(error);
      };
      try {
        for (size_t iteration = 0; iteration < repetitions; ++iteration) {
          submit_one(frames, workspace);
        }
      } catch (...) {
        const auto error = std::current_exception();
        // Publish all submitted frames even if one registration fails. That
        // path drains the stream before propagating an error, preserving leases.
        try { publish_frames(); } catch (...) {}
        std::rethrow_exception(error);
      }
      publish_frames();
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
    impl_->state = NativePlanState::Failed;
    impl_->rejection = "native submission failed; reset before replay";
    throw;
  }
  impl_->stats.replays += repetitions;
  impl_->stats.coalesced_replays += impl_->regions.empty() ? 0 : repetitions;
  return true;
}

NativePlanState NativeExecutionPlan::state() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->state;
}

NativeReplayStats NativeExecutionPlan::stats() const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  auto result = impl_->stats;
  result.ready = impl_->state == NativePlanState::Ready;
  result.failed = impl_->state == NativePlanState::Failed;
  result.reason = impl_->rejection;
  result.compiled_regions = impl_->policy.compiled_regions;
  result.event_batching = impl_->policy.event_batching;
  result.retained_bindings = impl_->leases.size();
  for (const auto& region : impl_->regions) if (region.program) {
    result.address_cache_hits += region.program->address_plan->hits;
    result.address_cache_misses += region.program->address_plan->misses;
  }
  return result;
}

void NativeExecutionPlan::set_event_batching(bool enabled) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  TORCH_CHECK(impl_->state == NativePlanState::Ready, "Native plan is not ready");
  impl_->policy.event_batching = enabled;
  if (enabled) impl_->policy.compiled_regions = false;
}

void NativeExecutionPlan::set_compiled_regions(bool enabled) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  TORCH_CHECK(impl_->state == NativePlanState::Ready, "Native plan is not ready");
  impl_->policy.compiled_regions = enabled;
  if (enabled) impl_->policy.event_batching = false;
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
