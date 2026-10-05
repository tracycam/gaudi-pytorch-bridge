// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "backend/kernel/hpu_habana_cache.h"

namespace at::hpu {

struct NativeReplayStats {
  size_t commands = 0;
  size_t replays = 0;
  size_t collectives = 0;
  size_t queued_replays = 0;
  size_t raw_edges = 0;
  size_t war_edges = 0;
  size_t waw_edges = 0;
  size_t collective_order_edges = 0;
  size_t serial_collectives = 0;
  size_t external_events = 0;
  size_t incomplete_access_commands = 0;
  size_t coalesced_collectives = 0;
  size_t coalesced_replays = 0;
  bool event_batching = false;
  size_t event_batches = 0;
  size_t batched_publications = 0;
  size_t interrupted_event_batches = 0;
  bool compiled_regions = false;
  size_t prepared_regions = 0;
  size_t prepared_region_replays = 0;
  bool ready = false;
  bool failed = false;
  std::string reason;
};

struct NativeAccessDiagnostic {
  size_t command;
  bool compute;
  size_t collectives;
  size_t external_events;
  bool complete;
  size_t reads;
  size_t writes;
  std::vector<std::string> issues;
};

// An HPUGraph-owned compiled execution plan. The first admitted subset retains
// static device tensors and delegates address locks/async leases to the bridge
// RecipeLauncher. No private ABI hook, SDK recorder, or second allocator.
class NativeExecutionPlan {
 public:
  NativeExecutionPlan();
  ~NativeExecutionPlan();
  void append(
      const habana::RecipeLauncher& launcher,
      synapse_helpers::hpuStream_t stream,
      at::ArrayRef<habana_torch::jit::IValue> inputs,
      const std::shared_ptr<VecOfIValPtrSh>& intermediates,
      const VecOfIValPtrSh& outputs,
      const std::vector<synLaunchTensorInfo>& bindings,
      const std::vector<size_t>& external_events,
      const VecOfIValPtrSh& dma_inputs);
  void reject(std::string reason);
  void seal(size_t subgraphs);
  // Record exact boundary slots independently of storage deduplication.
  bool capture_boundary(const at::Tensor& tensor);
  bool matches_boundary(size_t slot, const at::Tensor& tensor) const;
  // false means rejection before submission; submission failures throw and
  // poison the plan, so a partially submitted chain can never run twice.
  bool replay();
  void set_event_batching(bool enabled);
  void set_compiled_regions(bool enabled);
  NativeReplayStats stats() const;
  std::vector<NativeAccessDiagnostic> dependency_diagnostics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace at::hpu
