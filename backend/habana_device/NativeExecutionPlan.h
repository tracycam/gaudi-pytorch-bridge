// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include "backend/kernel/hpu_habana_cache.h"

namespace at::hpu {

struct NativeReplayStats {
  size_t commands = 0;
  size_t replays = 0;
  bool ready = false;
  bool failed = false;
  std::string reason;
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
  NativeReplayStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace at::hpu
