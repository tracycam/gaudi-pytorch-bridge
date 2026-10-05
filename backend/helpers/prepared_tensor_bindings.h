// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/Tensor.h>
#include <memory>
#include <vector>
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/prepared_launch_bindings.h"

namespace habana {

// Capture-time Tensor references and abstract dependency addresses. These are
// reused only after the plan's full storage/layout preflight. The existing
// collective interface takes non-const vectors but admitted snapshots never
// rewrite them. Physical addresses still come from the allocator each launch.
struct PreparedTensorBindings {
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputs;
  std::vector<c10::Storage> storage_owners;
  std::vector<synapse_helpers::device_ptr> reads;
  std::vector<synapse_helpers::device_ptr> producers;
  std::shared_ptr<const synapse_helpers::PreparedLaunchBindings> launch;
};

} // namespace habana
