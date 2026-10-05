// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "backend/synapse_helpers/device_types.h"

namespace synapse_helpers {

// Immutable capture-time abstract addresses. Physical addresses are resolved
// under the allocator's normal lock for every launch, never cached here.
struct PreparedLaunchBindings {
  explicit PreparedLaunchBindings(
      const std::vector<synLaunchTensorInfo>& bindings) {
    abstract_addresses.reserve(bindings.size());
    allocator_addresses.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& binding = bindings[i];
      abstract_addresses.push_back(binding.pTensorAddress);
      allocator_addresses.push_back(
          binding.tensorType == HOST_TO_DEVICE_TENSOR ? 0
                                                     : binding.pTensorAddress);
      if (binding.pTensorAddress != 0 && binding.tensorSize[0] == 0) {
        scalar_slots.push_back(i);
      }
    }
  }

  void restore_addresses(std::vector<synLaunchTensorInfo>& bindings) const {
    if (bindings.size() != abstract_addresses.size()) {
      throw std::invalid_argument("prepared launch binding count changed");
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      bindings[i].pTensorAddress = abstract_addresses[i];
    }
  }

  void initialize(std::vector<synLaunchTensorInfo>& bindings) const {
    restore_addresses(bindings);
    // Match ordinary launch's 0D fix while leaving zero-sized tensors intact.
    for (auto slot : scalar_slots) {
      bindings[slot].tensorSize[0] = 1;
    }
  }

  std::vector<device_ptr> abstract_addresses;
  std::vector<device_ptr> allocator_addresses;
  std::vector<size_t> scalar_slots;
};

} // namespace synapse_helpers
