// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <vector>
#include <absl/container/flat_hash_map.h>
#include "backend/synapse_helpers/device_types.h"

namespace synapse_helpers {
// Producer addresses are a set; their iteration order has no ordering meaning.
// The event manager supplies unique addresses and serializes all mutations.
class EventAddresses {
 public:
  explicit EventAddresses(std::vector<device_ptr> values) : values_(std::move(values)) {}
  const std::vector<device_ptr>& values() const { return values_; }
  void remove(device_ptr value) {
    if (!indexed_ && values_.size() >= 64) {
      positions_.reserve(values_.size());
      for (size_t i = 0; i < values_.size(); ++i) {
        positions_.emplace(values_[i], i);
      }
      indexed_ = true;
    }
    if (!indexed_) {
      values_.erase(std::remove(values_.begin(), values_.end(), value), values_.end());
      return;
    }
    auto found = positions_.find(value);
    if (found == positions_.end()) return;
    auto slot = found->second;
    if (slot != values_.size()-1) {
      auto moved = values_.back();
      values_[slot] = moved;
      positions_.at(moved) = slot;
    }
    values_.pop_back();
    positions_.erase(found);
  }
 private:
  std::vector<device_ptr> values_;
  absl::flat_hash_map<device_ptr, size_t> positions_;
  bool indexed_ = false;
};
} // namespace synapse_helpers
