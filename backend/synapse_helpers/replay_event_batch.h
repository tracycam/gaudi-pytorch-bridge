// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <unordered_set>
#include <vector>
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/event.h"

namespace synapse_helpers {
class device;
class stream;

// Only a compiled same-stream region with complete access information may
// instantiate this scope. Normal launches keep their existing event protocol.
class ReplayEventBatch {
 public:
  struct Footprint {
    explicit Footprint(std::vector<device_ptr> values);
    std::vector<device_ptr> addresses;
    std::unordered_set<device_ptr> membership;
  };
  ReplayEventBatch(device&, stream&, const Footprint&);
  ~ReplayEventBatch();
  void finish();
  size_t publications() const { return publications_; }
  bool interrupted() const { return interrupted_; }
  static bool wait_is_internal(device&, stream&, const std::vector<device_ptr>&);
  static bool defer(device&, stream&, const std::vector<device_ptr>&, event_done_callback&);
  static void interrupt();
 private:
  struct Completion;
  static thread_local ReplayEventBatch* active_;
  bool contains(device&, stream&, const std::vector<device_ptr>&) const;
  device& device_;
  stream& stream_;
  const Footprint& footprint_;
  std::shared_ptr<Completion> completion_;
  size_t publications_ = 0;
  bool interrupted_ = false;
};
} // namespace synapse_helpers
