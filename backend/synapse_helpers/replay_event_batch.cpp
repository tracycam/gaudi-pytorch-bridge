// SPDX-License-Identifier: Apache-2.0
#include "backend/synapse_helpers/replay_event_batch.h"
#include "backend/synapse_helpers/device.h"
#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace synapse_helpers {
thread_local ReplayEventBatch* ReplayEventBatch::active_ = nullptr;
struct ReplayEventBatch::Completion {
  std::atomic<bool> completed{false};
  std::vector<event_done_callback> callbacks;
  void complete() {
    if (completed.exchange(true)) return;
    for (auto& callback : callbacks) if (callback) callback();
    callbacks.clear();
  }
};

ReplayEventBatch::Footprint::Footprint(std::vector<device_ptr> values)
    : addresses(std::move(values)) {
  std::sort(addresses.begin(), addresses.end());
  addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
  addresses.erase(std::remove(addresses.begin(), addresses.end(), device_ptr{0}), addresses.end());
  membership.insert(addresses.begin(), addresses.end());
}

ReplayEventBatch::ReplayEventBatch(device& dev, stream& stream, const Footprint& footprint)
    : device_(dev), stream_(stream), footprint_(footprint), completion_(std::make_shared<Completion>()) {
  if (active_) throw std::logic_error("Nested replay event batch");
  // Include reads as well as writes: later outside writers must wait for reads.
  device_.add_wait_events_on_stream(footprint_.addresses, stream_);
  completion_->callbacks.reserve(512);
  active_ = this;
}

ReplayEventBatch::~ReplayEventBatch() {
  try { finish(); } catch (...) { std::terminate(); }
}

bool ReplayEventBatch::contains(device& dev, stream& stream,
                                const std::vector<device_ptr>& addresses) const {
  return &dev == &device_ && stream == stream_ &&
      std::all_of(addresses.begin(), addresses.end(), [this](device_ptr p) {
        return !p || footprint_.membership.count(p);
      });
}

bool ReplayEventBatch::wait_is_internal(device& dev, stream& stream,
                                        const std::vector<device_ptr>& addresses) {
  if (!active_) return false;
  if (active_->contains(dev, stream, addresses)) return true;
  interrupt();
  return false;
}

bool ReplayEventBatch::defer(device& dev, stream& stream,
                             const std::vector<device_ptr>& addresses,
                             event_done_callback& callback) {
  if (!active_) return false;
  if (!active_->contains(dev, stream, addresses)) { interrupt(); return false; }
  active_->completion_->callbacks.push_back(std::move(callback));
  ++active_->publications_;
  return true;
}

void ReplayEventBatch::interrupt() {
  if (active_) {
    active_->interrupted_ = true;
    active_->finish();
  }
}

void ReplayEventBatch::finish() {
  if (active_ != this) return;
  active_ = nullptr;
  if (!publications_) return;
  auto completion = completion_;
  try {
    device_.register_producer_on_stream(std::vector<device_ptr>(footprint_.addresses),
                                       stream_, [completion] { completion->complete(); });
  } catch (...) {
    // Never release a submitted prefix's Tensor/recipe/address-lock owners
    // before completion. If the device cannot be drained, fail closed.
    try { stream_.synchronize(); completion->complete(); }
    catch (...) { std::terminate(); }
    throw;
  }
}
} // namespace synapse_helpers
