// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace habana::HPUDeviceContext {

// Graph/cache owners may outlive the device at interpreter shutdown. They must
// release SDK resources while the device's destruction queues are still alive.
class DeviceResource {
 public:
  virtual ~DeviceResource() = default;
  virtual void release_for_device_shutdown() = 0;
};

class DeviceResourceRegistry {
 public:
  void keep(const std::shared_ptr<DeviceResource>& resource) {
    if (!resource) throw std::invalid_argument("Null device resource");
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) throw std::logic_error("Device resource registry is closed");
    resources_.erase(
        std::remove_if(resources_.begin(), resources_.end(),
                       [](const auto& value) { return value.expired(); }),
        resources_.end());
    resources_.push_back(resource);
  }

  void release() {
    std::vector<std::weak_ptr<DeviceResource>> resources;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) return;
      closed_ = true;
      resources.swap(resources_);
    }
    // Do not hold the registry mutex while resources drain their own work.
    std::exception_ptr error;
    for (const auto& value : resources) {
      if (auto resource = value.lock()) {
        try { resource->release_for_device_shutdown(); }
        catch (...) { if (!error) error = std::current_exception(); }
      }
    }
    if (error) std::rethrow_exception(error);
  }

 private:
  std::mutex mutex_;
  bool closed_ = false;
  std::vector<std::weak_ptr<DeviceResource>> resources_;
};

void register_device_resource(const std::shared_ptr<DeviceResource>& resource);

} // namespace habana::HPUDeviceContext
