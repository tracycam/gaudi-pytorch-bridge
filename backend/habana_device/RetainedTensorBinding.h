// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/Tensor.h>
#include <vector>

namespace at::hpu {

struct RetainedTensorBinding {
  at::Tensor tensor;
  // TensorImpl may be rebound in place. Retain Storage independently so
  // pointer reuse can never impersonate the old allocation owner.
  c10::Storage owner;
  const c10::StorageImpl* storage;
  void* address;
  size_t bytes;
  int64_t offset;
  at::ScalarType dtype;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  explicit RetainedTensorBinding(const at::Tensor& value)
      : tensor(value),
        owner(value.storage()),
        storage(owner.unsafeGetStorageImpl()),
        address(value.storage().data_ptr().get()),
        bytes(owner.nbytes()),
        offset(value.storage_offset()),
        dtype(value.scalar_type()),
        sizes(value.sizes().vec()),
        strides(value.strides().vec()) {}

  bool matches(const at::Tensor& value) const {
    return value.defined() && value.has_storage() &&
        value.storage().unsafeGetStorageImpl() == storage &&
        value.storage().data_ptr().get() == address &&
        value.storage().nbytes() == bytes &&
        value.storage_offset() == offset && value.scalar_type() == dtype &&
        value.sizes().equals(sizes) && value.strides().equals(strides);
  }
};

} // namespace at::hpu
