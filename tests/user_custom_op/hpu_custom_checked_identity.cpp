// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_custom_op_pt2.h"

namespace {
habana::PartialOutputMetaDataVector checked_meta(const at::Stack& inputs) {
  const auto input = inputs.at(0).toTensor();
  const auto expected = static_cast<at::ScalarType>(inputs.at(1).toInt());
  TORCH_CHECK(input.scalar_type() == expected,
              "checked identity expected ", expected, " received ", input.scalar_type());
  return {{expected, input.sizes().vec()}};
}

at::Tensor checked_identity(at::Tensor input, int64_t dtype) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
      "custom_op::checked_identity");
  return descriptor.execute({input, dtype}).at(0);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("checked_identity(Tensor input, int dtype) -> Tensor");
  habana::custom_op::registerUserCustomOp(
      "custom_op::checked_identity", "identity", checked_meta, nullptr);
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("checked_identity", checked_identity);
}
