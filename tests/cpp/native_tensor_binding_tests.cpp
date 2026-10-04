// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include "habana_kernels/hccl_kernels.h"
#include "backend/habana_device/RetainedTensorBinding.h"

// Exercise real Storage/TensorImpl mutation on CPU without relying on Lazy
// set_, which the pinned HPU bridge does not support. The binding contract is
// device-independent; HPU admission remains the execution adapter's job.
TEST(NativeTensorBindingTest, SameValuesDifferentOwnerIsNotSameSlot) {
  auto x = at::ones({8, 8});
  auto other = x.clone();
  at::hpu::RetainedTensorBinding x_slot(x);
  at::hpu::RetainedTensorBinding other_slot(other);
  EXPECT_TRUE(x_slot.matches(x));
  EXPECT_TRUE(other_slot.matches(other));
  EXPECT_FALSE(x_slot.matches(other));
  EXPECT_FALSE(other_slot.matches(x));
}

TEST(NativeTensorBindingTest, ExactAliasesAndValueUpdatesAreAccepted) {
  auto x = at::ones({8, 8});
  at::hpu::RetainedTensorBinding slot(x);
  x.add_(1);
  EXPECT_TRUE(slot.matches(x));
  EXPECT_TRUE(slot.matches(x.alias()));
  EXPECT_FALSE(slot.matches(x.transpose(0, 1)));
  EXPECT_FALSE(slot.matches(x.view({4, 16})));
  EXPECT_FALSE(slot.matches(at::Tensor()));
}

TEST(NativeTensorBindingTest, TensorImplRebindKeepsOriginalStorageAlive) {
  auto x = at::ones({8, 8});
  auto old_storage = x.storage();
  auto other = at::zeros({8, 8});
  at::hpu::RetainedTensorBinding slot(x);
  x.set_(other);
  EXPECT_EQ(slot.owner.unsafeGetStorageImpl(), old_storage.unsafeGetStorageImpl());
  EXPECT_FALSE(slot.matches(x));
  EXPECT_FALSE(slot.matches(other));
  EXPECT_FALSE(slot.matches(slot.tensor));
  auto alias = at::empty({0}, x.options());
  alias.set_(old_storage, 0, {8, 8}, {8, 1});
  EXPECT_TRUE(slot.matches(alias));
}

TEST(NativeTensorBindingTest, OffsetDtypeAndAllocationExtentMatter) {
  auto base = at::ones({64});
  auto x = base.narrow(0, 0, 32);
  at::hpu::RetainedTensorBinding slot(x);
  EXPECT_FALSE(slot.matches(base.narrow(0, 32, 32)));
  EXPECT_FALSE(slot.matches(x.to(at::kDouble)));
  base.resize_({128});
  EXPECT_FALSE(slot.matches(x));
}


TEST(NativeCollectiveSubmissionTest, ExceptionReturnsToCallerAndQueueSurvives) {
  EXPECT_THROW(
      habana::SubmitCollectivePlan([] { throw std::runtime_error("probe"); }),
      std::runtime_error);
  bool submitted = false;
  habana::SubmitCollectivePlan([&] { submitted = true; });
  EXPECT_TRUE(submitted);
}

TEST(NativeCollectiveSubmissionTest, OneJobKeepsNestedSubmissionOnSameThread) {
  auto caller = std::this_thread::get_id();
  std::thread::id submission;
  std::thread::id nested;
  int completed = 0;
  habana::SubmitCollectivePlan([&] {
    submission = std::this_thread::get_id();
    ++completed;
    habana::SubmitCollectivePlan([&] {
      nested = std::this_thread::get_id();
      ++completed;
    });
    ++completed;
  });
  EXPECT_NE(caller, submission);
  EXPECT_EQ(submission, nested);
  EXPECT_EQ(completed, 3);
}
