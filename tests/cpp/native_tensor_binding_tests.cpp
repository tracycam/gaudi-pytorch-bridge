// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include "habana_kernels/hccl_kernels.h"
#include "backend/habana_device/RetainedTensorBinding.h"
#include "backend/habana_device/NativeDependencyPlan.h"
#include "backend/kernel/hpu_habana_cache.h"

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

TEST(PreparedTensorBindingsTest, PreservesReferencesDependenciesAndIndependentOwners) {
  auto x = at::ones({8, 8});
  auto middle = at::ones({8, 8});
  auto output = at::zeros({8, 8});
  auto original_owner = x.storage();
  auto input_address = reinterpret_cast<uint64_t>(x.storage().data_ptr().get());
  auto output_address = reinterpret_cast<uint64_t>(output.storage().data_ptr().get());
  habana::RecipeValueSpec recipe;
  habana::RecipeLauncher launcher(recipe);
  std::vector<habana_torch::jit::IValue> inputs{x, int64_t{7}};
  auto intermediates = std::make_shared<VecOfIValPtrSh>();
  intermediates->push_back(std::make_shared<habana_torch::jit::IValue>(middle));
  VecOfIValPtrSh outputs{std::make_shared<habana_torch::jit::IValue>(output)};
  synLaunchTensorInfo external{};
  external.pTensorAddress = input_address;
  launcher.PrepareTensorBindings(inputs, intermediates, outputs, {external}, {0});
  auto prepared = launcher.prepared_bindings_;
  ASSERT_EQ(prepared->inputs.size(), 2);
  ASSERT_EQ(prepared->outputs.size(), 1);
  ASSERT_EQ(prepared->storage_owners.size(), 3);
  EXPECT_EQ(prepared->reads, std::vector<uint64_t>{input_address});
  EXPECT_EQ(prepared->producers, std::vector<uint64_t>{output_address});
  x.set_(at::zeros({8, 8}));
  EXPECT_EQ(prepared->storage_owners.front().unsafeGetStorageImpl(),
            original_owner.unsafeGetStorageImpl());
  EXPECT_NE(prepared->inputs.front().storage().unsafeGetStorageImpl(),
            original_owner.unsafeGetStorageImpl());
  // A completion resource can outlive the launcher/plan owner.
  launcher.prepared_bindings_.reset();
  EXPECT_EQ(prepared->reads.front(), input_address);
}

TEST(PreparedTensorBindingsTest, ExternalAliasKeepsEntryWaitButUsesExternalProducerEvent) {
  auto x = at::ones({8});
  habana::RecipeValueSpec recipe;
  habana::RecipeLauncher launcher(recipe);
  std::vector<habana_torch::jit::IValue> inputs{x};
  VecOfIValPtrSh outputs{std::make_shared<habana_torch::jit::IValue>(x)};
  synLaunchTensorInfo external{};
  external.pTensorAddress = reinterpret_cast<uint64_t>(x.storage().data_ptr().get());
  launcher.PrepareTensorBindings(inputs, {}, outputs, {external}, {0});
  EXPECT_EQ(launcher.prepared_bindings_->reads.size(), 1);
  EXPECT_TRUE(launcher.prepared_bindings_->producers.empty());
}

TEST(PreparedLaunchBindingsTest, RestoresAbstractAddressesWithoutCopyingStaticMetadata) {
  synLaunchTensorInfo data{};
  data.pTensorAddress = 0x1000;
  data.tensorId = 17;
  data.tensorType = DATA_TENSOR;
  data.tensorSize[0] = 8;
  data.tensorName = "data";
  auto alias = data;
  alias.tensorId = 19;
  std::vector<synLaunchTensorInfo> live{data, alias};
  const synapse_helpers::PreparedLaunchBindings prepared(live);
  prepared.initialize(live);
  // Mimic repeated allocator resolution, including two aliases of one handle.
  for (uint64_t physical : {0x9000, 0xa000}) {
    live[0].pTensorAddress = physical;
    live[1].pTensorAddress = physical;
    prepared.restore_addresses(live);
    EXPECT_EQ(live[0].pTensorAddress, 0x1000);
    EXPECT_EQ(live[1].pTensorAddress, 0x1000);
    EXPECT_EQ(live[0].tensorId, 17);
    EXPECT_EQ(live[1].tensorId, 19);
    EXPECT_EQ(live[0].tensorSize[0], 8);
    EXPECT_STREQ(live[0].tensorName, "data");
  }
}

TEST(PreparedLaunchBindingsTest, HostPointersAndZeroSizedTensorAreNotDeviceHandles) {
  synLaunchTensorInfo scalar{};
  scalar.pTensorAddress = 0x1000;
  scalar.tensorType = DATA_TENSOR;
  auto host = scalar;
  host.pTensorAddress = 0x2000;
  host.tensorType = HOST_TO_DEVICE_TENSOR;
  auto empty = scalar;
  empty.pTensorAddress = 0;
  std::vector<synLaunchTensorInfo> live{scalar, host, empty};
  const synapse_helpers::PreparedLaunchBindings prepared(live);
  EXPECT_EQ(prepared.allocator_addresses,
            (std::vector<uint64_t>{0x1000, 0, 0}));
  prepared.initialize(live);
  EXPECT_EQ(live[0].tensorSize[0], 1);
  EXPECT_EQ(live[1].tensorSize[0], 1);
  EXPECT_EQ(live[2].tensorSize[0], 0);
  live[0].pTensorAddress = 0x9000;
  live[1].pTensorAddress = 0;
  prepared.restore_addresses(live);
  EXPECT_EQ(live[1].pTensorAddress, 0x2000);
}

TEST(PreparedLaunchBindingsTest, InvalidCountRejectsBeforeChangingAnyAddress) {
  synLaunchTensorInfo data{};
  data.pTensorAddress = 0x1000;
  const synapse_helpers::PreparedLaunchBindings prepared({data});
  std::vector<synLaunchTensorInfo> invalid{data, data};
  invalid.front().pTensorAddress = 0x9000;
  EXPECT_THROW(prepared.restore_addresses(invalid), std::invalid_argument);
  EXPECT_EQ(invalid.front().pTensorAddress, 0x9000);
}

namespace {
using at::hpu::NativeCommandAccess;
using at::hpu::NativeDependencyKind;
using at::hpu::NativeDependencyPlan;

size_t count_edges(const NativeDependencyPlan& plan, NativeDependencyKind kind) {
  return std::count_if(plan.edges.begin(), plan.edges.end(),
                       [kind](const auto& edge) { return edge.kind == kind; });
}
} // namespace

TEST(NativeDependencyPlanTest, OnlyTrueWritesCreateRawEdges) {
  // Both recipes read a shared weight. WAR event registration for that weight
  // must not turn them into a fictitious producer/consumer pair.
  auto plan = NativeDependencyPlan::compile({
      {{1, 2}, {3}, true}, {{2, 4}, {5}, true}, {{3, 5}, {6}, true}});
  ASSERT_EQ(plan.edges.size(), 2);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Raw), 2);
  EXPECT_TRUE(plan.serial_collectives.empty());
}

TEST(NativeDependencyPlanTest, ReadBeforeOverwriteAndInplaceRetainAllHazards) {
  auto plan = NativeDependencyPlan::compile({
      {{}, {1}, true}, {{1}, {2}, true}, {{1}, {1}, false, 1},
      {{1}, {3}, true}});
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Raw), 3);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::War), 1);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Waw), 1);
  for (const auto& edge : plan.edges) {
    EXPECT_LT(edge.producer, edge.consumer);
  }
  // The immediately previous command writes key2, not key1 consumed by HCCL.
  EXPECT_TRUE(plan.serial_collectives.empty());
}

TEST(NativeDependencyPlanTest, CollectiveChainRequiresBothAdjacentRawEdges) {
  auto plan = NativeDependencyPlan::compile({
      {{1}, {2}, true}, {{2}, {3}, false, 1}, {{3}, {4}, true},
      {{4}, {4}, false, 1}, {{4}, {5}, true}});
  EXPECT_EQ(plan.serial_collectives, (std::vector<size_t>{1, 3}));
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::CollectiveOrder), 1);
  // A collective with an independent compute consumer must keep overlap.
  auto independent = NativeDependencyPlan::compile({
      {{1}, {2}, true}, {{2}, {3}, false, 1}, {{7}, {8}, true}});
  EXPECT_TRUE(independent.serial_collectives.empty());
}

TEST(NativeDependencyPlanTest, PartialUnknownAndMixedCommandsBlockSerialAdmission) {
  std::vector<NativeCommandAccess> chain{
      {{1}, {2}, true}, {{2}, {3}, false, 1}, {{3}, {4}, true}};
  for (size_t position : {0, 1, 2}) {
    auto partial = chain;
    partial[position].external_events = 1;
    EXPECT_TRUE(NativeDependencyPlan::compile(partial).serial_collectives.empty());
    auto unknown = chain;
    unknown[position].complete = false;
    EXPECT_TRUE(NativeDependencyPlan::compile(unknown).serial_collectives.empty());
  }
  chain[1].compute = true;
  EXPECT_TRUE(NativeDependencyPlan::compile(chain).serial_collectives.empty());
  chain[1].compute = false;
  chain[1].collectives = 2;
  EXPECT_TRUE(NativeDependencyPlan::compile(chain).serial_collectives.empty());
}

TEST(NativeDependencyPlanTest, CoarsenedAliasesDeduplicateKeysAndKeepWar) {
  auto plan = NativeDependencyPlan::compile({
      {{1, 1, 0}, {2, 2}, true}, {{1}, {3}, true},
      {{}, {1, 1}, true}, {{2, 3}, {4}, true}});
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::War), 2);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Raw), 2);
  EXPECT_EQ(plan.edges.size(), 4);
}

TEST(NativeDependencyPlanTest, ReadHistoryResetsAtWriteAndQueueOrderIsSeparate) {
  auto plan = NativeDependencyPlan::compile({
      {{1}, {}, true}, {{1}, {}, true}, {{}, {1}, false, 1},
      {{1}, {}, true}, {{}, {1}, false, 1}});
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::War), 3);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Waw), 1);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::Raw), 1);
  EXPECT_EQ(count_edges(plan, NativeDependencyKind::CollectiveOrder), 1);
  EXPECT_TRUE(NativeDependencyPlan::compile({}).edges.empty());
}

TEST(NativeDependencyPlanTest, RegionsStopAtUnknownAndCommunicatorBoundaries) {
  std::vector<NativeCommandAccess> commands;
  std::vector<uint64_t> communicators;
  for (size_t i = 0; i < 13; ++i) {
    commands.push_back({{i + 1}, {i + 2}, i % 2 == 0, i % 2});
    communicators.push_back(i % 2 ? 7 : 0);
  }
  auto regions = NativeDependencyPlan::compile(commands).serial_regions(commands, communicators);
  ASSERT_EQ(regions.size(), 1);
  EXPECT_EQ(regions[0].collectives, (std::vector<size_t>{1,3,5,7,9,11}));
  commands[6].complete = false;
  regions = NativeDependencyPlan::compile(commands).serial_regions(commands, communicators);
  ASSERT_EQ(regions.size(), 2);
  EXPECT_EQ(regions[0].collectives, (std::vector<size_t>{1,3}));
  EXPECT_EQ(regions[1].collectives, (std::vector<size_t>{9,11}));
  commands[6].complete = true;
  communicators[7] = 8;
  regions = NativeDependencyPlan::compile(commands).serial_regions(commands, communicators);
  ASSERT_EQ(regions.size(), 2);
  EXPECT_EQ(regions[0].collectives, (std::vector<size_t>{1,3,5}));
  EXPECT_EQ(regions[1].collectives, (std::vector<size_t>{9,11}));
  EXPECT_THROW(NativeDependencyPlan::compile(commands).serial_regions(commands, {}),
               std::invalid_argument);
}

TEST(NativeDependencyPlanTest, OneCollectiveCannotAmortizeRegionGuards) {
  std::vector<NativeCommandAccess> commands{
      {{1}, {2}, true}, {{2}, {3}, false, 1}, {{3}, {4}, true}};
  auto plan = NativeDependencyPlan::compile(commands);
  ASSERT_EQ(plan.serial_collectives.size(), 1);
  EXPECT_TRUE(plan.serial_regions(commands, {0,7,0}).empty());
}
