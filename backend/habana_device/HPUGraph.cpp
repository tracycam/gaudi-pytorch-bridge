/**
 * Copyright (c) 2021-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "HPUGraph.h"
#include "NativeExecutionPlan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/core/StreamGuard.h>

#include "backend/backend_meta.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/h2d_scales.h"
#include "habana_kernels/h2d_scales_lazy.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/view_utils.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace at::hpu {

namespace {
void ValidateMarkedInputs(
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<at::Tensor>& inputs) {
  TORCH_CHECK(inputs.size() >= sizes.size(),
              "HPUGraph replay requires all marked inputs; expected ",
              sizes.size(), ", got ", inputs.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    TORCH_CHECK(inputs[i].defined() && inputs[i].sizes().equals(sizes[i]),
                "HPUGraph marked input shape changed at slot ", i);
  }
}
} // namespace

template <typename T>
inline bool isExists(
    const std::unordered_set<T>& setContainer,
    const T& element) {
  return (setContainer.count(element) > 0);
}

HPUGraph::HPUGraph()
    // HPUStreams may not be default-constructed.
    : capture_stream_(c10::hpu::getCurrentHPUStream()) {}

void HPUGraph::capture_begin(bool dry_run) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (capturing_) {
    // already captured started, error. only one graph
    // capture is suported.
    PT_DEVICE_FATAL("GRAPH:: graph Capture already in progress");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  capture_stream_ = stream;
  /*flush current Accumulated graph, before capture */
  PT_IRGRAPH_DEBUG("step marker due to new HPUGraph::capture_begin");
  habana_lazy::HbLazyTensor::StepMarker({});
  native_boundaries_.clear();
  native_boundary_values_.clear();
  native_boundary_references_ = 0;
  adapter_replays_ = 0;
  last_replay_native_ = false;
  last_replay_reason_.clear();
  native_plan_ = std::make_unique<NativeExecutionPlan>();
  if (dry_run) {
    native_plan_->reject("dry-run requires deferred plan materialization");
  }

  dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }

  capturing_ = true;

  /* Set graph capture mode on */
  context->setCapturing(true);
  /* pass this struct to global context */
  context->setCaptureGraph(this);
  context->setDryRun(dry_run);
}

void HPUGraph::capture_end() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!capturing_) {
    // need to start the capture.
    PT_DEVICE_DEBUG("GRAPH:: Use Graph capture to Begin the capture");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();

  if (stream != capture_stream_) {
    PT_DEVICE_FATAL("GRAPH:: Capture must end on the same stream it began on.");
    return;
  }

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  /*flush graph to capture in the end */
  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::capture_end");
  habana_lazy::HbLazyTensor::StepMarker({});
  capturing_ = false;

  /* Set graph capture mode off */
  context->setCapturing(false);
  context->setCaptureGraph(nullptr);
  context->setDryRun(false);

  // Save all input lazy tensors and free the output IR values
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    auto single_graph = captured_graphs[i];
    auto num_inputs = single_graph->input_vals_.size();
    single_graph->hblazy_tensors_in_.clear();
    for (size_t inp = 0; inp < num_inputs; ++inp) {
      std::shared_ptr<habana_lazy::Data> d =
          single_graph->input_vals_[inp].m_data_ptr.lock();
      single_graph->hblazy_tensors_in_.emplace_back(std::move(d));
    }
  }

  // Clear the user marked inputs list
  context->ClearHPUGraphUserMarkedInputs();
  if (native_plan_) {
    // End-of-capture boundary snapshots alone are insufficient: Lazy lowering
    // can replace an in-place input's storage during capture (e.g. INT64 add_).
    // A fixed-address recipe would then repeatedly read the obsolete input.
    // Compare each recipe's capture-time inputs with the final Lazy bindings;
    // the existing adapter owns rebinding when this static contract fails.
    for (size_t i = 0; i < captured_graphs.size(); ++i) {
      std::vector<const at::Tensor*> inputs;
      for (const auto& input : captured_graphs[i]->hblazy_tensors_in_) {
        const auto& data = input.getDataPtr();
        inputs.push_back(data->tensor_data ? &*data->tensor_data : nullptr);
      }
      if (!native_plan_->matches_captured_inputs(i, inputs)) {
        native_plan_->reject("captured recipe input requires Lazy rebinding");
        break;
      }
    }
    for (const auto& graph : captured_graphs) {
      if (!graph->seed_tensors_generator_.empty() ||
          !graph->user_input_indices_.empty()) {
        native_plan_->reject("RNG or explicit input rebinding requires adapter");
      }
    }
    // Compile the Lazy adapter once. A value shared by adjacent recipes needs
    // one live binding guard, not a repeated recipe walk and mutex per use.
    // The backend plan continues to know only Tensor leases, never Lazy IR.
    std::unordered_map<const habana_lazy::Data*, size_t> boundary_slots;
    auto capture_boundary = [&](const auto& tensor, bool plain_input) {
      ++native_boundary_references_;
      const auto& data = tensor.getDataPtr();
      if (!data->tensor_data || (plain_input && data->stride_params.has_value())) {
        native_plan_->reject("unsupported captured Lazy boundary");
        return;
      }
      auto found = boundary_slots.find(data.get());
      if (found != boundary_slots.end()) {
        native_boundaries_[found->second].plain_input |= plain_input;
        return;
      }
      if (!native_plan_->capture_boundary(*data->tensor_data)) {
        native_plan_->reject("unsupported captured Lazy boundary");
        return;
      }
      boundary_slots.emplace(data.get(), native_boundaries_.size());
      native_boundaries_.push_back({data, plain_input});
    };
    for (const auto& graph : captured_graphs) {
      for (const auto& input : graph->hblazy_tensors_in_) capture_boundary(input, true);
      for (const auto& output : graph->hblazy_tensors_out_) capture_boundary(output, false);
    }
    native_boundary_values_.resize(native_boundaries_.size());
    native_plan_->seal(captured_graphs.size());
  }

  // Not enabling DS back once HPU graph detected
  /*if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }*/
}

void HPUGraph::destroy() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Clear all captured SingleHpuGraphs
  captured_graphs.clear();
  user_input_sizes_.clear();
  user_input_match_indices_.clear();
  hblazy_tensors_in_out_.clear();
  capturing_ = false;
  native_plan_.reset();
  native_boundaries_.clear();
  native_boundary_values_.clear();
  native_boundary_references_ = 0;
  adapter_replays_ = 0;
  last_replay_native_ = false;
  last_replay_reason_.clear();

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  if (context != nullptr) {
    /* Set graph capture mode off */
    context->setCapturing(false);
    context->setCaptureGraph(nullptr);
  }
}

void HPUGraph::mark_step() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!capturing_) {
    // need to start the capture.
    PT_DEVICE_DEBUG("GRAPH:: Use Graph capture to Begin the capture");
    return;
  }
  auto stream = c10::hpu::getCurrentHPUStream();
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  context->JoinPendingLaunchThread();
  if (context->getGraph()) {
    auto captured_graph = std::make_shared<SingleHPUGraph>(
        context->getRecipeArgSpec(),
        context->getGraph(),
        context->getInputs(),
        context->getOutputs(),
        context->getHbLazyTensors(),
        context->getUserInputIndices(),
        context->getSeedTensorMap(),
        context->getHash(),
        context->getGraphKey(),
        context->getOpStrs(),
        stream);
    captured_graphs.push_back(captured_graph);
    auto user_inp_match = context->getUserInputMatchIndices();
    user_input_match_indices_.insert(
        user_inp_match.begin(), user_inp_match.end());
    PT_IRGRAPH_DEBUG("GRAPH:: captured graph");
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured input size ", captured_graph->input_vals_.size());
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured output size ", captured_graph->output_vals_.size());
    PT_HPUGRAPH_DEBUG(
        "GRAPH:: captured hblazy_tensors_out_ size ",
        captured_graph->hblazy_tensors_out_.size());
  }
  context->getSeedTensorMap().clear();
  context->resetGraph();
}

/* Find all the lazy tensor id of inputs
 it should not be input of any subsequent graph
 for example graph 1 - input x, inplace op on x
 graph 2- input x
 graph 1- again in loop
 then x can only be deleted after the forward function has run completely
 and user doesn't need the tensor anymore */
void HPUGraph::clear_inputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (native_plan_) {
    native_plan_->reject("input lease release requested");
  }
  // call synchronize before this API
  for (auto out_tensor : hblazy_tensors_in_out_) {
    PT_HPUGRAPH_DEBUG(
        "Freeing input-output tensor with id: ",
        out_tensor.getTensorUniqueId());
    out_tensor.SetTensorDataNullOpt();
  }
}

void HPUGraph::replay(std::vector<at::Tensor>& inputs, bool async) {
  replay_impl(inputs, async, 1, false);
}

void HPUGraph::replay_many(size_t repetitions) {
  TORCH_CHECK(repetitions >= 1, "Replay repetitions must be positive");
  std::vector<at::Tensor> inputs;
  replay_impl(inputs, false, repetitions, true);
}

void HPUGraph::replay_impl(
    std::vector<at::Tensor>& inputs, bool async,
    size_t repetitions, bool require_native) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(
      !native_plan_ || native_plan_->state() != NativePlanState::Failed,
      "Native HPUGraph plan failed after submission; reset before replay");
  if (capturing_) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  ValidateMarkedInputs(user_input_sizes_, inputs);
  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replay");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }
  const bool requires_adapter = async || !inputs.empty();
  if (!requires_adapter && native_plan_ &&
      native_plan_->state() == NativePlanState::Ready) {
    bool bound = true;
    for (size_t slot=0; slot<native_boundaries_.size(); ++slot) {
      const auto& boundary = native_boundaries_[slot];
      const auto& data = boundary.data;
      if (!data->tensor_data || (boundary.plain_input && data->stride_params.has_value())) {
        bound = false;
        break;
      }
      native_boundary_values_[slot] = &*data->tensor_data;
    }
    if (bound) bound = native_plan_->matches_boundaries(native_boundary_values_);
    if (bound) {
      c10::StreamGuard stream_guard(capture_stream_.unwrap());
      if (native_plan_->replay(repetitions)) {
        last_replay_native_ = true;
        last_replay_reason_.clear();
        return;
      }
    } else {
      native_plan_->reject("Lazy tensor binding changed before replay");
    }
  }
  TORCH_CHECK(
      !require_native, "Repeated replay requires a valid fixed-address native plan: ",
      native_plan_ ? native_plan_->stats().reason : "graph has no captured plan");
  ++adapter_replays_;
  last_replay_native_ = false;
  last_replay_reason_ = requires_adapter
      ? "asynchronous or rebound replay requires adapter"
      : (native_plan_ ? native_plan_->stats().reason : "graph has no captured plan");
  for (size_t i = 0; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replay(inputs, async);
  }
}

void HPUGraph::replayV2(
    std::vector<at::Tensor>& static_inputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(
      !native_plan_ || native_plan_->state() != NativePlanState::Failed,
      "Native HPUGraph plan failed after submission; reset before replay");
  ++adapter_replays_;
  last_replay_native_ = false;
  last_replay_reason_ = "replayV2 requires input rebinding adapter";
  if (capturing_) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replayV2");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  // Use replaytV2 for the first captured graph, as the user input is
  // for the first captured graph
  if (captured_graphs.empty()) {
    return;
  }
  captured_graphs[0]->replayV2(static_inputs, inputs, async);

  for (size_t i = 1; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replay(inputs, async);
  }
}

std::unordered_set<int64_t> get_hb_base_tensor_id_list_if_view(
    const habana_lazy::HbLazyTensor& hl_t_in) {
  // handle multi level views
  auto hl_t = hl_t_in;
  std::unordered_set<int64_t> view_t_list;
  view_t_list.insert(hl_t.getTensorUniqueId());
  auto stride_params = hl_t.getDataPtr()->stride_params;
  while (stride_params.has_value()) {
    auto out = stride_params.value().base;
    hl_t = habana_lazy::GetHbLazyTensor(out, true, false);
    view_t_list.insert(hl_t.getTensorUniqueId());
    stride_params = hl_t.getDataPtr()->stride_params;
  }
  return view_t_list;
}

bool IsNodeSliceOrStridedInsert(const habana_lazy::ir::NodePtr& mp_node) {
  if (mp_node && !mp_node->is_control_edge()) {
    std::string_view node_name = mp_node->op().toQualString();
    return node_name == "hpu::slice_insert" ||
        node_name == "hpu::strided_insert";
  }
  return false;
}

void HPUGraph::mark_user_outputs(
    std::vector<at::Tensor>& outputs,
    bool free_inplace) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (native_plan_) {
    native_plan_->reject("output lease rewriting requires adapter");
  }
  PT_HPUGRAPH_DEBUG("mark_user_outputs with outputs size = ", outputs.size());
  if (capturing_) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  if (captured_graphs.empty()) {
    return;
  }

  for (size_t graphIdx = 0; graphIdx < captured_graphs.size(); graphIdx++) {
    auto single_graph = captured_graphs[graphIdx];
    HABANA_ASSERT(
        single_graph->prev_graph_interdep_out_t_list_.empty() &&
            single_graph->user_out_indices_tlist_.empty(),
        "Error: mark_user_outputs called more than once for the same graph?");
  }

  // Go over all captured SingleHPUGraphs
  for (size_t graphIdx = 0; graphIdx < captured_graphs.size(); graphIdx++) {
    auto& single_graph = captured_graphs[graphIdx];
    if (single_graph->graph_) {
      // This set shows have the indices of user_output tensors in
      // hblazy_tensors_out_
      std::unordered_set<size_t> user_out_tensors_idx_set;

      // Find all the lazy tensor id of inputs
      std::unordered_set<int64_t> input_lazyt_id_set;
      for (const auto& in_t : single_graph->hblazy_tensors_in_) {
        input_lazyt_id_set.emplace(in_t.getTensorUniqueId());
      }

      // Find the interdependant tensors
      for (size_t out_pos = 0; out_pos < outputs.size(); out_pos++) {
        auto user_out_hbl_t = habana_lazy::GetHbLazyTensor(outputs[out_pos]);
        const auto base_view_tids =
            get_hb_base_tensor_id_list_if_view(user_out_hbl_t);
        [[maybe_unused]] auto& ir_value = user_out_hbl_t.CurrentIrValue();
        for (size_t idx = 0; idx < single_graph->hblazy_tensors_out_.size();
             ++idx) {
          auto& out_tensor = single_graph->hblazy_tensors_out_[idx];
          auto tid = out_tensor.getTensorUniqueId();
          if (tid == user_out_hbl_t.getTensorUniqueId() ||
              base_view_tids.count(tid) != 0U) {
            // This is used for replay to match the user out tensor indices
            single_graph->user_out_indices_tlist_.emplace_back(out_pos, idx);
            // This is used later during replay
            user_out_tensors_idx_set.insert(idx);
          }
        }
      }

      // Go over all outputs from the current SingleHPUGraph
      for (size_t outIdx = 0; outIdx < single_graph->hblazy_tensors_out_.size();
           outIdx++) {
        auto& out_tensor = single_graph->hblazy_tensors_out_[outIdx];
        const auto tid = out_tensor.getTensorUniqueId();

        // exclude view tensors &  Inplace tensors
        bool is_view = out_tensor.getDataPtr()->stride_params.has_value() ||
            out_tensor.IsCollective();
        bool is_input = input_lazyt_id_set.count(tid) != 0U &&
            user_out_tensors_idx_set.count(outIdx) == 0U;

        if (is_input) {
          hblazy_tensors_in_out_.push_back(out_tensor);
          PT_BRIDGE_DEBUG(
              "Graph: ", graphIdx, " Input found for output id: ", tid);
        }

        // Exclude slice_insert/strided_insert nodes which will come as part of
        // inplace ops
        bool is_inplace = !free_inplace &&
            IsNodeSliceOrStridedInsert(
                single_graph->output_vals_[outIdx].mp_node);
        // Check if any of the following SingleHPUGraphs use this output as an
        // input
        size_t last_use = 0;
        bool is_interdependent = false;
        // if its not an useroutput or if its not inplace
        if (!is_input && !is_view && !is_inplace &&
            user_out_tensors_idx_set.count(outIdx) == 0U) {
          for (size_t j = graphIdx + 1; j < captured_graphs.size(); j++) {
            auto next_graph = captured_graphs[j];
            // Go over all the inputs in the following SingleHPUGraph
            for (auto& hbt_in : next_graph->hblazy_tensors_in_) {
              // Found a match
              if (hbt_in.getTensorUniqueId() == tid) {
                is_interdependent = true;
                last_use = j;
                break;
              }
            }
          }
        }

        // If this tensor is used within the subgraphs, note the
        // details, this will be used during replay
        if (last_use > 0) {
          captured_graphs[last_use]->prev_graph_interdep_out_t_list_.push_back(
              out_tensor);
        }

        // Can start freeing memory for output tensors that have no dependency.
        // SetHpuGraphOutTensor mark to false so that next replay it can be
        // freed Or if its an all_reduce output
        if (!is_input && !is_view && !is_inplace && !is_interdependent &&
            user_out_tensors_idx_set.count(outIdx) == 0U) {
          out_tensor.SetHpuGraphOutTensor(false);
          out_tensor.SetTensorDataNullOpt();
        }
      }

      // Remove the output vals (used to check if it's an inplace node or not)
      single_graph->output_vals_.clear();
    }
  }
}

void HPUGraph::replayV3(std::vector<at::Tensor>& inputs, bool async) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(
      !native_plan_ || native_plan_->state() != NativePlanState::Failed,
      "Native HPUGraph plan failed after submission; reset before replay");
  ++adapter_replays_;
  last_replay_native_ = false;
  last_replay_reason_ = "replayV3 requires input rebinding adapter";
  PT_HPUGRAPH_DEBUG(
      "replayV3 with inputs size = ", inputs.size(), " aysnc = ", async);
  if (capturing_) {
    // if capturing is in progress, replay is not allowed.
    PT_DEVICE_FATAL("GRAPH:: Capture in progress");
    return;
  }

  ValidateMarkedInputs(user_input_sizes_, inputs);
  PT_IRGRAPH_DEBUG("step marker due to HPUGraph::replayV3");
  if (async && GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD)) {
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, true);
  } else {
    habana_lazy::HbLazyTensor::StepMarker({});
  }

  if (captured_graphs.empty()) {
    return;
  }

  PT_HPUGRAPH_DEBUG("Total Number Of Captured Graph: ", captured_graphs.size());

  for (size_t i = 0; i < captured_graphs.size(); i++) {
    captured_graphs[i]->replayV3(inputs, async);
  }
}

void HPUGraph::mark_user_inputs(std::vector<at::Tensor>& static_inputs) {
  PT_LAZY_TRACE;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PT_HPUGRAPH_DEBUG(
      "mark_user_inputs with static_inputs size = ", static_inputs.size());
  if (!capturing_) {
    // if capturing is not in progress, mark_user_inputs is not allowed.
    PT_DEVICE_FATAL(
        "GRAPH:: mark_user_inputs must be while capturing in progress");
    return;
  }

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  context->setMarkedInputs(static_inputs);
  for (const auto& t : static_inputs) {
    user_input_sizes_.emplace_back(t.sizes().vec());
  }
}

HPUGraph::~HPUGraph() {
  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();
  if (context != nullptr) {
    /* Set graph capture mode off */
    context->setCapturing(false);
    context->setCaptureGraph(nullptr);
  }
}

NativeExecutionPlan* HPUGraph::native_capture_plan() {
  return capturing_ ? native_plan_.get() : nullptr;
}

NativeReplayStats HPUGraph::native_replay_stats() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto result = native_plan_ ? native_plan_->stats() : NativeReplayStats{};
  result.adapter_replays = adapter_replays_;
  result.last_replay_native = last_replay_native_;
  result.last_replay_reason = last_replay_reason_;
  result.boundary_references = native_boundary_references_;
  result.unique_boundaries = native_boundaries_.size();
  return result;
}

void HPUGraph::set_native_event_batching(bool enabled) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(native_plan_, "Graph has no native execution plan");
  native_plan_->set_event_batching(enabled);
}

void HPUGraph::set_native_compiled_regions(bool enabled) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  TORCH_CHECK(native_plan_, "Graph has no native plan");
  native_plan_->set_compiled_regions(enabled);
}

std::vector<NativeAccessDiagnostic> HPUGraph::native_dependency_diagnostics() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return native_plan_ ? native_plan_->dependency_diagnostics()
                      : std::vector<NativeAccessDiagnostic>{};
}

SingleHPUGraph::~SingleHPUGraph() {
  graph_.reset();
  input_vals_.clear();
  output_vals_.clear();
  hblazy_tensors_in_.clear();
  hblazy_tensors_out_.clear();
  prev_graph_interdep_out_t_list_.clear();
  user_out_indices_tlist_.clear();
  seed_tensors_generator_.clear();
}

void SingleHPUGraph::replayGraph(
    habana_lazy::ir::ValueList& input_vals,
    bool async) {
  PT_LAZY_TRACE;
  bool dynamic_env_ = habana_helpers::GetRefineDynamicShapeStatus();
  if (dynamic_env_) {
    habana_helpers::DisableRefineDynamicShape();
  }
  auto old_stream = c10::hpu::getCurrentHPUStream();
  c10::hpu::setCurrentHPUStream(capture_stream_);

  habana_lazy::HbExecutionContext* context =
      habana_lazy::get_device_lazy_execution_context();

  size_t launch_jobid = context->GetUniqueJobId();
  context->AddToJobidStreamidMap(launch_jobid, capture_stream_.stream());

  // set exec for input/output tensors
  bool queue_in_thread_pool = async &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_HPUGRAPH_THREAD) &&
      GET_ENV_FLAG_NEW(PT_HPU_QUEUE_SYNLAUNCHES);
  if (!queue_in_thread_pool) {
    std::unordered_set<size_t> in_uid;
    in_uid.reserve(hblazy_tensors_in_.size());
    for (const auto& t : hblazy_tensors_in_) {
      auto uid = t.getTensorUniqueId();
      in_uid.insert(uid);
      t.SetExecutionInProgress();
    }

    for (auto& t : hblazy_tensors_out_) {
      auto uid = t.getTensorUniqueId();
      if (in_uid.find(uid) == in_uid.end()) {
        t.SetTensorDataNullOpt();
      }
      t.SetExecutionInProgress();
    }
  } else {
    for (const auto& t : hblazy_tensors_in_) {
      t.SetExecutionInProgress();
    }

    for (const auto& t : hblazy_tensors_out_) {
      t.SetExecutionInProgress();
    }
  }

  std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh = nullptr;
  if (GET_ENV_FLAG_NEW(PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK)) {
    cached_rarg_psh = cached_rarg_psh_;
  }

  if (queue_in_thread_pool) {
    context->m_launch_thread_handle =
        habana_lazy::SingleTonExecThreadPool::getInstance().enqueue(
            habana_lazy::HbLazyTensor::ExecuteCachedGraph,
            cached_rarg_psh,
            graph_,
            hash_,
            graphKey_,
            opStrs_,
            hblazy_tensors_in_,
            hblazy_tensors_out_,
            prev_graph_interdep_out_t_list_,
            seed_tensors_generator_,
            launch_jobid,
            capture_stream_);
  } else {
    habana_lazy::HbLazyTensor::ExecuteCachedGraph(
        cached_rarg_psh,
        graph_,
        hash_,
        graphKey_,
        opStrs_,
        hblazy_tensors_in_,
        hblazy_tensors_out_,
        prev_graph_interdep_out_t_list_,
        seed_tensors_generator_,
        launch_jobid,
        capture_stream_);
  }

  auto num_inputs = input_vals.size();
  for (size_t i = 0; i < num_inputs; ++i) {
    if (user_input_indices_.count(i) > 0) {
      hblazy_tensors_in_[i] = habana_lazy::HbLazyTensor();
      input_vals_[i] = habana_lazy::ir::Value();
    }
  }

  c10::hpu::setCurrentHPUStream(old_stream);

  // Not enabling DS back once HPU graph detected
  /*if (dynamic_env_) {
    habana_helpers::EnableRefineDynamicShape();
  }*/
}

void SingleHPUGraph::replay(std::vector<at::Tensor>& inputs, bool async) {
  if (inputs.empty()) {
    if (graph_) {
      replayGraph(input_vals_, async);
    }
    return;
  }

  // Reuse the existing binding adapter for both HPU and CPU scale tensors.
  // The previous H2D-only condition silently skipped HPU-input execution.
  replayV3(inputs, async);
}

void SingleHPUGraph::replayV3(std::vector<at::Tensor>& inputs, bool async) {
  PT_HPUGRAPH_DEBUG(
      "In HPUGraph::replayV3 with ", inputs.size(), " input tensors");
  PT_DEVICE_DEBUG(graph_ ? (graph_->dump(), "") : "null graph");
  if (graph_) {
    auto num_inputs = input_vals_.size();
    for (size_t i = 0; i < num_inputs; ++i) {
      if (user_input_indices_.count(i) > 0) {
        auto t = inputs[user_input_indices_[i]];
        auto maybe_h2dt = habana_lazy::maybe_convert_tensor_to_h2d(
            t,
            (habana_helpers::is_h2d_scales_enabled() && t.is_cpu()),
            "HPUGraph");
        auto new_t = (maybe_h2dt.has_value()) ? maybe_h2dt.value() : t;

        auto hbl = habana_lazy::GetHbLazyTensor(new_t);
        if (!hbl.getDataPtr()->tensor_data) {
          auto& stride_params_opt = hbl.getDataPtr()->stride_params;
          if (stride_params_opt.has_value()) {
            hbl = habana_lazy::GetHbLazyTensor(
                habana_lazy::HbLazyTensorViews::HandleViewsD2H(t));
          } else {
            HABANA_ASSERT(
                0,
                "Neither storage attached to input tensor, not its view.",
                " Uniqueid: ",
                hbl.getDataPtr()->unique_id,
                " Strided Params Has Value: ",
                hbl.getDataPtr()->stride_params.has_value(),
                " Failing Graph: ",
                (graph_->dump(), ""));
          }
        }
        hblazy_tensors_in_[i] = hbl;
        input_vals_[i] = hbl.CurrentIrValue();
      }
    }
    replayGraph(input_vals_, async);
  }
}

void SingleHPUGraph::replayV2(
    std::vector<at::Tensor>& static_inputs,
    std::vector<at::Tensor>& inputs,
    bool async) {
  PT_HPUGRAPH_DEBUG(
      "In HPUGraph::replayV2 with ", inputs.size(), " input tensors");
  PT_DEVICE_DEBUG(graph_ ? (graph_->dump(), "") : "null graph");
  if (graph_) {
    habana_lazy::ir::ValueList input_val_list;
    std::vector<habana_lazy::HbLazyTensor> static_input_lazy_tensors;
    static_input_lazy_tensors.reserve(static_inputs.size());
    for (auto& t : static_inputs) {
      static_input_lazy_tensors.emplace_back(habana_lazy::GetHbLazyTensor(t));
    }
    // Input arguments:
    //  - New inputs are provided against Static inputs
    //    -> Replace the static inputs with the new inputs
    //  - No new inputs provided against static inputs (possibly weights)
    //    -> Reuse static inputs
    //  - Scalar inputs
    //    -> Reuse static scalar inputs
    // Extract the input value list from user provided input arguments
    HABANA_ASSERT(static_inputs.size() == inputs.size());
    std::transform(
        input_vals_.begin(),
        input_vals_.end(),
        std::back_inserter(input_val_list),
        [&static_input_lazy_tensors,
         &inputs](habana_lazy::ir::Value saved_ir_v_) {
          size_t idx = 0;
          for (auto& static_lazy_t : static_input_lazy_tensors) {
            // TBD: HPU graph needs to eliminate dependency on ir values
            auto& ir_value = static_lazy_t.CurrentIrValue();
            if (ir_value == saved_ir_v_) {
              return habana_lazy::GetHbLazyTensor(inputs[idx]).CurrentIrValue();
            }
            ++idx;
          }
          return saved_ir_v_;
        });
    replayGraph(input_val_list, async);
  }
}
} // namespace at::hpu
