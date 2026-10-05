// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::hpu {

// Allocation-level keys, never physical addresses. Views of an allocation are
// deliberately coarsened to one key: disjoint ranges may lose optimization,
// but a write cannot escape the dependency compiler through an alias.
struct NativeCommandAccess {
  std::vector<uint64_t> reads;
  std::vector<uint64_t> writes;
  bool compute = false;
  size_t collectives = 0;
  size_t external_events = 0;
  bool complete = true;
};

enum class NativeDependencyKind { Raw, War, Waw, CollectiveOrder };

struct NativeDependencyEdge {
  size_t producer;
  size_t consumer;
  uint64_t allocation;
  NativeDependencyKind kind;
};

struct NativeSerialRegion {
  uint64_t communicator;
  std::vector<size_t> collectives;
};

struct NativeDependencyPlan {
  std::vector<NativeDependencyEdge> edges;
  std::vector<size_t> serial_collectives;

  std::vector<NativeSerialRegion> serial_regions(
      const std::vector<NativeCommandAccess>& commands,
      const std::vector<uint64_t>& communicators) const {
    if (commands.size() != communicators.size()) {
      throw std::invalid_argument("collective communicator table size changed");
    }
    std::vector<bool> eligible(commands.size(), false);
    for (auto index : serial_collectives) {
      eligible.at(index) = true;
    }
    std::vector<NativeSerialRegion> result;
    NativeSerialRegion region{};
    auto finish = [&]() {
      if (region.collectives.size() > 1) {
        result.push_back(std::move(region));
      }
      region = {};
    };
    for (size_t i = 0; i < commands.size(); ++i) {
      const auto& command = commands[i];
      if (!command.complete || command.external_events ||
          (command.compute && command.collectives)) {
        finish();
        continue;
      }
      if (!command.collectives) {
        continue;
      }
      if (!eligible[i] || !communicators[i]) {
        finish();
        continue;
      }
      if (region.communicator && region.communicator != communicators[i]) {
        finish();
      }
      region.communicator = communicators[i];
      region.collectives.push_back(i);
    }
    finish();
    return result;
  }

  static NativeDependencyPlan compile(
      const std::vector<NativeCommandAccess>& commands) {
    NativeDependencyPlan result;
    struct State {
      size_t writer = SIZE_MAX;
      std::vector<size_t> readers;
    };
    std::unordered_map<uint64_t, State> allocations;
    size_t last_collective = SIZE_MAX;
    auto unique = [](std::vector<uint64_t> values) {
      values.erase(std::remove(values.begin(), values.end(), 0), values.end());
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
      return values;
    };
    auto edge = [&](size_t from, size_t to, uint64_t key,
                    NativeDependencyKind kind) {
      if (from != SIZE_MAX && from != to) {
        result.edges.push_back({from, to, key, kind});
      }
    };
    for (size_t i = 0; i < commands.size(); ++i) {
      const auto& command = commands[i];
      for (auto key : unique(command.reads)) {
        auto& state = allocations[key];
        edge(state.writer, i, key, NativeDependencyKind::Raw);
        state.readers.push_back(i);
      }
      for (auto key : unique(command.writes)) {
        auto& state = allocations[key];
        edge(state.writer, i, key, NativeDependencyKind::Waw);
        for (auto reader : state.readers) {
          edge(reader, i, key, NativeDependencyKind::War);
        }
        state.readers.clear();
        state.writer = i;
      }
      if (command.collectives) {
        // Preserve the existing ordered HCCL host queue. This edge is not a
        // Tensor dependency and must never be counted as useful data reuse.
        edge(last_collective, i, 0, NativeDependencyKind::CollectiveOrder);
        last_collective = i;
      }
    }
    auto raw = [&](size_t from, size_t to) {
      return std::any_of(result.edges.begin(), result.edges.end(),
                         [&](const auto& e) {
        return e.producer == from && e.consumer == to &&
            e.kind == NativeDependencyKind::Raw;
      });
    };
    for (size_t i = 1; i + 1 < commands.size(); ++i) {
      const auto& before = commands[i - 1];
      const auto& middle = commands[i];
      const auto& after = commands[i + 1];
      if (before.complete && middle.complete && after.complete &&
          before.compute && !before.collectives &&
          !middle.compute && middle.collectives == 1 &&
          after.compute && !after.collectives &&
          !before.external_events && !middle.external_events &&
          !after.external_events && raw(i - 1, i) && raw(i, i + 1)) {
        result.serial_collectives.push_back(i);
      }
    }
    return result;
  }
};

} // namespace at::hpu
