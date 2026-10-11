# SPDX-License-Identifier: Apache-2.0
"""A logical in-place mutation must survive repeated graph replay.

INT64 add_ can replace storage during Lazy lowering. This must either have a
binding adapter or be rejected before native submission, never silently reuse
the original recipe input. Read-only INT64 inputs and explicit copy_ stores
remain eligible for fixed-address replay.
"""
import pytest
import torch
import habana_frameworks.torch as ht
from habana_frameworks.torch import _hpu_C


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("store", ["add_", "copy_", "read-only"])
@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_mutation(dtype, store, steps, asynchronous):
    with torch.inference_mode():
        state = torch.tensor([126], dtype=dtype, device="hpu")
        ht.core.mark_step()
        ht.hpu.synchronize()
        graph = ht.hpu.HPUGraph()
        try:
            graph.capture_begin()
            for _ in range(steps):
                if store == "add_":
                    state.add_(1)
                elif store == "copy_":
                    state.copy_(state + 1)
                output = state + 0
                ht.core.mark_step()
            graph.capture_end()
            ht.hpu.synchronize()
            state.copy_(torch.tensor([126], dtype=dtype, device="hpu"))
            ht.core.mark_step()
            ht.hpu.synchronize()
            for replay in range(1, 4):
                graph.replay(asynchronous=asynchronous)
                ht.hpu.synchronize()
                expected = 126 if store == "read-only" else 126 + steps * replay
                assert state.cpu().item() == expected
                assert output.cpu().item() == expected
            stats = _hpu_C.native_replay_stats(graph.hpu_graph)
            if not asynchronous and store in ("copy_", "read-only"):
                assert stats["ready"], stats
                assert stats["replays"] == 3, stats
            assert not stats["failed"], stats
        finally:
            graph.reset()


@pytest.mark.parametrize("repetitions", [1, 4, 16])
@pytest.mark.parametrize("eos", [-1, 129])
def test_replay_many_persistent_feedback(repetitions, eos):
    with torch.inference_mode():
        position = torch.tensor([126], dtype=torch.int64, device="hpu")
        iteration = torch.tensor([0], dtype=torch.int64, device="hpu")
        active = torch.tensor([True], device="hpu")
        history = torch.full((64,), -1, dtype=torch.int64, device="hpu")
        ht.core.mark_step()
        ht.hpu.synchronize()
        graph = ht.hpu.HPUGraph()
        try:
            graph.capture_begin()
            token = position + 1
            emitted = torch.where(active, token, -torch.ones_like(token))
            history.copy_(history.scatter(0, iteration, emitted))
            position.copy_(position + active.to(torch.int64))
            iteration.copy_(iteration + 1)
            active.copy_(active & (token != eos))
            ht.core.mark_step()
            graph.capture_end()
            ht.hpu.synchronize()
            for value, source in ((position, [126]), (iteration, [0]), (active, [True])):
                value.copy_(torch.tensor(source, dtype=value.dtype, device="hpu"))
            history.copy_(torch.full((64,), -1, dtype=torch.int64, device="hpu"))
            ht.core.mark_step()
            ht.hpu.synchronize()
            for block in range(1, 4):
                graph.replay_many(repetitions)
                ht.hpu.synchronize()
                steps = block * repetitions
                valid = min(steps, 3) if eos == 129 else steps
                assert position.cpu().item() == 126 + valid
                assert iteration.cpu().item() == steps
                expected = list(range(127, 127 + valid)) + [-1] * (64 - valid)
                assert history.cpu().tolist() == expected
                stats = graph.execution_info()
                assert stats["ready"] and not stats["failed"], stats
                assert stats["replays"] == steps, stats
        finally:
            graph.reset()


def test_replay_many_rejects_rebinding_before_submission():
    with torch.inference_mode():
        state = torch.tensor([126], dtype=torch.int64, device="hpu")
        ht.core.mark_step()
        graph = ht.hpu.HPUGraph()
        try:
            graph.capture_begin()
            state.add_(1)
            ht.core.mark_step()
            graph.capture_end()
            ht.hpu.synchronize()
            state.copy_(torch.tensor([126], dtype=torch.int64, device="hpu"))
            ht.core.mark_step()
            ht.hpu.synchronize()
            with pytest.raises(RuntimeError, match="requires a valid fixed-address native plan"):
                graph.replay_many(4)
            assert state.cpu().item() == 126
            assert graph.execution_info()["replays"] == 0
            graph.replay()
            ht.hpu.synchronize()
            assert state.cpu().item() == 127
        finally:
            graph.reset()


@pytest.mark.parametrize("repetitions", [0, -1, 17, True, 1.5])
def test_replay_many_invalid_count(repetitions):
    graph = ht.hpu.HPUGraph()
    try:
        with pytest.raises(ValueError, match="integer in 1..16"):
            graph.replay_many(repetitions)
    finally:
        graph.reset()
