# SPDX-License-Identifier: Apache-2.0
"""A consumer checks lowered dtype, not merely the frontend tensor dtype."""
import pytest
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as hc
from common_test import is_lazy, load_library


@pytest.mark.skipif(not is_lazy, reason="Lazy lowering regression")
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("destination, scalar_type", [(torch.int32, 3), (torch.int64, 4)])
def test_identity_cast_preserves_consumer_metadata(capture, destination, scalar_type):
    load_library(is_lazy, False)

    class Program(torch.nn.Module):
        def forward(self, scores):
            probabilities = scores.sigmoid()
            ids = probabilities.topk(8, dim=-1, sorted=False).indices
            gathered = probabilities.gather(1, ids)
            source = ids if destination == torch.int32 else ids.to(torch.int32)
            converted = source.to(destination).clone()
            checked = torch.ops.custom_op.checked_identity(converted, scalar_type)
            return checked, gathered

    with torch.inference_mode():
        scores = (torch.arange(384, dtype=torch.float32).expand(16, 384) / 384).to("hpu")
        hc.mark_step()
        torch.hpu.synchronize()
        program = Program()
        if capture:
            program = ht.hpu.wrap_in_hpu_graph(program, asynchronous=False)
        for _ in range(3):
            actual, gathered = program(scores)
            hc.mark_step()
            torch.hpu.synchronize()
            expected = scores.cpu().sigmoid().topk(8, dim=-1, sorted=False).indices
            assert actual.dtype == destination
            assert torch.equal(actual.cpu().sort(-1).values.long(), expected.sort(-1).values)
            assert gathered.cpu().isfinite().all()
