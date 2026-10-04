#!/usr/bin/env python3
"""Normal API checks for stream dependencies, lowered views and async reset."""

import json
import os
from pathlib import Path


def main():
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C

    records = []
    for case in ("cross-stream", "transpose", "offset", "churn", "reset"):
        host = torch.arange(128, dtype=torch.float32).reshape(8, 16).remainder(4)
        base = host.to("hpu")
        ht.core.mark_step()
        ht.hpu.synchronize()
        x = base.T if case == "transpose" else base[2:6] if case == "offset" else base
        capture_stream, producer_stream = ht.hpu.Stream(), ht.hpu.Stream()
        graph = ht.hpu.HPUGraph()
        with torch.inference_mode(), ht.hpu.stream(capture_stream):
            graph.capture_begin()
            z = x * 2 + 1
            ht.core.mark_step()
            y = z * 3 + 5
            graph.capture_end()
        for i in range(16):
            current = host + (i % 8)
            stream = producer_stream if case == "cross-stream" else capture_stream
            with torch.inference_mode(), ht.hpu.stream(stream):
                if case == "churn":
                    garbage = torch.empty((65536,), device="hpu") + i
                    ht.core.mark_step()
                    del garbage
                base.copy_(current.to("hpu"))
                ht.core.mark_step()
                graph.replay()
                if case == "reset" and i == 15:
                    stats = _hpu_C.native_replay_stats(graph.hpu_graph)
                    # No explicit synchronize before releasing the graph owner.
                    graph.reset()
                actual = (y + 7).cpu()
            sliced = current.T if case == "transpose" else current[2:6] if case == "offset" else current
            expected = ((sliced * 2 + 1) * 3 + 5) + 7
            if not torch.equal(actual, expected):
                torch.save(dict(actual=actual, expected=expected),
                           Path(os.environ["PROBE_OUT"]) / f"failure-{case}-{i}.pt")
                raise AssertionError(f"{case}, iteration {i}")
        if case != "reset":
            stats = _hpu_C.native_replay_stats(graph.hpu_graph)
        if case in ("cross-stream", "churn", "reset") and stats["replays"] != 16:
            raise AssertionError(f"Expected native replay in {case}: {stats}")
        ht.hpu.synchronize()
        graph.reset()
        records.append(dict(case=case, checks=16, native=stats))
    result = dict(status="PASS", scope="Functional lifecycle checks; not TPS", records=records)
    (Path(os.environ["PROBE_OUT"]) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
