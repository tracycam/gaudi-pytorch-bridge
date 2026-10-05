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
    # Two outputs from one recipe share a producer event; a third output from
    # another stream must retain its distinct dependency. Exercise both paths.
    from probe_device_identity import verify_device
    for separate_producer in (False, True):
        host = torch.arange(128, dtype=torch.float32).reshape(8,16).remainder(4)
        x = host.to('hpu')
        ht.core.mark_step()
        identity = verify_device()
        producer_stream, extra_stream, consumer_stream = (ht.hpu.Stream() for _ in range(3))
        producer, extra, consumer = (ht.hpu.HPUGraph() for _ in range(3))
        try:
            with torch.inference_mode(), ht.hpu.stream(producer_stream):
                producer.capture_begin()
                a, b = x * 2, x * 3
                producer.capture_end()
            if separate_producer:
                with torch.inference_mode(), ht.hpu.stream(extra_stream):
                    extra.capture_begin()
                    c = x * 5
                    extra.capture_end()
            with torch.inference_mode(), ht.hpu.stream(consumer_stream):
                consumer.capture_begin()
                total = a + b
                if separate_producer:
                    total = total + c
                ht.core.mark_step()
                y = total * 4
                consumer.capture_end()
            for i in range(16):
                current = host + (i % 8)
                with torch.inference_mode():
                    x.copy_(current.to('hpu'))
                    ht.core.mark_step()
                    producer.replay()
                    if separate_producer:
                        extra.replay()
                    consumer.replay()
                    assert torch.equal(y.cpu(), current * (40 if separate_producer else 20))
            states = [_hpu_C.native_replay_stats(g.hpu_graph)
                      for g in ([producer, extra, consumer] if separate_producer else [producer, consumer])]
            assert all(s['ready'] and s['replays']==16 for s in states), states
            records.append(dict(case='multiple-producers' if separate_producer else 'shared-producer-event',
                                checks=16, native=states, device_identity=identity))
        finally:
            ht.hpu.synchronize()
            producer.reset()
            extra.reset()
            consumer.reset()
    result = dict(status="PASS", scope="Functional lifecycle checks; not TPS", records=records)
    (Path(os.environ["PROBE_OUT"]) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
