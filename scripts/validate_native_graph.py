#!/usr/bin/env python3
"""Normal HPUGraph API, two connected recipes, changing input and consumer.

Run through the existing owned single-device supervisor. This functional probe
is deliberately not a model-TPS or kernel-bandwidth benchmark.
"""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--timing-replays", type=int, default=0,
                        help="Optional hot-replay diagnostic, not model TPS")
    args = parser.parse_args()
    if args.iterations < 1 or args.timing_replays < 0:
        parser.error("iterations must be positive and timing-replays nonnegative")
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C

    out = Path(os.environ["PROBE_OUT"])
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        host = torch.arange(4096, dtype=torch.float32).reshape(64, 64).remainder(4).to(dtype)
        x = host.to("hpu")
        ht.core.mark_step()
        ht.hpu.synchronize()
        stream = ht.hpu.Stream()
        graph = ht.hpu.HPUGraph()
        with torch.inference_mode(), ht.hpu.stream(stream):
            graph.capture_begin(dry_run=args.dry_run)
            intermediate = x * 2 + 1
            ht.core.mark_step()
            y = intermediate * 3 + 5
            graph.capture_end()
        for i in range(args.iterations):
            current = host + (i % 8)
            with torch.inference_mode(), ht.hpu.stream(stream):
                x.copy_(current.to("hpu"))
                ht.core.mark_step()
                graph.replay()
                consumer = y + 7
                ht.core.mark_step()
                actual = consumer.cpu()
            expected = ((current * 2 + 1) * 3 + 5) + 7
            if not torch.equal(actual, expected):
                torch.save(dict(actual=actual, expected=expected), out / f"failure-{dtype}-{i}.pt")
                raise AssertionError(f"Changing-input/consumer mismatch: {dtype}, iteration {i}")
        stats = (_hpu_C.native_replay_stats(graph.hpu_graph)
                 if hasattr(_hpu_C, "native_replay_stats") else None)
        if args.require_native and (not stats or stats.get("replays", 0) != args.iterations):
            raise AssertionError(f"Native path not used for every replay: {stats}")
        timing = None
        if args.timing_replays:
            ht.hpu.synchronize()
            with torch.inference_mode(), ht.hpu.stream(stream):
                begin = time.perf_counter_ns()
                for _ in range(args.timing_replays):
                    graph.replay()
                submitted = time.perf_counter_ns()
                ht.hpu.synchronize()
                completed = time.perf_counter_ns()
            timing = dict(replays=args.timing_replays,
                          enqueue_us=(submitted - begin) / 1e3 / args.timing_replays,
                          complete_us=(completed - begin) / 1e3 / args.timing_replays,
                          scope="Two small recipes; not model TPS")
            if hasattr(_hpu_C, "native_replay_stats"):
                stats = _hpu_C.native_replay_stats(graph.hpu_graph)
            if args.require_native and stats.get("replays", 0) != args.iterations + args.timing_replays:
                raise AssertionError(f"Native timing path fell back: {stats}")
        ht.hpu.synchronize()
        graph.reset()
        records.append(dict(dtype=str(dtype), checks=args.iterations, native=stats,
                            replay_timing=timing))
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "libhabana_pytorch" in line or "libe1_" in line or "libgkg" in line})
    if any("libe1_" in path or "libgkg" in path for path in mapped):
        raise AssertionError("External executor loaded")
    result = dict(status="PASS", scope="Two-recipe functional probe, not performance qualification",
                  torch=torch.__version__, bridge=ht.__file__, dry_run=args.dry_run,
                  records=records, loaded_libraries=mapped)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
