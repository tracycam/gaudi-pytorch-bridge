#!/usr/bin/env python3
"""Normal model wrapper, changing input, actual matmul, no private replay API."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-native', action='store_true')
    parser.add_argument('--timing-replays', type=int, default=128)
    args = parser.parse_args()
    if args.timing_replays < 1:
        parser.error('timing-replays must be positive')
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C

    class Projection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer('weight', weight)

        def forward(self, x):
            y = x @ self.weight
            ht.core.mark_step()
            return y * 2 + 1

    out = Path(os.environ['PROBE_OUT'])
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for rows in (1, 2, 8, 64, 512):
            # Small integer operands avoid a change of precision contract in
            # this execution-path test. This is not a GEMM throughput study.
            host = torch.arange(rows * 256).reshape(rows, 256).remainder(3).to(dtype)
            weight = torch.arange(256 * 128).reshape(256, 128).remainder(2).to(dtype)
            model = Projection(weight.to('hpu')).eval()
            ht.core.mark_step()
            ht.hpu.synchronize()
            wrapped = ht.hpu.wrap_in_hpu_graph(model, asynchronous=False,
                                               disable_tensor_cache=False)
            with torch.inference_mode():
                wrapped(host.to('hpu'))
                for index in range(8):
                    current = host + index % 2
                    actual = wrapped(current.to('hpu')).cpu()
                    expected = (current.float() @ weight.float()).to(dtype) * 2 + 1
                    if not torch.equal(actual, expected):
                        torch.save(dict(actual=actual, expected=expected),
                                   out / f'failure-{dtype}-{rows}-{index}.pt')
                        raise AssertionError(f'wrapper mismatch: {dtype}, M={rows}, i={index}')
                static = host.to('hpu')
                ht.core.mark_step()
                ht.hpu.synchronize()
                begin = time.perf_counter_ns()
                for _ in range(args.timing_replays):
                    wrapped(static)
                submitted = time.perf_counter_ns()
                ht.hpu.synchronize()
                completed = time.perf_counter_ns()
            # Inspection is diagnostic only, after timing; no replacement of
            # the wrapped forward and no instrumentation in the hot path.
            cache = inspect.getclosurevars(wrapped.forward).nonlocals['cache']
            plans = [_hpu_C.native_replay_stats(value.graph.hpu_graph)
                     if hasattr(_hpu_C, 'native_replay_stats') else None
                     for value in cache.values()]
            if args.require_native and (not plans or any(
                    not plan or not plan['ready'] or plan['replays'] < args.timing_replays + 8
                    for plan in plans)):
                raise AssertionError(f'wrapper failed to use retained replay: {plans}')
            records.append(dict(dtype=str(dtype), rows=rows, checks=8, plans=plans,
                                enqueue_us=(submitted-begin)/1e3/args.timing_replays,
                                complete_us=(completed-begin)/1e3/args.timing_replays))
            wrapped.clear_cache()
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if 'libhabana_pytorch' in line or 'libe1_' in line or 'libgkg' in line})
    if any('libe1_' in path or 'libgkg' in path for path in mapped):
        raise AssertionError('External executor loaded')
    result = dict(status='PASS', records=records, loaded_libraries=mapped,
                  scope='Normal wrapped matmul functional/hot-forward probe; not model TPS')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
