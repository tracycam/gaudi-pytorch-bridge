#!/usr/bin/env python3
"""Normal graph replay versus recipe/binding count; paired runs, not model TPS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def counts(value):
    result = tuple(int(item) for item in value.split(','))
    if not result or any(item < 1 or item > 64 for item in result):
        raise argparse.ArgumentTypeError('counts must be 1..64')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--recipes', type=counts, default=(1, 8, 32))
    parser.add_argument('--inputs', type=counts, default=(1, 8, 32))
    parser.add_argument('--replays', type=int, default=256)
    parser.add_argument('--require-native', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.replays <= 4096:
        parser.error('replays must be 1..4096')
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    from importlib.metadata import version
    stats = getattr(_hpu_C, 'native_replay_stats', None)
    if args.require_native and stats is None:
        parser.error('--require-native needs candidate bridge diagnostics')
    output = Path(os.environ['PROBE_OUT'])
    result = dict(status='RUNNING', torch=torch.__version__,
                  bridge_version=version('habana-torch-plugin'), cases=[],
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_diagnostics_available=stats is not None,
                  scope='Hot static graph submission scalability; not model TPS or HBM bandwidth')
    def save():
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    for dtype in (torch.float32, torch.bfloat16):
        for inputs in args.inputs:
            for recipes in args.recipes:
                tensors = [torch.zeros((64, 64), dtype=dtype, device='hpu')
                           for _ in range(inputs)]
                state = torch.zeros_like(tensors[0])
                if 'device_identity' not in result:
                    from probe_device_identity import verify_device
                    result['device_identity'] = verify_device()
                ht.core.mark_step()
                ht.hpu.synchronize()
                graph = ht.hpu.HPUGraph()
                try:
                    with torch.inference_mode():
                        graph.capture_begin()
                        for step in range(recipes):
                            for tensor in tensors:
                                state = state + tensor
                            if step + 1 < recipes:
                                ht.core.mark_step()
                        graph.capture_end()
                        # Only one input contributes, so all intermediate values
                        # are exactly representable even for BF16/64 recipes.
                        for value in (1, 0, 1, 0):
                            tensors[0].fill_(value)
                            ht.core.mark_step()
                            graph.replay()
                            expected = torch.full((64,64), recipes*value, dtype=dtype)
                            assert torch.equal(state.cpu(), expected)
                        # Upstream and candidate run the exact same workload.
                        # Candidate-only diagnostics are outside timed replay;
                        # their absence must not reject the upstream control.
                        before = stats(graph.hpu_graph) if stats is not None else None
                        if args.require_native:
                            assert before['ready'] and before['commands'] == recipes, before
                        samples = []
                        for _ in range(3):
                            ht.hpu.synchronize()
                            begin = time.perf_counter_ns()
                            for _ in range(args.replays):
                                graph.replay()
                            submitted = time.perf_counter_ns()
                            ht.hpu.synchronize()
                            completed = time.perf_counter_ns()
                            samples.append(dict(enqueue_us=(submitted-begin)/1e3/args.replays,
                                                complete_us=(completed-begin)/1e3/args.replays))
                    after = stats(graph.hpu_graph) if stats is not None else None
                    if args.require_native:
                        assert after['replays'] == 4 + 3*args.replays, after
                    result['cases'].append(dict(dtype=str(dtype), inputs=inputs, recipes=recipes,
                        checks=4, before=before, after=after, samples=samples,
                        complete_us=statistics.median(s['complete_us'] for s in samples)))
                    save()
                finally:
                    ht.hpu.synchronize()
                    graph.reset()
    result['status'] = 'PASS'
    save()
    print(json.dumps(dict(status='PASS', cases=len(result['cases']),
                         bridge_version=result['bridge_version'])), flush=True)


if __name__ == '__main__':
    main()
