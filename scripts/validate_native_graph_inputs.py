#!/usr/bin/env python3
"""Public GPU input rebinding, missing/shape preflight, reset and recapture."""
import json
import os
from pathlib import Path


def main():
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        graph = ht.hpu.HPUGraph()
        try:
            def capture(shape):
                x = torch.full(shape, 2., dtype=dtype, device='hpu')
                ht.core.mark_step()
                graph.capture_begin()
                graph.mark_user_inputs([x])
                middle = x * 2 + 1
                ht.core.mark_step()
                y = middle * 3 + 5
                graph.capture_end()
                return y
            with torch.inference_mode():
                y = capture((8,8))
                # Missing/invalid inputs must fail before the first subgraph.
                for operation in (lambda: graph.replay(),
                                  lambda: graph.replay_with_inputs([torch.ones((4,16),device='hpu',dtype=dtype)])):
                    try:
                        operation()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('invalid marked input was not rejected')
                for value in range(8):
                    x_new = torch.full((8,8), float(value), dtype=dtype, device='hpu')
                    ht.core.mark_step()
                    graph.replay_with_inputs([x_new])
                    actual = y.cpu()
                    expected = torch.full((8,8), (value*2+1)*3+5, dtype=dtype)
                    if not torch.equal(actual, expected):
                        torch.save(dict(actual=actual, expected=expected),
                                   Path(os.environ['PROBE_OUT'])/f'failure-{dtype}-{value}.pt')
                        raise AssertionError(f'GPU input replay mismatch: {dtype}, {value}')
                ht.hpu.synchronize()
                graph.reset()
                y = capture((4,16))
                for value in (9, 12):
                    x_new = torch.full((4,16), float(value), dtype=dtype, device='hpu')
                    ht.core.mark_step()
                    graph.replay_with_inputs([x_new])
                    if not torch.equal(y.cpu(),torch.full((4,16),(value*2+1)*3+5,dtype=dtype)):
                        raise AssertionError('reset/recapture input metadata mismatch')
                stats = _hpu_C.native_replay_stats(graph.hpu_graph)
                if stats['replays'] != 0 or stats['ready']:
                    raise AssertionError('Binding adapter must not be mislabeled as native fast rebinding')
                records.append(dict(dtype=str(dtype), checks=12, native=stats))
        finally:
            ht.hpu.synchronize()
            graph.reset()
    result = dict(status='PASS', records=records,
                  scope='Public rebinding correctness and preflight; native fast rebinding remains separate')
    (Path(os.environ['PROBE_OUT'])/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
