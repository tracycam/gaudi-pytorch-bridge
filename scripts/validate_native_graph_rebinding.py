#!/usr/bin/env python3
"""A captured input moves to another plan-owned storage: reject before replay."""
import json
import os
from pathlib import Path


def main():
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    x = torch.full((64,), 2., device='hpu')
    z = torch.full((64,), 7., device='hpu')
    ht.core.mark_step()
    ht.hpu.synchronize()
    graph = ht.hpu.HPUGraph()
    result = None
    try:
        with torch.inference_mode():
            graph.capture_begin()
            intermediate = x * 2 + z * 3
            ht.core.mark_step()
            y = intermediate + 5
            graph.capture_end()
            before = _hpu_C.native_replay_stats(graph.hpu_graph)
            if not before['ready']:
                raise AssertionError(f'initial plan not ready: {before}')
            graph.replay()
            if not torch.equal(y.cpu(), torch.full((64,), 30.)):
                raise AssertionError('initial result')
            # z is a valid retained owner, but must never be accepted in x's slot.
            x.set_(z)
            ht.core.mark_step()
            graph.replay()
            actual = y.cpu()
            expected = torch.full((64,), 40.)
            after = _hpu_C.native_replay_stats(graph.hpu_graph)
            if after['ready'] or after['replays'] != 1 or 'binding changed' not in after['reason']:
                raise AssertionError(f'changed input did not reject before submission: {after}')
            if not torch.equal(actual, expected):
                torch.save(dict(actual=actual, expected=expected), Path(os.environ['PROBE_OUT'])/'failure.pt')
                raise AssertionError('fallback after rebinding mismatch')
        result = dict(status='PASS', before=before, after=after,
                      scope='Exact-slot rejection and ordinary fallback, not fast rebinding support')
    except RuntimeError as error:
        if 'This function should not be called in lazy flow' not in str(error):
            raise
        result = dict(status='UNSUPPORTED', reason=str(error),
                      scope='Lazy set_ cannot exercise wrong-slot rebinding; gate remains unverified')
    finally:
        # Release the graph while the bridge device context is still live,
        # including when an unsupported mutation raises before replay.
        ht.hpu.synchronize()
        graph.reset()
    (Path(os.environ['PROBE_OUT'])/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
