#!/usr/bin/env python3
"""CPU-only profiler needs no hardware source; then verify bridge replay traces."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-replay', action='store_true')
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    from habana_frameworks.torch.activity_profiler.activity_profiler import DebugActivity
    assert not os.getenv('HABANA_PROFILE'), 'CPU-only probe must not enable hardware profiling'
    output = Path(os.environ.get('PROBE_OUT', '.'))
    initially_initialized = torch.hpu.is_initialized()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                               debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]) as profile:
        result = (torch.ones(8) + 2).sum().item()
    assert result == 24
    assert torch.hpu.is_initialized() == initially_initialized, 'CPU trace initialized a device'
    profile.export_chrome_trace(str(output/'cpu-only.json'))
    cpu_events = json.loads((output/'cpu-only.json').read_text())['traceEvents']
    assert any(e.get('name') == 'aten::add' for e in cpu_events)
    record = dict(status='PASS', cpu_event_count=len(cpu_events), hardware_source=False)
    if args.device_replay:
        x = torch.ones((8, 8), device='hpu')
        ht.core.mark_step()
        graph = ht.hpu.HPUGraph()
        try:
            with torch.inference_mode():
                graph.capture_begin()
                middle = x * 2 + 1
                ht.core.mark_step()
                y = middle * 3 + 5
                graph.capture_end()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                        debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]) as profile:
                    for _ in range(16):
                        graph.replay()
                assert torch.equal(y.cpu(), torch.full((8, 8), 14.))
            profile.export_chrome_trace(str(output/'bridge-replay-cpu.json'))
            events = json.loads((output/'bridge-replay-cpu.json').read_text())['traceEvents']
            launches = [e for e in events if 'Launch' in e.get('name', '')]
            # Bridge source spans use B/E pairs; inspect the actual schema below.
            assert launches, 'CPU profiler did not collect bridge submission-thread calls'
            record.update(bridge_event_count=len(events), launch_events=len(launches),
                native=_hpu_C.native_replay_stats(graph.hpu_graph))
        finally:
            ht.hpu.synchronize()
            graph.reset()
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
