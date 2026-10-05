#!/usr/bin/env python3
"""CPU-only profiler needs no hardware source; then verify bridge replay traces."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-replay', action='store_true')
    parser.add_argument('--hardware-trace', action='store_true',
                        help='After CPU validation, explicitly collect HPU activity for 16 replays')
    args = parser.parse_args()
    if args.hardware_trace and not args.device_replay:
        parser.error('--hardware-trace requires --device-replay')
    # Importing the bridge sets its normal profile_api[_light] marker. Reject
    # an externally configured hardware profile before that initialization.
    assert not os.getenv('HABANA_PROFILE'), 'probe must start without a hardware profile configuration'
    import torch
    import habana_frameworks.torch as ht
    from habana_frameworks.torch import _hpu_C
    from habana_frameworks.torch.activity_profiler.activity_profiler import DebugActivity
    from importlib.metadata import version
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
    record = dict(status='PASS', cpu_event_count=len(cpu_events), hardware_source=False,
                  bridge_version=version('habana-torch-plugin'),
                  hpu_initialized_before_cpu_trace=initially_initialized,
                  hpu_initialized_after_cpu_trace=torch.hpu.is_initialized(),
                  profile_api_marker=os.getenv('HABANA_PROFILE'))
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
                        with torch.profiler.record_function('HPUGraph.replay'):
                            graph.replay()
                assert torch.equal(y.cpu(), torch.full((8, 8), 14.))
            profile.export_chrome_trace(str(output/'bridge-replay-cpu.json'))
            events = json.loads((output/'bridge-replay-cpu.json').read_text())['traceEvents']
            launches = [e for e in events if 'Launch' in e.get('name', '')]
            # Bridge source spans use B/E pairs; inspect the actual schema below.
            assert launches, 'CPU profiler did not collect bridge submission-thread calls'
            record.update(bridge_event_count=len(events), launch_events=len(launches),
                native=_hpu_C.native_replay_stats(graph.hpu_graph))
            if args.hardware_trace:
                ht.hpu.synchronize()
                with torch.inference_mode(), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.HPU],
                        debug_activities=[DebugActivity.BRIDGE_FUNCTION_CALLS]) as profile:
                    for _ in range(16):
                        with torch.profiler.record_function('HPUGraph.replay'):
                            graph.replay()
                    ht.hpu.synchronize()
                assert torch.equal(y.cpu(), torch.full((8, 8), 14.))
                trace = output/'bridge-replay-device.json'
                profile.export_chrome_trace(str(trace))
                record.update(hardware_trace=str(trace),
                              after_hardware_trace=_hpu_C.native_replay_stats(graph.hpu_graph),
                              hardware_trace_scope='Explicit diagnostic after CPU-only validation; '
                                                   'profiled durations are not performance results')
        finally:
            ht.hpu.synchronize()
            graph.reset()
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
