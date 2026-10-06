#!/usr/bin/env python3
"""Local pretrained Transformers forward through ordinary HPUGraph APIs.

Run with the owned single-device supervisor. No vLLM, model-specific kernels,
private replay interface or extra executor. This measures full-forward graph
completion, not serving decode TPS, KV-cached generation or answer quality.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time


def dimensions(text):
    values = tuple(int(value) for value in text.split(','))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError('dimensions must be positive')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--batches', type=dimensions, default=(1, 2))
    parser.add_argument('--lengths', type=dimensions, default=(1, 8, 128))
    parser.add_argument('--replays', type=int, default=16)
    parser.add_argument('--use-input-cache', action='store_true')
    parser.add_argument('--require-native', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.replays <= 64 or max(args.batches) > 8 or max(args.lengths) > 2048:
        parser.error('bounded batch<=8, length<=2048, replays<=64 required')
    import torch
    import habana_frameworks.torch as ht
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(os.environ['PROBE_OUT'])
    result = dict(status='RUNNING', bridge_version=version('habana-torch-plugin'),
                  torch=torch.__version__, model=str(args.model.resolve()), cases=[],
                  use_input_cache=args.use_input_cache,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Full trained-model forward, no KV cache; not service/decode TPS')
    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation='eager').eval().to('hpu')
    result['architecture'] = type(model).__name__
    result['parameters'] = sum(parameter.numel() for parameter in model.parameters())
    encoded = tokenizer('The purpose of this experiment is to verify reusable graph execution. ',
                        add_special_tokens=False)['input_ids']
    ht.core.mark_step(); ht.hpu.synchronize()

    def forward(ids, positions):
        return model(input_ids=ids, position_ids=positions, use_cache=False).logits

    with torch.inference_mode():
        for batch in args.batches:
            for length in args.lengths:
                host_ids = torch.tensor((encoded*((length+len(encoded)-1)//len(encoded)))[:length],
                                        dtype=torch.int64).repeat(batch, 1)
                host_positions = torch.arange(length, dtype=torch.int64).repeat(batch, 1)
                if args.use_input_cache:
                    buffers = ht.hpu.InputBufferCache('hpu', max_entries=2)
                    ids = buffers.copy('ids', host_ids)
                    positions = buffers.copy('positions', host_positions)
                else:
                    ids, positions = host_ids.to('hpu'), host_positions.to('hpu')
                ht.core.mark_step(); ht.hpu.synchronize()
                forward(ids, positions).cpu()  # Materialize the ordinary program before capture.
                graph = ht.hpu.HPUGraph()
                try:
                    graph.capture_begin()
                    output = forward(ids, positions)
                    graph.capture_end()
                    for step in range(4):
                        current = host_ids.roll(step, dims=1)
                        if length == 1:
                            current = current + step
                        if args.use_input_cache:
                            assert buffers.copy('ids', current) is ids
                            assert buffers.copy('positions', host_positions) is positions
                        else:
                            torch._foreach_copy_((ids, positions), (current, host_positions), non_blocking=True)
                        ht.core.mark_step()
                        graph.replay()
                        actual = output.cpu()
                        expected = forward(ids, positions).cpu()
                        # A captured replay must preserve the same HPU arithmetic.
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if not bool(torch.isfinite(actual).all()):
                            raise AssertionError('nonfinite pretrained model output')
                    samples = []
                    for arm in ('eager', 'graph', 'graph', 'eager'):
                        ht.hpu.synchronize()
                        began = time.perf_counter_ns()
                        for _ in range(args.replays):
                            if args.use_input_cache:
                                buffers.copy('ids', host_ids)
                                buffers.copy('positions', host_positions)
                            else:
                                torch._foreach_copy_((ids, positions), (host_ids, host_positions), non_blocking=True)
                            ht.core.mark_step()
                            if arm == 'graph':
                                graph.replay()
                            else:
                                forward(ids, positions)
                                ht.core.mark_step()
                            ht.hpu.synchronize()
                        samples.append(dict(arm=arm, iterations=args.replays,
                                            complete_us=(time.perf_counter_ns()-began)/1e3/args.replays))
                    info = graph.execution_info() if hasattr(graph, 'execution_info') else None
                    if args.require_native:
                        assert info and info['ready'] and info['replays'] == 4+2*args.replays, info
                    result['cases'].append(dict(batch=batch, length=length, exact_logits_checks=4,
                                                execution=info, samples=samples))
                    save()
                finally:
                    ht.hpu.synchronize()
                    graph.reset()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split()) >= 6 and '.so' in line.split()[-1]})
    assert not any('libe1_' in name or 'libgkg' in name for name in libraries)
    result.update(status='PASS', loaded_libraries=libraries)
    save()
    print(json.dumps(dict(status='PASS', cases=len(result['cases']), architecture=result['architecture'])), flush=True)


if __name__ == '__main__':
    main()
