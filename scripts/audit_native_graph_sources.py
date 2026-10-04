#!/usr/bin/env python3
"""Read-only source/provenance audit; this is not a build or performance test.

The output deliberately excludes prompts, generated text, tensors, environments,
absolute asset paths, and model weights. Historical results remain local.
"""

import argparse
import hashlib
import json
from pathlib import Path


BRIDGE = {
    "python_packages/habana_frameworks/torch/hpu/graphs.py": [
        "def wrapped_hpugraph_forward(", "def input_hash(", "def copy_to("],
    "backend/habana_device/HPUGraph.cpp": [
        "void HPUGraph::mark_step()", "void HPUGraph::replay(",
        "void SingleHPUGraph::replayGraph("],
    "habana_lazy/hpu_lazy_tensors.cpp": ["void HbLazyTensor::ExecuteCachedGraph("],
    "habana_lazy/hlexec.cpp": ["void HlExec::Launch("],
    "backend/kernel/hpu_habana_cache.cpp": ["void RecipeLauncher::Launch("],
    "backend/kernel/hpu_habana_cache.h": ["struct RecipeLauncher {", "struct RecipeHolder {"],
    "backend/kernel/hpu_habana_launch_op_pt.cpp": ["if (!dry_run_) {"],
    "backend/synapse_helpers/graph.cpp": ["device.lock_addresses", "synLaunchWithExternalEvents("],
    "backend/helpers/collective_kernel_info.cpp": ["void CollectiveKernelInfos::Launch("],
    "habana_kernels/hccl_kernels.cpp": ["JobThreadLazyHCCL::getInstance()->addJob", "deviceCtxt->lock_address("],
    "CMakeLists.txt": ["cmake_minimum_required(VERSION 4.0"],
    "backend/CMakeLists.txt": ["habana_device/HPUGraph.cpp"],
    ".devops/build.py": ["--install-ext", "--configure", "--release"],
}
KERNELS = {
    "csrc/legacy_executor/e1_replay.cpp": ["static int enqueue(", "RECIPE_LAUNCH_SYMBOL"],
    "csrc/legacy_executor/tensor_hold.cpp": ["static void keep("],
    "csrc/legacy_executor/native_multi.inc": ["native_multi_step("],
    "csrc/native_graph/graph.cpp": ["gkg_instantiate(", "gkg_replay("],
    "csrc/native_graph/capture.cpp": ["multiple host submission threads", "host I/O must be declared outside capture"],
    "python/gaudi_kernels/graph.py": ["def capture(", "def replay("],
    "python/gaudi_kernels/serving/executor/native_backend.py": ["original_sample =", "native.single_rpc"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, definitions):
    records = []
    for relative, needles in definitions.items():
        path = root / relative
        lines = path.read_text().splitlines()
        anchors = {}
        for needle in needles:
            matches = [i for i, line in enumerate(lines, 1) if needle in line]
            if not matches:
                raise ValueError(f"Missing source anchor: {relative}: {needle}")
            anchors[needle] = matches
        records.append(dict(file=relative, sha256=digest(path), anchors=anchors))
    return records


def historical_result(path):
    data = json.loads(path.read_text())
    runs = data["runs"]
    pairs = []
    for i, control in enumerate(runs):
        if control.get("active") is not False:
            continue
        candidate = next((r for r in runs[i + 1:]
                          if r.get("active") is True and r.get("policy") == control.get("policy")), None)
        if candidate is None:
            continue
        def measurement(run):
            timing = run.get("serving_timing", run.get("diagnostic_timing", {}))
            return {key: timing[key] for key in ("tps", "p50_itl_ms", "steady_steps", "scope") if key in timing}
        pairs.append(dict(
            policy=control.get("policy"), bridge=measurement(control),
            legacy_native=measurement(candidate),
            token_sequence_equal=control["ids"] == candidate["ids"],
            scope="Historical paired arms, not a new controlled bridge-only experiment",
        ))
    frozen = []
    source = path.parent / "source"
    pins_path = source / "sha256.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    for name in ("native_backend.py", "native_service_test.py", "e1_replay.cpp",
                 "native_step.inc", "tensor_hold.cpp", "libe1_replay.so",
                 "libe1_tensor_hold.so",
                 "production-runtime/plugin/vllm_gaudi/v1/worker/hpu_model_runner.py"):
        file = source / name
        if not file.exists():
            continue
        actual = digest(file)
        expected = pins.get(name)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
        frozen.append(dict(file=name, sha256=actual,
                           matches_saved_pin=actual == expected if expected else None))
    revision_path = source / "git-revision.json"
    revision = json.loads(revision_path.read_text()).get("commit") if revision_path.exists() else None
    return dict(case=path.parent.name, result_sha256=digest(path),
                historical_status=data.get("status"), single_rpc=data.get("single_rpc"),
                reuse_pages=data.get("reuse_pages"), frozen_executor_commit=revision,
                frozen_artifacts=frozen, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--historical-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = dict(
        schema=1, bridge_baseline="5176b1b2d6865608514e83a2b1fd66459920676e",
        scope="Source audit and historical evidence extraction only; no build, device run or performance prediction",
        bridge=inventory(root, BRIDGE), kernels=inventory(args.kernels, KERNELS),
        historical=[historical_result(p) for p in args.historical_result],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), bridge_files=len(report["bridge"]),
                          kernel_files=len(report["kernels"]), historical_cases=len(report["historical"]))))


if __name__ == "__main__":
    main()
