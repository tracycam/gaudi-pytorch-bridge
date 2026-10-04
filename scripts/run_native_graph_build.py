#!/usr/bin/env python3
"""Run the recorded upstream build command and retain its result and logs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.case):
        raise ValueError("Invalid build case")
    root = args.config.resolve().parent
    work = root / "build-runs" / args.case
    work.mkdir(parents=True, exist_ok=False)
    result = work / "build-exit.json"
    if result.exists():
        raise ValueError("Build result already exists; use a fresh case/config")
    config = json.loads(args.config.read_text())
    env = dict(os.environ, **config["environment"])
    start = time.monotonic()
    state = dict(status="RUNNING", command=config["command"], pid=os.getpid())
    (work / "build-state.json").write_text(json.dumps(state, indent=2) + "\n")
    with (work / "build.log").open("w") as log:
        child = subprocess.Popen(config["command"], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state["child_pid"] = child.pid
        (work / "build-state.json").write_text(json.dumps(state, indent=2) + "\n")
        try:
            code = child.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            code = 124
    state.update(status="PASS" if code == 0 else "FAIL", returncode=code,
                 elapsed_seconds=time.monotonic() - start)
    result.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
