#!/usr/bin/env python3
"""Prepare an isolated bridge build using a retained, read-only Torch install.

Uses upstream build.py afterwards. Does not install to the shared environment,
change /usr, acquire an accelerator, or override HOME.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request


def run(command, log, *, env=None):
    with log.open("a") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                       check=True, env=env)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"Existing link differs: {path}")
    elif path.exists():
        raise ValueError(f"Refusing existing nonlink: {path}")
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--torch-site", type=Path, required=True)
    parser.add_argument("--media-version", default="1.24.1.482")
    parser.add_argument("--offline-dependencies-directory", type=Path)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    work, source, torch_site = (p.resolve() for p in (args.work, args.source, args.torch_site))
    if not (torch_site / "torch").is_dir():
        raise ValueError("Retained Torch package not found")
    work.mkdir(parents=True, exist_ok=True)
    log = work / "prepare.log"
    venv = work / "build-venv"
    if not (venv / "bin/python").exists():
        run([sys.executable, "-m", "venv", str(venv)], log)
    python = venv / "bin/python"
    site = Path(subprocess.check_output([
        str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"
    ], text=True).strip())
    (site / "retained-torch.pth").write_text(str(torch_site) + "\n")
    run([str(python), "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
         "--timeout", "20", "--retries", "1", "-r", str(source / "requirements.txt")], log)

    index = "https://vault.habana.ai/artifactory/api/pypi/gaudi-python/simple/habana-media-loader/"
    with urllib.request.urlopen(index, timeout=30) as response:
        html = response.read().decode()
    (work / "media-index.html").write_text(html)
    matches = [href for href in re.findall(r'href="([^"]+)"', html)
               if f"habana_media_loader-{args.media_version}-" in href]
    if len(matches) != 1:
        raise ValueError(f"Expected one media wheel for {args.media_version}, found {len(matches)}")
    url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(index, matches[0]))
    expected = urllib.parse.parse_qs(fragment)["sha256"][0]
    wheel = work / "downloads" / Path(urllib.parse.urlsplit(url).path).name
    wheel.parent.mkdir(exist_ok=True)
    if not wheel.exists():
        with urllib.request.urlopen(url, timeout=60) as response, wheel.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    if sha(wheel) != expected:
        raise ValueError("Media wheel checksum mismatch")
    run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], log)
    media = site / "habana_frameworks/mediapipe"
    if not (media / "include/media_pytorch_proxy.h").exists():
        raise ValueError("Official media wheel lacks required build header")

    latest = work / "runtime-libs"
    latest.mkdir(exist_ok=True)
    for file in Path("/usr/lib/habanalabs").glob("*.so*"):
        link(file, latest / file.name)
    link(Path("/usr/include/habanalabs/hl_logger"), work / "sdk/hl_logger/include")
    env = dict(os.environ)
    env.update({
        "VIRTUAL_ENV": str(venv),
        "PATH": str(venv / "bin") + os.pathsep + env["PATH"],
        "HABANA_SOFTWARE_STACK": str(work),
        "BUILD_ROOT": str(work / "build"),
        "BUILD_ROOT_LATEST": str(latest),
        "PYTORCH_MODULES_ROOT_PATH": str(source),
        "PYTORCH_MODULES_RELEASE_BUILD": str(work / "build-release"),
        "PYTORCH_MODULES_DEBUG_BUILD": str(work / "build-debug"),
        "SWTOOLS_SDK_ROOT": str(work / "sdk"),
        "MEDIA_ROOT": str(media),
        **{key: "/usr/include/habanalabs" for key in [
            "HCL_INCLUDE_DIR", "SPECS_EXT_ROOT", "SPECS_EMBEDDED_ROOT",
            "SYNAPSE_INCLUDE_DIR", "SYNAPSE_UTILS_INCLUDE_DIR"]},
    })
    tools = {}
    for name in ("git", "cmake", "ninja", "c++", "protoc"):
        executable = shutil.which(name, path=env["PATH"])
        if executable is None:
            raise RuntimeError(f"Required build tool missing: {name}; prepare it in the isolated toolchain")
        version = subprocess.check_output([executable, "--version"], env=env,
                                          text=True, stderr=subprocess.STDOUT)
        tools[name] = dict(path=executable, version=version.splitlines()[0])
    command = [str(python), str(source / ".devops/build.py"), "-cr", "--no-iwyu",
               "--pt-versions", "preinstalled", "--recreate-venv", "never",
               "-j", str(args.jobs)]
    if args.offline_dependencies_directory is not None:
        dependencies = args.offline_dependencies_directory.resolve(strict=True)
        command.extend(["--offline-dependencies-directory", str(dependencies)])
    explicit = {key: env[key] for key in [
        "VIRTUAL_ENV", "PATH", "HABANA_SOFTWARE_STACK", "BUILD_ROOT", "BUILD_ROOT_LATEST", "PYTORCH_MODULES_ROOT_PATH",
        "PYTORCH_MODULES_RELEASE_BUILD", "PYTORCH_MODULES_DEBUG_BUILD", "SWTOOLS_SDK_ROOT",
        "MEDIA_ROOT", "HCL_INCLUDE_DIR", "SPECS_EXT_ROOT", "SPECS_EMBEDDED_ROOT",
        "SYNAPSE_INCLUDE_DIR", "SYNAPSE_UTILS_INCLUDE_DIR"]}
    for key in ("GIT_EXEC_PATH", "GIT_TEMPLATE_DIR"):
        if key in env:
            explicit[key] = env[key]
    metadata = dict(command=command, environment=explicit, tools=tools,
                    media=dict(url=url, sha256=expected), requirements_sha256=sha(source / "requirements.txt"),
                    retained_torch_site=str(torch_site))
    (work / "build-config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run([str(python), "-m", "pip", "freeze"], work / "packages.txt", env=env)
    print(json.dumps(dict(status="PREPARED", config=str(work / "build-config.json"))))


if __name__ == "__main__":
    main()
