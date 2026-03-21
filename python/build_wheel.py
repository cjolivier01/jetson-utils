#!/usr/bin/env python3
import base64
import csv
import hashlib
import os
import re
import shutil
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path


def norm_project(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name)


def wheel_tags_from_ext(ext_path: Path):
    # Derive tags from extension SOABI to match the built binary
    # Expect filename like jetson_utils_python.cpython-312-<arch>.so
    m = re.search(r"jetson_utils_python\.(?P<soabi>[^.]+)\.so$", ext_path.name)
    if not m:
        raise RuntimeError(f"Unable to parse SOABI from {ext_path.name}")
    soabi = m.group("soabi")
    m2 = re.match(r"cpython-(\d{2,3})", soabi)
    if not m2:
        raise RuntimeError(f"Unexpected SOABI format: {soabi}")
    pyver = m2.group(1)
    py_tag = f"cp{pyver}"
    abi_tag = py_tag
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return py_tag, abi_tag, plat


def find_version(pkg_root: Path) -> str:
    init_py = pkg_root / "jetson_utils" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    m = re.search(r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    return m.group(1) if m else "0.0.0"


def hash_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", path.stat().st_size


def main():
    # Bazel sets BUILD_WORKSPACE_DIRECTORY when running via `bazel run`
    workspace = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd()))
    out_dir = workspace / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)

    runfiles = Path(os.environ.get("RUNFILES_DIR", "")) or Path(sys.argv[0]).parent

    # Locate the extension .so in runfiles (provided as data dep)
    ext_candidates = list(runfiles.rglob("jetson_utils_python.*.so"))
    if not ext_candidates:
        print("error: extension module not found in runfiles", file=sys.stderr)
        return 1
    ext_path = ext_candidates[0]

    # Locate the shared core lib to bundle alongside the extension
    core_candidates = [p for p in runfiles.rglob("libjetson_cuda.so")]
    core_path = core_candidates[0] if core_candidates else None

    # Locate the pure-Python package root under runfiles
    pkg_root_candidates = [p for p in runfiles.rglob("python/python") if p.is_dir()]
    if not pkg_root_candidates:
        print("error: python package files not found in runfiles", file=sys.stderr)
        return 1
    pkg_root = pkg_root_candidates[0]

    version = find_version(pkg_root)
    py_tag, abi_tag, plat_tag = wheel_tags_from_ext(ext_path)

    dist_name = "jetson_utils"
    project_name = "jetson-utils"  # METADATA Name
    wheel_basename = f"{dist_name}-{version}-{py_tag}-{abi_tag}-{plat_tag}"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "wheel"
        root.mkdir()

        # Copy python package files
        for src in pkg_root.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(pkg_root)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Place compiled extension inside the package directory so it installs under
        # site-packages/jetson_utils/ per standard packaging expectations.
        # The package code will import it relatively (with fallback to top-level for Bazel dev).
        pkg_ext_dir = root / "jetson_utils"
        pkg_ext_dir.mkdir(parents=True, exist_ok=True)
        ext_dst = pkg_ext_dir / ext_path.name
        shutil.copy2(ext_path, ext_dst)

        # Include shared core library (for DT_NEEDED resolution at runtime)
        if core_path and core_path.is_file():
            shutil.copy2(core_path, root / core_path.name)

        # Create .dist-info
        dist_info = root / f"{dist_name}-{version}.dist-info"
        dist_info.mkdir()

        (dist_info / "WHEEL").write_text(
            """Wheel-Version: 1.0
Generator: bazel bdist_wheel
Root-Is-Purelib: false
Tag: {py}-{abi}-{plat}
""".format(py=py_tag, abi=abi_tag, plat=plat_tag),
            encoding="utf-8",
        )

        (dist_info / "METADATA").write_text(
            """Metadata-Version: 2.1
Name: {name}
Version: {version}
Summary: C++/CUDA/Python multimedia utilities for NVIDIA Jetson
Home-page: https://github.com/dusty-nv/jetson-utils
""".format(name=project_name, version=version),
            encoding="utf-8",
        )

        # Build RECORD with hashes
        record_rows = []
        for file in root.rglob("*"):
            if file.is_dir():
                continue
            rel = file.relative_to(root).as_posix()
            if rel.endswith(".dist-info/RECORD"):
                # RECORD entry is allowed to have empty hash/size; write later
                continue
            digest, size = hash_file(file)
            record_rows.append((rel, digest, str(size)))

        record_path = dist_info / "RECORD"
        with record_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in record_rows:
                w.writerow(row)
            # RECORD entry for itself (empty hash/size)
            w.writerow((f"{dist_name}-{version}.dist-info/RECORD", "", ""))

        # Zip to wheel
        wheel_path = out_dir / f"{wheel_basename}.whl"
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in root.rglob("*"):
                if file.is_dir():
                    continue
                z.write(file, file.relative_to(root).as_posix())

        print(f"Wrote {wheel_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
