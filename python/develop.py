#!/usr/bin/env python3
import argparse
import fnmatch
import os
import shutil
import site
import sys
import sysconfig
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Install jetson-utils in editable/develop mode")
    parser.add_argument("--user", action="store_true", help="install into the user site-packages")
    parser.add_argument(
        "--prefix",
        type=str,
        help="custom prefix to resolve site-packages (similar to --prefix for setup.py develop)",
    )
    parser.add_argument(
        "--pth-name",
        default="jetson_utils_develop.pth",
        help="name for the .pth file that points at the source tree",
    )
    return parser.parse_args()


def find_runfiles_root() -> Optional[Path]:
    runfiles_env = os.environ.get("RUNFILES_DIR")
    if runfiles_env:
        path = Path(runfiles_env)
        if path.exists():
            return path

    fallback = Path(__file__).resolve().parent
    return fallback if fallback.exists() else None


def resolve_workspace(runfiles: Optional[Path]) -> Path:
    workspace_env = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_env:
        return Path(workspace_env)

    workspace_name = os.environ.get("BAZEL_WORKSPACE", "jetson-utils")
    if runfiles:
        candidate = runfiles / workspace_name
        if candidate.exists():
            return candidate

    return Path.cwd()


def site_packages_dir(prefix: Optional[str], user: bool) -> Path:
    if prefix and user:
        raise ValueError("only one of --user or --prefix may be provided")

    if prefix:
        paths = sysconfig.get_paths(vars={"base": prefix, "platbase": prefix})
        return Path(paths["platlib"])

    if user:
        return Path(site.getusersitepackages())

    paths = sysconfig.get_paths()
    return Path(paths.get("platlib") or paths["purelib"])


def search_tree(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None

    for current_root, _, files in os.walk(root, followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return Path(current_root) / name
    return None


def find_in_manifest(pattern: str) -> Optional[Path]:
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if not manifest:
        return None

    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ", 1)
            logical = real = ""
            if len(parts) == 2:
                logical, real = parts
            elif parts:
                logical = real = parts[0]
            if not real:
                continue
            if fnmatch.fnmatch(Path(logical).name, pattern) or fnmatch.fnmatch(Path(real).name, pattern):
                candidate = Path(real)
                if candidate.is_file():
                    return candidate
    return None


def find_first(pattern: str, runfiles: Optional[Path], workspace: Path) -> Optional[Path]:
    roots = []
    if runfiles:
        roots.append(runfiles)

    script_dir = Path(__file__).resolve().parent
    if script_dir not in roots:
        roots.append(script_dir)

    bazel_bin = workspace / "bazel-bin"
    if bazel_bin.exists():
        roots.append(bazel_bin)

    for root in roots:
        found = search_tree(root, pattern)
        if found:
            return found

    return find_in_manifest(pattern)


def main() -> int:
    args = parse_args()

    try:
        target_site = site_packages_dir(args.prefix, args.user)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target_site.mkdir(parents=True, exist_ok=True)

    runfiles = find_runfiles_root()
    workspace = resolve_workspace(runfiles)
    pkg_root = workspace / "python" / "python"

    if not pkg_root.exists():
        print(f"error: python sources not found at {pkg_root}", file=sys.stderr)
        return 1

    ext_path = find_first("jetson_utils_python.*.so", runfiles, workspace)
    if not ext_path:
        print("error: extension module not found in runfiles", file=sys.stderr)
        return 1

    ext_dst = target_site / ext_path.name
    ext_dst.parent.mkdir(parents=True, exist_ok=True)
    if ext_dst.exists():
        ext_dst.unlink()
    shutil.copy2(ext_path, ext_dst)

    bundled_libs = []
    for lib_name in ("libjetson_utils.so", "libjetson_cuda.so"):
        lib_path = find_first(lib_name, runfiles, workspace)
        if not lib_path:
            continue
        dst = target_site / lib_path.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(lib_path, dst)
        bundled_libs.append(dst.name)

    pth_path = target_site / args.pth_name
    pth_path.write_text(str(pkg_root.resolve()) + "\n", encoding="utf-8")

    print(f"Installed editable jetson-utils to {target_site}")
    print(f"  - sources: {pkg_root}")
    print(f"  - extension: {ext_dst}")
    if bundled_libs:
        print(f"  - bundled shared libs: {', '.join(bundled_libs)}")
    print(f"  - .pth: {pth_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
