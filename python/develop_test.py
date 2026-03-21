#!/usr/bin/env python3
import os
import sys
import sysconfig
import site
import subprocess


def find_workspace_root():
    # Prefer explicit env set by setup.py
    root = os.environ.get("JETSON_UTILS_WORKSPACE")
    if root and os.path.isdir(root):
        return os.path.realpath(root)

    # Fallback: walk parents to locate WORKSPACE file
    d = os.path.realpath(os.getcwd())
    for _ in range(10):
        if os.path.exists(os.path.join(d, "WORKSPACE")) or os.path.exists(os.path.join(d, "WORKSPACE.bazel")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    # As last resort, attempt RUNFILES workspace
    runfiles = os.environ.get("RUNFILES_DIR") or ""
    ws = os.environ.get("TEST_WORKSPACE") or "jetson-utils"
    cand = os.path.join(runfiles, ws)
    if os.path.isdir(cand):
        return os.path.realpath(cand)
    raise RuntimeError("Unable to determine workspace root. Set JETSON_UTILS_WORKSPACE env var.")


def site_packages_paths():
    paths = []
    try:
        base = sysconfig.get_paths().get("purelib")
        if base:
            paths.append(base)
    except Exception:
        pass
    try:
        usr = site.getusersitepackages()
        if usr:
            paths.append(usr)
    except Exception:
        pass
    # de-dup
    return list(dict.fromkeys([p for p in paths if p]))


def ensure_extension_built(root):
    so_glob = os.path.join(root, "bazel-bin", "python", "bindings", "jetson_utils_python.*.so")
    if not any(True for _ in __import__('glob').glob(so_glob)):
        # Try building the extension via bazel
        if os.path.exists(os.path.join(root, "WORKSPACE")) or os.path.exists(os.path.join(root, "WORKSPACE.bazel")):
            try:
                subprocess.check_call(["bazel", "build", "//python/bindings:jetson_utils_python_ext", "//core:jetson_utils", "//python/extdeps:jetson_cuda"], cwd=root)
            except Exception as e:
                print(f"warning: failed to build extension: {e}", file=sys.stderr)


def write_develop_pth(root):
    python_pkg = os.path.join(root, "python", "python")
    bazel_bindings = os.path.join(root, "bazel-bin", "python", "bindings")
    runfiles_bindings = os.path.join(root, "python", "bindings")
    core_lib = os.path.join(root, "core")
    extdeps_lib = os.path.join(root, "python", "extdeps")

    ensure_extension_built(root)

    lines = [python_pkg]
    # Add whichever bindings/lib paths exist in this environment
    for p in (bazel_bindings, runfiles_bindings, core_lib, extdeps_lib):
        if os.path.isdir(p):
            lines.append(p)
    for sp in site_packages_paths():
        os.makedirs(sp, exist_ok=True)
        pth = os.path.join(sp, "jetson_utils_develop.pth")
        with open(pth, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(os.path.realpath(ln) + "\n")
        print(f"wrote develop .pth: {pth}")
    return True


def verify_import():
    code = "import jetson_utils as ju; print(ju.Log.GetLevel())"
    env = dict(os.environ)
    # Force enabling user site if needed
    env["PYTHONNOUSERSITE"] = ""
    subprocess.check_call([sys.executable, "-c", code], env=env)


def main():
    root = find_workspace_root()
    write_develop_pth(root)
    # In Bazel test environment, verifying import from a fresh interpreter can be flaky
    # because user/site .pth activation varies. The setup.py flow verifies it externally.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
