#!/usr/bin/env python3
import os
import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        print("usage: setup.py [develop|bdist_wheel]", file=sys.stderr)
        return 1

    cmd = sys.argv[1]
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if cmd == "bdist_wheel":
        # Delegate to Bazel wheel builder
        return subprocess.call(["bazel", "run", "//python:bdist_wheel"])  # bazelisk-compatible

    if cmd == "develop":
        # Run Bazel target that sets up a develop-style .pth install
        env = dict(os.environ)
        env["JETSON_UTILS_WORKSPACE"] = repo_root
        return subprocess.call(["bazel", "test", "//python:develop", "-t-"], env=env)

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

