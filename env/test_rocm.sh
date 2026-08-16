#!/usr/bin/env bash
set -euo pipefail

TOPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_BUILD=0

for arg in "$@"; do
	case "$arg" in
		--full)
			FULL_BUILD=1
			;;
		-h|--help)
			cat <<'USAGE'
Usage: env/test_rocm.sh [--full]

Validates the HIP/ROCm Bazel path.

Options:
  --full   Build //... after the core ROCm validation targets pass.
USAGE
			exit 0
			;;
		*)
			printf 'Unknown option: %s\n' "$arg" >&2
			exit 1
			;;
	esac
done

source "$TOPDIR/toolchain_env.sh"
ensure_python_env
ensure_rocm_env

cd "$TOPDIR"

bazelisk build --repo_env=PYTHON_BIN_PATH="$PYTHON_BIN_PATH" --config=rocm \
	//cuda:cuda_lib \
	//display:display \
	//display/gl-display-test:gl_display_test \
	//image:image \
	//video:video_output \
	//camera:camera

bazelisk test --repo_env=PYTHON_BIN_PATH="$PYTHON_BIN_PATH" --config=rocm \
	//base:logging_test \
	//cuda:hip_build_all_test \
	//cuda:hip_runtime_smoke_test

if [ "$FULL_BUILD" -eq 1 ]; then
	bazelisk build --repo_env=PYTHON_BIN_PATH="$PYTHON_BIN_PATH" --config=rocm //...
fi
