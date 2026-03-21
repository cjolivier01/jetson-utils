#!/bin/bash

append_path_once() {
	local entry="$1"

	case ":$PATH:" in
	*":$entry:"*) ;;
	*)
		export PATH="$entry:$PATH"
		;;
	esac
}

valid_cuda_root() {
	local root="$1"
	local rel=""

	[ -n "$root" ] || return 1
	[ -x "$root/bin/nvcc" ] || return 1

	for rel in \
		targets/x86_64-linux/lib/libculibos.a \
		targets/aarch64-linux/lib/libculibos.a \
		targets/sbsa-linux/lib/libculibos.a
	do
		[ -f "$root/$rel" ] && return 0
	done

	return 1
}

valid_rocm_root() {
	local root="$1"

	[ -n "$root" ] || return 1
	[ -x "$root/bin/hipcc" ] || return 1
	[ -f "$root/include/hip/hip_runtime.h" ] || return 1
	[ -f "$root/lib/libamdhip64.so" ] || return 1

	return 0
}

detect_cuda_path() {
	local candidate=""

	for candidate in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "${CUDA_ROOT:-}"; do
		[ -n "$candidate" ] || continue
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	if command -v nvcc >/dev/null 2>&1; then
		candidate="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	fi

	for candidate in /usr/local/cuda /usr/local/cuda-*; do
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	return 1
}

detect_rocm_path() {
	local candidate=""

	for candidate in "${ROCM_PATH:-}" "${HIP_PATH:-}"; do
		[ -n "$candidate" ] || continue
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	if command -v hipcc >/dev/null 2>&1; then
		candidate="$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")"
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	fi

	for candidate in /opt/rocm /opt/rocm-*; do
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	return 1
}

ensure_cuda_env() {
	local cuda_root=""

	cuda_root="$(detect_cuda_path)" || return 1
	export CUDA_PATH="$cuda_root"
	append_path_once "$cuda_root/bin"
}

ensure_rocm_env() {
	local rocm_root=""

	rocm_root="$(detect_rocm_path)" || return 1
	export ROCM_PATH="$rocm_root"
	export HIP_PATH="$rocm_root"
	append_path_once "$rocm_root/bin"
}

detect_glibc_conflict() {
	local ver=""
	local major=""
	local minor=""

	if command -v getconf >/dev/null 2>&1; then
		ver="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
	else
		ver="$(ldd --version 2>/dev/null | head -n1 | awk '{print $NF}')"
	fi

	[ -n "$ver" ] || return 1

	major="${ver%%.*}"
	minor="${ver#*.}"
	minor="${minor%%.*}"

	if [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -ge 38 ]; }; then
		return 0
	fi

	return 1
}
