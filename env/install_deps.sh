#!/usr/bin/env bash
set -euo pipefail

BASE_PACKAGES=(
	curl
	libglew-dev
	libgstreamer-plugins-bad1.0-dev
	libgstrtspserver-1.0-dev
	libjson-glib-dev
	libsoup2.4-dev
)

VULKAN_PACKAGES=(
	libvulkan-dev
	vulkan-tools
	glslang-tools
	spirv-tools
)

ROCM_PACKAGES=(
	hipcc
	rocm-hip-runtime-dev
	rocm-device-libs
	rocminfo
)

usage() {
	cat <<'USAGE'
Usage: env/install_deps.sh [--base] [--vulkan] [--rocm] [--all]

Installs Ubuntu/Debian packages used by the Bazel/CMake builds.

Options:
  --base     Install the multimedia/OpenGL dependencies used by the core build.
  --vulkan   Install Vulkan headers and shader/compiler tools for future backend work.
  --rocm     Install ROCm HIP userspace packages if the AMD ROCm apt repo is already configured.
  --all      Install base + Vulkan + ROCm package sets.

If no option is provided, --base is used.
USAGE
}

have_pkg() {
	apt-cache show "$1" >/dev/null 2>&1
}

install_group() {
	local -a packages=("$@")
	local -a available=()
	local pkg=""

	for pkg in "${packages[@]}"; do
		if have_pkg "$pkg"; then
			available+=("$pkg")
		else
			printf 'Skipping unavailable package: %s\n' "$pkg" >&2
		fi
	done

	if [ "${#available[@]}" -eq 0 ]; then
		return 0
	fi

	sudo apt-get update
	sudo apt-get install -y "${available[@]}"
}

want_base=0
want_vulkan=0
want_rocm=0

if [ "$#" -eq 0 ]; then
	want_base=1
fi

for arg in "$@"; do
	case "$arg" in
		--base)
			want_base=1
			;;
		--vulkan)
			want_vulkan=1
			;;
		--rocm)
			want_rocm=1
			;;
		--all)
			want_base=1
			want_vulkan=1
			want_rocm=1
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			printf 'Unknown option: %s\n\n' "$arg" >&2
			usage >&2
			exit 1
			;;
	esac
done

if [ "$want_base" -eq 1 ]; then
	install_group "${BASE_PACKAGES[@]}"
fi

if [ "$want_vulkan" -eq 1 ]; then
	install_group "${VULKAN_PACKAGES[@]}"
fi

if [ "$want_rocm" -eq 1 ]; then
	install_group "${ROCM_PACKAGES[@]}"
fi
