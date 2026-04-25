TOPDIR := $(shell pwd)
CPU := $(shell uname -m)
ifeq ($(CPU),x86_64)
CPU := k8
endif
GLIBC_RSQRT_DEFINE := $(shell \
	ver=$$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $$2}'); \
	if [ -z "$$ver" ]; then ver=$$(ldd --version 2>/dev/null | head -n1 | awk '{print $$NF}'); fi; \
	major=$${ver%%.*}; \
	minor=$${ver#*.}; minor=$${minor%%.*}; \
	if [ -n "$$major" ] && [ -n "$$minor" ] && { [ "$$major" -gt 2 ] || { [ "$$major" -eq 2 ] && [ "$$minor" -ge 38 ]; }; }; then \
		echo "--define=glibc_math_rsqrt_conflict=1"; \
	fi \
)
OPT_BAZEL_FLAGS := --config=opt --cpu=$(CPU) $(GLIBC_RSQRT_DEFINE)
CUDA_TOOLCHAIN_ENV := source "$(TOPDIR)/toolchain_env.sh"; ensure_python_env; ensure_cuda_env; ensure_rocm_env || true;
ROCM_TOOLCHAIN_ENV := source "$(TOPDIR)/toolchain_env.sh"; ensure_python_env; ensure_cuda_env; ensure_rocm_env;

all: print_targets

.PHONY: all cuda rocm rocm-test missing_toolkit print_targets perf debug test wheel develop clean distclean expunge

cuda:
	bash -lc '$(CUDA_TOOLCHAIN_ENV) bazelisk build $(OPT_BAZEL_FLAGS) --repo_env=PYTHON_BIN_PATH=$$PYTHON_BIN_PATH //...'

rocm:
	bash -lc '$(ROCM_TOOLCHAIN_ENV) bazelisk build $(OPT_BAZEL_FLAGS) --repo_env=PYTHON_BIN_PATH=$$PYTHON_BIN_PATH --config=rocm //...'

rocm-test:
	bash -lc '$(ROCM_TOOLCHAIN_ENV) ./env/test_rocm.sh'

missing_toolkit:
	@printf '%s\n' 'No supported GPU toolkit detected. Install CUDA or ROCm, or export CUDA_PATH/ROCM_PATH.' >&2
	@exit 1

perf:
	./perf

debug:
	./bld

test:
	bash -lc '$(CUDA_TOOLCHAIN_ENV) bazelisk test $(OPT_BAZEL_FLAGS) --repo_env=PYTHON_BIN_PATH=$$PYTHON_BIN_PATH //...'

wheel:
	bash -lc '$(CUDA_TOOLCHAIN_ENV) bazelisk run $(OPT_BAZEL_FLAGS) --repo_env=PYTHON_BIN_PATH=$$PYTHON_BIN_PATH //python:bdist_wheel'

develop:
	bash -lc '$(CUDA_TOOLCHAIN_ENV) bazelisk run $(OPT_BAZEL_FLAGS) --repo_env=PYTHON_BIN_PATH=$$PYTHON_BIN_PATH //python:develop'

clean:
	bazelisk clean

distclean expunge:
	bazelisk clean --expunge

print_targets:
	@printf '%s\n' \
		"Available make targets (run 'make <target>'):" \
		'' \
		'Build Outputs' \
		'-------------' \
		'all          Show this help text (same as print_targets).' \
		'cuda         Build every Bazel target with the CUDA backend.' \
		'rocm         Build every Bazel target with the HIP/ROCm backend.' \
		'rocm-test    Build and test the committed HIP/ROCm validation target set.' \
		'perf         Build every Bazel target with optimized flags via ./perf (includes glibc conflict workaround detection).' \
		'debug        Build every Bazel target with debug flags via ./bld; use for local iteration with symbols.' \
		'' \
		'Developer Workflow' \
		'------------------' \
		'test         Runs the optimized Bazel test suite (bazelisk test --config=opt //...).' \
		'wheel        Builds the Python wheel with //python:bdist_wheel.' \
		'develop      Runs //python:develop to stage a local editable-style Python setup.' \
		'' \
		'Maintenance & Cleanup' \
		'---------------------' \
		'clean        bazel clean to drop cached outputs when builds behave strangely or after branch switches.' \
		'distclean    bazel clean --expunge (also aliased as expunge) for a fully fresh Bazel state.' \
		'expunge      Same as distclean; provided for convenience.' \
		'' \
		'Meta' \
		'----' \
		'print_targets  Shows this help text.'
