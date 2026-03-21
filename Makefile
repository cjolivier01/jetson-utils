TOPDIR := $(shell pwd)
GLIBC_RSQRT_DEFINE := $(shell \
	ver=$$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $$2}'); \
	if [ -z "$$ver" ]; then ver=$$(ldd --version 2>/dev/null | head -n1 | awk '{print $$NF}'); fi; \
	major=$${ver%%.*}; \
	minor=$${ver#*.}; minor=$${minor%%.*}; \
	if [ -n "$$major" ] && [ -n "$$minor" ] && { [ "$$major" -gt 2 ] || { [ "$$major" -eq 2 ] && [ "$$minor" -ge 38 ]; }; }; then \
		echo "--define=glibc_math_rsqrt_conflict=1"; \
	fi \
)

all: print_targets

.PHONY: all print_targets perf debug test wheel develop clean distclean expunge

perf:
	./perf

debug:
	./bld

test:
	bazelisk test --config=opt $(GLIBC_RSQRT_DEFINE) //...

wheel:
	bazelisk run --config=opt $(GLIBC_RSQRT_DEFINE) //python:bdist_wheel

develop:
	bazelisk run --config=opt $(GLIBC_RSQRT_DEFINE) //python:develop

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
