#!/bin/bash
SHARED_OBJECT="bazel-bin/python/bindings/jetson_utils_python.cpython-312-x86_64-linux-gnu.so"
ldd -r "${SHARED_OBJECT}" | grep "undefined symbol:" | awk '{print$3}' | xargs demangle | grep -v ^"FAIL:"
