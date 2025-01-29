
cc_library(
    name = "pythonxdde",
    hdrs = glob(["include/**/*.h"]),
    includes = [
      "include",
      "include/python3.11"
    ],  # This makes #include <Python.h> work
    linkopts = [
      "-Llib",
      "-lpython3",
    ],
    visibility = ["//visibility:public"],
)
