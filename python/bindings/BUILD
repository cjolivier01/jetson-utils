#load("@rules_python//python:defs.bzl", "py_extension")

cc_library(
    name = "jetson_utils_lib",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-std=c++17",  # Use C++17 (or higher if needed)
    ],
    deps = [
        "//base:base",
        "//image:image",
        "//camera:camera",
        "@conda_python//:python",
    ],
)

# # py_extension(
#     name = "jetson_utils",  # The Python extension's name
#     srcs = [],  # No additional source files needed
#     extension = "jetson_utils_lib",  # Link against the C++ library above
# )
