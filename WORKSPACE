_workspace_name = "jetson-utils"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "rules_cuda",
    # v0.2.3 breaks some lubcupti for our version of bazel
    commit = "3f2429254ec956220557e79ea9d5f5e8871c2907",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

load("//bazel:dependencies.bzl", "local_cuda_sdk_repository", "local_libpython_repository", "local_rocm_sdk_repository")

local_cuda_sdk_repository(
    name = "cuda_sdk",
)

local_libpython_repository(
    name = "libpython",
)

# load("//:conda.bzl", "detect_conda_python")

# detect_conda_python(name = "conda_python")

http_archive(
    name = "rules_python",
    sha256 = "9c6e26911a79fbf510a8f06d8eedb40f412023cf7fa6d1461def27116bff022c",
    strip_prefix = "rules_python-1.1.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/1.1.0/rules_python-1.1.0.tar.gz",
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-34206c29f891dbd5f6f5face7b91664c2ff7185c",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/34206c29f891dbd5f6f5face7b91664c2ff7185c.zip"],
  sha256 = "8d0b776ea5b67891f8585989d54aa34869fc12f14bf33f1dc7459458dd222e95",
)

http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-a54eab92d265337996b8e4b4149d9176c2d428a6",
  urls = ["https://github.com/pybind/pybind11/archive/a54eab92d265337996b8e4b4149d9176c2d428a6.tar.gz"],
  sha256 = "c9375b7453bef1ba0106849c83881e6b6882d892c9fae5b2572a2192100ffb8a",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

new_local_repository(
    name = "glibconfig_x86",
    build_file = "@//buildfiles:third_party/glibconfig.BUILD",
    path = "/usr/lib/x86_64-linux-gnu/glib-2.0/include",
)

new_local_repository(
    name = "glibconfig_aarch64",
    build_file = "@//buildfiles:third_party/glibconfig.BUILD",
    path = "/usr/lib/aarch64-linux-gnu/glib-2.0/include",
)

new_local_repository(
    name = "glib",
    build_file = "@//buildfiles:third_party/glib_nobuild.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "json_glib",
    build_file = "@//buildfiles:third_party/json_glib.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "gstreamer",
    build_file = "@//buildfiles:third_party/gstreamer_nobuild.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "libsoup",
    build_file = "@//buildfiles:third_party/libsoup.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "opencv_linux",
    build_file = "@//buildfiles:third_party/opencv_linux.BUILD",
    path = "/usr/include",
)

# Expose Python SOABI for naming the extension module
load("//bazel:py_soabi.bzl", "py_soabi")
py_soabi(name = "py_soabi")

local_rocm_sdk_repository(
    name = "rocm_sdk_includes",
)
