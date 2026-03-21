# jetson-utils
C++/CUDA/Python multimedia utilities for NVIDIA Jetson:

|                        |                                                 |
|------------------------|-------------------------------------------------|
| [`/`](/)               | Filesystem, CSV/JSON/XML parsing, command-line  |
| [`camera/`](camera/)   | GStreamer-based camera capture (V4L2, MIPI CSI) |
| [`codec/`](codec/)     | GStreamer-based hardware video encoder/decoder  |
| [`cuda/`](cuda/)       | CUDA image processing functions                 |
| [`display/`](display/) | OpenGL window & rendering                       |
| [`image/`](image/)     | Image loading & saving                          |
| [`input/`](input/)     | Human Interface Devices (HID) from `/dev/input` |
| [`network/`](network/) | Sockets, IPv4/IPv6, WebRTC/RTSP server          |
| [`python/`](python/)   | Python bindings and examples                    |
| [`threads/`](threads/) | Multithreading, locks, and events               |
| [`video/`](video/)     | Video streaming interfaces                      |


### Documentation

Documentation for jetson-utils can be found here:

* [API Reference](https://github.com/dusty-nv/jetson-inference#api-reference)
* [Camera Streaming and Multimedia](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md)
* [Image Manipulation with CUDA](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md)

### Building from Source

jetson-utils is typically built as a submodule of [jetson-inference](https://github.com/dusty-nv/jetson-inference), but it can also be compiled/installed standalone:

``` bash
git clone https://github.com/dusty-nv/jetson-utils
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

If you're missing dependencies, run the [`jetson-inference/CMakePreBuild.sh`](https://github.com/dusty-nv/jetson-inference/blob/master/CMakePreBuild.sh) script.

### Bazel Build (project-local)

This repository also includes Bazel build files for development, tests, and Python wheels:

```bash
# build everything (debug)
./bld

# optimized build with the glibc/CUDA workaround when needed
./perf

# or directly
bazel build //...

# run tests
bazel test //...
```

On hosts with glibc 2.38 or newer, `<math.h>` can conflict with CUDA's `rsqrt`/`rsqrtf`
declarations. `./perf` detects that case and adds `--define=glibc_math_rsqrt_conflict=1`
so the CUDA targets receive the required compiler flags automatically.

### HIP/ROCm Support (AMD GPUs)

The CUDA code now has a HIP compatibility layer so it can compile for AMD GPUs under ROCm/HIP:

- `cuda/cuda_runtime_compat.h` maps selected CUDA runtime APIs/types to HIP when building with `-DJETSON_USE_HIP`.
- `cuda/cuda_gl_interop_shim.h` maps CUDA-OpenGL interop calls to HIP-GL.

CUDA remains the default backend. To enable the HIP/ROCm backend, use:

```bash
bazel build --config=rocm //...
```

The Bazel HIP path compiles the `.cu` kernels with `hipcc` and links against `libamdhip64`.
