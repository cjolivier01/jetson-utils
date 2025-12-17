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

### Bazel / `./perf` build and glibc 2.38+

This repository also includes a Bazel-based build used for performance testing:

``` bash
./perf
```

On hosts with glibc 2.38 or newer, `<math.h>` now declares `rsqrt`/`rsqrtf` with `noexcept`, which conflicts with CUDA's own declarations in `crt/math_functions.h` when compiling `.cu` files with `nvcc`. To work around this, `./perf`:

- Detects the glibc version at runtime.
- For glibc >= 2.38, adds `--define=glibc_math_rsqrt_conflict=1` to the Bazel flags.
- Bazel then conditionally adds a small set of `-Xcompiler` flags only to the CUDA targets to avoid the `rsqrt`/`rsqrtf` conflict.

If you want to override this behavior:

- To force the workaround on or off when calling Bazel directly:

  ```bash
  # Force enable
  bazelisk build --config=opt --cpu=k8 --define=glibc_math_rsqrt_conflict=1 //...

  # Force disable
  bazelisk build --config=opt --cpu=k8 --define=glibc_math_rsqrt_conflict=0 //...
  ```

- To bypass `./perf` entirely, invoke Bazel directly as above with your desired `--define` setting. The `glibc_math_rsqrt_conflict` define only affects how the CUDA `.cu` files are compiled with `nvcc`; it does not change the rest of the project.
