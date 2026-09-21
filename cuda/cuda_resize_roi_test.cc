#include "cudaResizeRoi.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int kInputWidth = 4;
constexpr int kInputHeight = 4;
constexpr int kRoiWidth = 2;
constexpr int kRoiHeight = 2;
constexpr int kOutputWidth = 4;
constexpr int kOutputHeight = 4;

bool cuda_ok(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return true;
  std::cerr << operation << " failed: " << cudaGetErrorString(error) << '\n';
  return false;
}

uchar3 gray(uint8_t value) { return make_uchar3(value, value, value); }

bool run_linear_case(int src_x, int src_y) {
  std::vector<uchar3> input(kInputWidth * kInputHeight, gray(240));
  input[src_y * kInputWidth + src_x] = gray(0);
  input[src_y * kInputWidth + src_x + 1] = gray(100);
  input[(src_y + 1) * kInputWidth + src_x] = gray(100);
  input[(src_y + 1) * kInputWidth + src_x + 1] = gray(200);

  uchar3 *device_input = nullptr;
  uchar3 *device_output = nullptr;
  const size_t input_bytes = input.size() * sizeof(uchar3);
  const size_t output_bytes = kOutputWidth * kOutputHeight * sizeof(uchar3);
  if (!cuda_ok(cudaMalloc(&device_input, input_bytes), "cudaMalloc(input)") ||
      !cuda_ok(cudaMalloc(&device_output, output_bytes),
               "cudaMalloc(output)")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return false;
  }

  bool ok =
      cuda_ok(cudaMemcpy(device_input, input.data(), input_bytes,
                         cudaMemcpyHostToDevice),
              "input copy") &&
      cuda_ok(cudaResizeROI(device_input, kInputWidth, kInputHeight, src_x,
                            src_y, kRoiWidth, kRoiHeight, device_output,
                            kOutputWidth, kOutputHeight, 0, 0, kOutputWidth,
                            kOutputHeight, FILTER_LINEAR, 0),
              "cudaResizeROI") &&
      cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<uchar3> output(kOutputWidth * kOutputHeight);
  if (ok)
    ok = cuda_ok(cudaMemcpy(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost),
                 "output copy");
  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok)
    return false;

  constexpr std::array<uint8_t, kOutputWidth * kOutputHeight> expected = {
      0, 25, 75, 100, 25, 50, 100, 125, 75, 100, 150, 175, 100, 125, 175, 200,
  };
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i].x != expected[i] || output[i].y != expected[i] ||
        output[i].z != expected[i]) {
      std::cerr << "unexpected output at index " << i << " for ROI origin ("
                << src_x << ", " << src_y << "): got "
                << static_cast<int>(output[i].x) << ", expected "
                << static_cast<int>(expected[i]) << '\n';
      return false;
    }
  }
  return true;
}

bool run_linear_rgba_case() {
  const std::array<uchar4, kInputWidth * kInputHeight> input = {
      make_uchar4(0, 10, 20, 30),  make_uchar4(100, 110, 120, 130),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
      make_uchar4(40, 50, 60, 70), make_uchar4(140, 150, 160, 170),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
      make_uchar4(0, 0, 0, 0),     make_uchar4(0, 0, 0, 0),
  };
  uchar4 *device_input = nullptr;
  uchar4 *device_output = nullptr;
  const size_t input_bytes = input.size() * sizeof(uchar4);
  const size_t output_bytes = kOutputWidth * kOutputHeight * sizeof(uchar4);
  if (!cuda_ok(cudaMalloc(&device_input, input_bytes),
               "cudaMalloc(RGBA input)") ||
      !cuda_ok(cudaMalloc(&device_output, output_bytes),
               "cudaMalloc(RGBA output)")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return false;
  }

  bool ok =
      cuda_ok(cudaMemcpy(device_input, input.data(), input_bytes,
                         cudaMemcpyHostToDevice),
              "RGBA input copy") &&
      cuda_ok(cudaResizeROI(device_input, kInputWidth, kInputHeight, 0, 0,
                            kRoiWidth, kRoiHeight, device_output, kOutputWidth,
                            kOutputHeight, 0, 0, kOutputWidth, kOutputHeight,
                            FILTER_LINEAR, 0),
              "cudaResizeROI(RGBA)") &&
      cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(RGBA)");

  std::vector<uchar4> output(kOutputWidth * kOutputHeight);
  if (ok)
    ok = cuda_ok(cudaMemcpy(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost),
                 "RGBA output copy");
  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok)
    return false;

  constexpr std::array<uint8_t, kOutputWidth * kOutputHeight> expected_red = {
      0, 25, 75, 100, 10, 35, 85, 110, 30, 55, 105, 130, 40, 65, 115, 140,
  };
  for (size_t i = 0; i < output.size(); ++i) {
    const uchar4 expected =
        make_uchar4(expected_red[i], expected_red[i] + 10, expected_red[i] + 20,
                    expected_red[i] + 30);
    if (output[i].x != expected.x || output[i].y != expected.y ||
        output[i].z != expected.z || output[i].w != expected.w) {
      std::cerr << "unexpected RGBA output at index " << i << '\n';
      return false;
    }
  }
  return true;
}

bool rejects_negative_extents() {
  uchar3 *device_input = nullptr;
  uchar3 *device_output = nullptr;
  const size_t input_bytes = kInputWidth * kInputHeight * sizeof(uchar3);
  const size_t output_bytes = kOutputWidth * kOutputHeight * sizeof(uchar3);
  if (!cuda_ok(cudaMalloc(&device_input, input_bytes),
               "cudaMalloc(invalid input)") ||
      !cuda_ok(cudaMalloc(&device_output, output_bytes),
               "cudaMalloc(invalid output)")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return false;
  }

  const auto resize = [&](int src_width, int src_height, int dst_width,
                          int dst_height) {
    return cudaResizeROI(device_input, kInputWidth, kInputHeight, 0, 0,
                         src_width, src_height, device_output, kOutputWidth,
                         kOutputHeight, 0, 0, dst_width, dst_height,
                         FILTER_LINEAR, 0);
  };
  const bool ok = resize(-1, 1, 1, 1) == cudaErrorInvalidValue &&
                  resize(1, -1, 1, 1) == cudaErrorInvalidValue &&
                  resize(1, 1, -1, 1) == cudaErrorInvalidValue &&
                  resize(1, 1, 1, -1) == cudaErrorInvalidValue;
  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok)
    std::cerr << "a negative ROI extent was accepted\n";
  return ok;
}

} // namespace

int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
    return 1;
  if (device_count == 0) {
    std::cout << "SKIP: no CUDA device available\n";
    return 0;
  }
  return run_linear_case(1, 1) && run_linear_case(0, 0) &&
                 run_linear_rgba_case() && rejects_negative_extents()
             ? 0
             : 1;
}
