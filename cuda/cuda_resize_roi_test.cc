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

} // namespace

int main() {
  int device_count = 0;
  if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
    return 1;
  if (device_count == 0) {
    std::cout << "SKIP: no CUDA device available\n";
    return 0;
  }
  return run_linear_case(1, 1) && run_linear_case(0, 0) ? 0 : 1;
}
