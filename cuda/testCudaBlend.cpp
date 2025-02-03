#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "cudaBlend.h"

#include <cassert>
#include <iostream>

class CudaMat {
 private:
  void* d_data = nullptr;
  size_t size;
  int rows, cols, type;

 public:
  CudaMat(const cv::Mat& mat) : rows(mat.rows), cols(mat.cols), type(mat.type()) {
    size = mat.total() * mat.elemSize();
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, mat.data, size, cudaMemcpyHostToDevice);
  }

  ~CudaMat() {
    if (d_data) {
      cudaFree(d_data);
    }
  }

  cv::Mat download() const {
    cv::Mat mat(rows, cols, type);
    cudaMemcpy(mat.data, d_data, size, cudaMemcpyDeviceToHost);
    return mat;
  }

  void* data() {
    return d_data;
  }
  const void* data() const {
    return d_data;
  }
  size_t bytes() const {
    return size;
  }
};

cudaError_t cudaLaplacianBlend(
    const float* image1,
    const float* image2,
    const float* mask,
    float* output,
    int imageWidth,
    int imageHeight,
    int numLevels);

int main(int argc, char** argv) {
  // Usage check.
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2> <mask> <output>" << std::endl;
    return -1;
  }

  // Load the two images (in color).
  cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
  cv::Mat seam_mask = cv::imread(argv[3], cv::IMREAD_ANYDEPTH);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Error loading images!" << std::endl;
    return -1;
  }

  // For this simple test, require both images to have the same dimensions.
  if (img1.size() != img2.size()) {
    std::cerr << "Images must have the same dimensions!" << std::endl;
    return -1;
  }

  // Convert images to float (CV_32FC3) and scale pixel values to [0,1].
  cv::Mat img1_float, img2_float, mask;
  img1.convertTo(img1_float, CV_32FC3, 1.0 / 255.0);
  img2.convertTo(img2_float, CV_32FC3, 1.0 / 255.0);

  seam_mask.convertTo(mask, CV_32FC1, 1.0 / 255.0);

  // Create a simple seam mask (single–channel, CV_32FC1):
  // Here we use a hard–coded seam: the left half of the image is taken entirely from image1
  // (mask value 1.0) and the right half from image2 (mask value 0.0). In a more complex case,
  // the mask can be generated based on feature detection or user input.
  // cv::Mat mask(img1.size(), CV_32FC1);
  // for (int y = 0; y < mask.rows; y++) {
  //   for (int x = 0; x < mask.cols; x++) {
  //     mask.at<float>(y, x) = (x < mask.cols / 2) ? 1.0f : 0.0f;
  //   }
  // }

  // Prepare the output image (as float).
  cv::Mat blended_float(img1.size(), CV_32FC3);

  // Configurable parameter: number of pyramid levels.
  int numLevels = 12;
  int width = img1.cols;
  int height = img1.rows;

  CudaLaplacianBlendContext context(width, height, numLevels);

  CudaMat cudaImage1Float(img1_float);
  CudaMat cudaImage2Float(img2_float);
  CudaMat cudaMask(mask);
  CudaMat cudaBlendedFloat(blended_float);

  // Call the CUDA–based blending function.
  // It is assumed that blendImages copies data to/from device memory,
  // launches the appropriate kernels, and returns the blended image.

  cudaLaplacianBlendWithContext(
      (const float*)cudaImage1Float.data(),
      (const float*)cudaImage2Float.data(),
      (const float*)cudaMask.data(),
      (float*)cudaBlendedFloat.data(),
      context);

#if 1 /* perf test */
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    cudaLaplacianBlendWithContext(
        (const float*)cudaImage1Float.data(),
        (const float*)cudaImage2Float.data(),
        (const float*)cudaMask.data(),
        (float*)cudaBlendedFloat.data(),
        context);
  }

  auto stop_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  float ms = stop_ms - start_ms;
  float sec_per_frame = (ms / 1000) / frame_count;
  std::cout << "Blend speed: " << (1.0 / sec_per_frame) << "fps" << std::endl;
#endif

  // Convert the blended image from float back to 8–bit for saving.
  cv::Mat blended;
  blended_float = cudaBlendedFloat.download();
  blended_float.convertTo(blended, CV_8UC3, 255.0);

  // Save the final blended image.
  if (!cv::imwrite(argv[4], blended)) {
    std::cerr << "Failed to save the blended image!" << std::endl;
    return -1;
  }

  std::cout << "Blended image saved as: " << argv[4] << std::endl;
  return 0;
}

void test_remapping() {
  // Define image dimensions.
  const int srcW = 4, srcH = 4; // Source image dimensions.
  const int destW = 4, destH = 4; // Destination image dimensions.

  // Allocate and initialize the host source image.
  // Each pixel has 3 channels (RGB) stored as floats.
  float h_src[srcW * srcH * 3];
  for (int y = 0; y < srcH; y++) {
    for (int x = 0; x < srcW; x++) {
      int idx = (y * srcW + x) * 3;
      // Set the pixel to (x, y, x+y)
      h_src[idx + 0] = static_cast<float>(x);
      h_src[idx + 1] = static_cast<float>(y);
      h_src[idx + 2] = static_cast<float>(x + y);
    }
  }

  // Allocate and initialize the host destination image.
  float h_dest[destW * destH * 3] = {0};

  // Allocate and initialize the host mapping arrays (unsigned short).
  // For each destination pixel (x, y), we want to map to source pixel (x-1, y-1).
  // If (x-1) or (y-1) is negative, we set the mapping to an out-of-range value.
  unsigned short h_mapX[destW * destH];
  unsigned short h_mapY[destW * destH];
  for (int y = 0; y < destH; y++) {
    for (int x = 0; x < destW; x++) {
      int idx = y * destW + x;
      int mapXVal = x - 1;
      int mapYVal = y - 1;
      // If the mapping is negative, assign an out-of-range value.
      h_mapX[idx] = (mapXVal < 0) ? static_cast<unsigned short>(srcW) : static_cast<unsigned short>(mapXVal);
      h_mapY[idx] = (mapYVal < 0) ? static_cast<unsigned short>(srcH) : static_cast<unsigned short>(mapYVal);
    }
  }

  // Allocate device memory.
  float *d_src = nullptr, *d_dest = nullptr;
  unsigned short *d_mapX = nullptr, *d_mapY = nullptr;
  cudaMalloc(&d_src, sizeof(float) * srcW * srcH * 3);
  cudaMalloc(&d_dest, sizeof(float) * destW * destH * 3);
  cudaMalloc(&d_mapX, sizeof(unsigned short) * destW * destH);
  cudaMalloc(&d_mapY, sizeof(unsigned short) * destW * destH);

  // Copy data from host to device.
  cudaMemcpy(d_src, h_src, sizeof(float) * srcW * srcH * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mapX, h_mapX, sizeof(unsigned short) * destW * destH, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mapY, h_mapY, sizeof(unsigned short) * destW * destH, cudaMemcpyHostToDevice);

  // Define kernel launch configuration.
  // dim3 blockDim(16, 16);
  // dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y);

  // Set default color for unmapped pixels.
  float defaultR = 100.0f, defaultG = 100.0f, defaultB = 100.0f;

  // Launch the remap kernel.
  remap_kernel(d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defaultR, defaultG, defaultB);

  // Wait for the kernel to finish.
  cudaDeviceSynchronize();

  // Copy the destination image back to host memory.
  cudaMemcpy(h_dest, d_dest, sizeof(float) * destW * destH * 3, cudaMemcpyDeviceToHost);

  // Print out the destination image.
  // Each pixel is printed as (R, G, B).
  std::cout << "Destination image:" << std::endl;
  for (int y = 0; y < destH; y++) {
    for (int x = 0; x < destW; x++) {
      int idx = (y * destW + x) * 3;
      std::cout << "(" << h_dest[idx + 0] << ", " << h_dest[idx + 1] << ", " << h_dest[idx + 2] << ") ";
    }
    std::cout << std::endl;
  }

  // Clean up device memory.
  cudaFree(d_src);
  cudaFree(d_dest);
  cudaFree(d_mapX);
  cudaFree(d_mapY);
}
