#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "cudaBlend.h"

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
  int numLevels = 4;
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

  // Convert the blended image from float back to 8–bit for saving.
  cv::Mat blended;
  blended_float = cudaBlendedFloat.download();
  blended_float.convertTo(blended, CV_8UC3, 255.0);

  // Save the final blended image.
  if (!cv::imwrite(argv[3], blended)) {
    std::cerr << "Failed to save the blended image!" << std::endl;
    return -1;
  }

  std::cout << "Blended image saved as " << argv[4] << std::endl;
  return 0;
}
