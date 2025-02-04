#include <opencv2/opencv.hpp>

#include "cudaBlend.h"
#include "glDisplay.h"
#include "imageFormat.h"
#include "videoOutput.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>
#include <opencv4/opencv2/highgui.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

int kbhit() {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

int wait_key() {
  int c;
  while (!(c = kbhit())) {
    usleep(100);
  }
  return c;
}

void show_image(const std::string& label, const cv::Mat& img, bool wait = true) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

namespace {

class CudaMat {
 private:
  void* d_data{nullptr};
  size_t size;
  int rows_, cols_, type_;
  int batch_size_{1};

 public:
  CudaMat(const cv::Mat& mat) : rows_(mat.rows), cols_(mat.cols), type_(mat.type()) {
    size = mat.total() * mat.elemSize();
    cudaMalloc(&d_data, size);
    assert(mat.isContinuous());
    cudaMemcpy(d_data, mat.data, size, cudaMemcpyHostToDevice);
  }

  CudaMat(const std::vector<cv::Mat>& mat_batch) : batch_size_(mat_batch.size()) {
    assert(batch_size_);
    const cv::Mat& first = mat_batch.at(0);
    rows_ = first.rows;
    cols_ = first.cols;
    type_ = first.type();
    const size_t size_each = first.total() * first.elemSize();
    const size_t size_total = size_each * batch_size_;
    cudaMalloc(&d_data, size_total);
    uint8_t* p = (uint8_t*)d_data;
    for (const cv::Mat& mat : mat_batch) {
      assert(mat.isContinuous());
      cudaMemcpy(p, mat.data, size_each, cudaMemcpyHostToDevice);
      p += size_each;
    }
  }

  ~CudaMat() {
    if (d_data) {
      cudaFree(d_data);
    }
  }

  cv::Mat download() const {
    cv::Mat mat(rows_, cols_, type_);
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
  constexpr int width() const {
    return cols_;
  }
  constexpr int height() const {
    return rows_;
  }
  constexpr int type() const {
    return type_;
  }
  constexpr int batch_size() const {
    return batch_size_;
  }
};

imageFormat get_image_format(const int cv_type) {
  switch (cv_type) {
    case CV_8UC3:
      return imageFormat::IMAGE_RGB8;
    case CV_8UC4:
      return imageFormat::IMAGE_RGBA8;
    case CV_32FC3:
      return imageFormat::IMAGE_RGB32F;
    case CV_32FC4:
      return imageFormat::IMAGE_RGBA32F;
    default:
      assert(false);
  }
}

struct CudaSurface {
  CudaSurface(int w, int h, imageFormat format, void* data)
      : width(w), height(h), image_format(format), dataptr(data) {}
  CudaSurface(const CudaMat& cm)
      : width(cm.width()), height(cm.height()), image_format(get_image_format(cm.type())), dataptr((void*)cm.data()) {}
  int width{0};
  int height{0};
  imageFormat image_format;
  void* dataptr{nullptr};
};

class RenderSet {
 public:
  void render(const std::string& name, const CudaSurface& surface, cudaStream_t stream = 0) {
    get_video_output(name, surface.width, surface.height)
        ->Render((void*)surface.dataptr, surface.width, surface.height, surface.image_format, stream);
  }

 private:
  static std::unique_ptr<glDisplay> create_video_output(const std::string& name, int width, int height) {
    videoOptions vo;
    vo.width = width;
    vo.height = height;
    auto video_output = std::unique_ptr<glDisplay>(glDisplay::Create(vo));
    video_output->SetTitle(name.c_str());
    return video_output;
  }

  videoOutput* get_video_output(const std::string& name, int width, int height) {
    std::unique_lock lk(mu_);
    auto found = video_outputs_.find(name);
    if (found == video_outputs_.end()) {
      found = video_outputs_.emplace(name, create_video_output(name, width, height)).first;
    }
    return found->second.get();
  }

  std::mutex mu_;
  std::map<std::string, std::unique_ptr<glDisplay>> video_outputs_;
};

} // namespace

void test_remapping();
// cudaError_t cudaLaplacianBlend(const float* image1, const float* image2,
//                                const float* mask, float* output, int
//                                imageWidth, int imageHeight, int numLevels);

cv::Mat load_seam_mask(const std::string& filename) {
  cv::Mat seam_mask = cv::imread(filename, cv::IMREAD_ANYDEPTH);
  if (!seam_mask.empty()) {
    // show_image("seam_mask", seam_mask);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    // Get the minimum and maximum values and their locations
    cv::minMaxLoc(seam_mask, &minVal, &maxVal, &minLoc, &maxLoc);

    // Create masks for min and max values
    cv::Mat minMask = (seam_mask == minVal); // Mask for min value
    cv::Mat maxMask = (seam_mask == maxVal); // Mask for max value

    // Set all min values to 0 and max values to 1
    // 1's to left, 0's to right (invert)
    seam_mask.setTo(0, maxMask); // Set min value locations to 0
    seam_mask.setTo(1, minMask); // Set max value locations to 1

    // cv::minMaxLoc(seam_mask, &minVal, &maxVal, &minLoc, &maxLoc);
    // printf("x=%d, m=%f\n", x, m);show_image("seam_mask", seam_mask * 255);
    // usleep(0);
  }
  return seam_mask;
}

std::pair<double, double> get_min_max(const cv::Mat& mat) {
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;

  // Get the minimum and maximum values and their locations
  cv::minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
  return std::make_pair(minVal, maxVal);
}

int countUniqueValues(cv::Mat mat) {
  std::set<int> uniqueValues;

  // Assume the matrix type is CV_32S (32-bit signed integer)
  for (int i = 0; i < mat.cols; ++i) {
    for (int j = 0; j < mat.rows; ++j) {
      uniqueValues.insert(mat.at<float>(i, j));
    }
  }

  return uniqueValues.size();
}

cv::Mat load_position_mask(const std::string& filename, double* minVal, double* maxVal) {
  cv::Mat pos_mask = cv::imread(filename, cv::IMREAD_ANYDEPTH);
  if (!pos_mask.empty()) {
    if (minVal || maxVal) {
      cv::Point minLoc, maxLoc;
      // Get the minimum and maximum values and their locations
      double min, max;
      cv::minMaxLoc(pos_mask, &min, &max, &minLoc, &maxLoc);
      if (minVal) {
        *minVal = min;
      }
      if (maxVal) {
        *maxVal = max;
      }
    }
  } else {
    if (minVal) {
      *minVal = std::nan("");
    }
    if (maxVal) {
      *maxVal = std::nan("");
    }
  }
  return pos_mask;
}

int main(int argc, char** argv) {
  // Usage check.
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2> <mask> <output>" << std::endl;
    return -1;
  }

  RenderSet display;

  std::string game_id = "stitch_fix";

  // assert(false);
  //  Load the two images (in color).
  cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
  cv::Mat seam_mask = load_seam_mask(argv[3]);

  // show_image("left", img1, false);
  // show_image("right", img2, true);
  // show_image("seam_mask", seam_mask);

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
  cv::Mat img1_float, img2_float;
  img1.convertTo(img1_float, CV_32FC3, 1.0 / 255.0);
  img2.convertTo(img2_float, CV_32FC3, 1.0 / 255.0);

  seam_mask.convertTo(seam_mask, CV_32FC1);
  auto minmax = get_min_max(seam_mask);
  std::cout << "min=" << minmax.first << ", max=" << minmax.second
            << ", unique val count=" << countUniqueValues(seam_mask) << std::endl;

  // float left = seam_mask.at<float>(0,0);
  // float right = seam_mask.at<float>(0,seam_mask.cols - 1);

  // Create a simple seam mask (single–channel, CV_32FC1):
  // Here we use a hard–coded seam: the left half of the image is taken entirely
  // from image1 (mask value 1.0) and the right half from image2 (mask value
  // 0.0). In a more complex case, the mask can be generated based on feature
  // detection or user input. cv::Mat mask(img1.size(), CV_32FC1); for (int y =
  // 0; y < mask.rows; y++) {
  //   for (int x = 0; x < mask.cols; x++) {
  //     mask.at<float>(y, x) = (x < mask.cols / 2) ? 1.0f : 0.0f;
  //   }
  // }

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

// Configurable parameter: number of pyramid levels.
#ifdef TEGRA
  // Lower compute, quick and dirty
  int numLevels = 1;
#else
  int numLevels = 6;
#endif
  int width = img1.cols;
  int height = img1.rows;

  // CudaLaplacianBlendContext context(width, height, numLevels);
  CudaBatchLaplacianBlendContext context(width, height, numLevels, /*batch_size=*/1);

  CudaMat cudaImage1Float(img1_float);
  CudaMat cudaImage2Float(img2_float);
  CudaMat cudaMask(seam_mask);

  // Prepare the output image (as float).
  cv::Mat blended_float(img1.size(), CV_32FC3);
  CudaMat cudaBlendedFloat(blended_float);

  cudaDeviceSynchronize();

  // cv::imshow("img1", img1_float);
  // cv::waitKey(0);

  // display.render("cudaImage1Float", CudaSurface(cudaImage1Float), stream);
  // wait_key();
  //  cv::waitKey(10);
  //  char c;
  //  std::cin >> c;

  // Call the CUDA–based blending function.
  // It is assumed that blendImages copies data to/from device memory,
  // launches the appropriate kernels, and returns the blended image.

  auto cu_err = cudaBatchedLaplacianBlendWithContext(
      (const float*)cudaImage1Float.data(),
      (const float*)cudaImage2Float.data(),
      (const float*)cudaMask.data(),
      (float*)cudaBlendedFloat.data(),
      context);
  // auto cu_err = cudaLaplacianBlendWithContext(
  //     (const float*)cudaImage1Float.data(),
  //     (const float*)cudaImage2Float.data(),
  //     (const float*)cudaMask.data(),
  //     (float*)cudaBlendedFloat.data(),
  //     context);

#if 1 /* perf test */
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    cudaBatchedLaplacianBlendWithContext(
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

  // cv::imshow("blended_float", blended_float);
  // cv::waitKey(0);

  blended_float.convertTo(blended, CV_8UC3, 255.0);

  // show_image("blended_float", blended_float);

  // Save the final blended image.
  if (!cv::imwrite(argv[4], blended)) {
    std::cerr << "Failed to save the blended image!" << std::endl;
    return -1;
  }

  std::cout << "Blended image saved as: " << argv[4] << std::endl;
  test_remapping();
  return cu_err;
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
  // For each destination pixel (x, y), we want to map to source pixel (x-1,
  // y-1). If (x-1) or (y-1) is negative, we set the mapping to an out-of-range
  // value.
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
  // dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y -
  // 1) / blockDim.y);

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
  // std::cout << "Done. Press a key." << std::endl;
  // char c;
  // std::cin >> c;
  // std::cout << "Exiting..." << std::endl;
}
