#include <opencv2/opencv.hpp>

#include "cudaBlend.h"
#include "glDisplay.h"
#include "imageFormat.h"
#include "videoOutput.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <cuda_runtime.h>
#include <opencv4/opencv2/highgui.hpp>

#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>

#include <opencv4/opencv2/imgcodecs.hpp>
#include <tiffio.h>
// #include <geotiff/geotiff.h>
//  #include <xtiffio.h>

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

// #ifndef TIFFTAG_GEOTIEPOINTS
// // The GeoTIFF ModelTiepointTag is defined as tag number 33922.
// #define TIFFTAG_GEOTIEPOINTS 33922
// #endif

// #ifndef TIFFTAG_GEOPIXELSCALE
// // The GeoTIFF ModelPixelScaleTag is defined as tag number 33550.
// #define TIFFTAG_GEOPIXELSCALE 33550
// #endif

// A structure to hold TIFF information
struct TiffInfo {
  // Resolution information
  bool validResolution = false;
  float xResolution = 0.0f;
  float yResolution = 0.0f;
  // Resolution unit (e.g., RESUNIT_INCH, RESUNIT_CENTIMETER)
  uint16_t resolutionUnit = 0;

  // GeoTIFF Tiepoints (each group of 6 values maps image to model coordinates)
  bool hasGeoTiePoints = false;
  float xPosition{0};
  float yPosition{0};
};

// Function that takes a file name and returns the TIFF information.
TiffInfo getTiffInfo(const std::string& filename) {
  TiffInfo info;
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if (!tif) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return info;
  }

  // --- Get Resolution Information ---
  float xres = 0.0f, yres = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XRESOLUTION, &xres) && TIFFGetField(tif, TIFFTAG_YRESOLUTION, &yres)) {
    info.xResolution = xres;
    info.yResolution = yres;
    info.validResolution = true;
  }

  uint16_t resUnit = 0;
  if (TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &resUnit)) {
    info.resolutionUnit = resUnit;
  }

  float xpos = 0.0f, ypos = 0.0f;
  if (TIFFGetField(tif, TIFFTAG_XPOSITION, &xpos)) {
    std::cout << "X Position: " << xpos << std::endl;
    info.xPosition = xpos;
  } else {
    std::cout << "No X Position information found." << std::endl;
  }

  if (TIFFGetField(tif, TIFFTAG_YPOSITION, &ypos)) {
    std::cout << "Y Position: " << ypos << std::endl;
    info.yPosition = ypos;
  } else {
    std::cout << "No Y Position information found." << std::endl;
  }

  TIFFClose(tif);
  return info;
}

namespace {

class CudaMat {
 private:
  void* d_data{nullptr};
  size_t size;
  int rows_, cols_, type_;
  int batch_size_{1};

 public:
  CudaMat(const cv::Mat& mat, bool copy = true) : rows_(mat.rows), cols_(mat.cols), type_(mat.type()) {
    size = mat.total() * mat.elemSize();
    cudaMalloc(&d_data, size);
    assert(mat.isContinuous());
    if (copy) {
      cudaMemcpy(d_data, mat.data, size, cudaMemcpyHostToDevice);
    }
  }

  CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy = true) : batch_size_(mat_batch.size()) {
    assert(batch_size_);
    const cv::Mat& first = mat_batch.at(0);
    rows_ = first.rows;
    cols_ = first.cols;
    type_ = first.type();
    const size_t size_each = first.total() * first.elemSize();
    const size_t size_total = size_each * batch_size_;
    cudaMalloc(&d_data, size_total);
    if (copy) {
      uint8_t* p = (uint8_t*)d_data;
      for (const cv::Mat& mat : mat_batch) {
        assert(mat.isContinuous());
        cudaMemcpy(p, mat.data, size_each, cudaMemcpyHostToDevice);
        p += size_each;
      }
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

struct SpatialTiff {
  // position in pixels
  float xpos;
  float ypos;
};

std::vector<SpatialTiff> normalize(std::vector<SpatialTiff>&& positions) {
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  std::for_each(positions.begin(), positions.end(), [&](const SpatialTiff& sp) {
    min_x = std::min(min_x, sp.xpos);
    min_y = std::min(min_y, sp.ypos);
  });
  std::for_each(positions.begin(), positions.end(), [&](SpatialTiff& sp) {
    sp.xpos -= min_x;
    sp.ypos -= min_y;
  });
  return positions;
}

std::tuple<float, float> get_canvas_size(const std::vector<SpatialTiff>& positions) {
  float max_x = 0;
  float max_y = 0;
  std::for_each(positions.begin(), positions.end(), [&](const SpatialTiff& sp) {
    assert(sp.xpos >= 0);
    assert(sp.ypos >= 0);
    max_x = std::max(max_x, sp.xpos);
    assert(sp.xpos >= 0);
    max_y = std::max(max_y, sp.ypos);
  });
  return {max_x, max_y};
}

SpatialTiff get_geo_tiff(const std::string& filename) {
  TiffInfo info = getTiffInfo(filename);
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

// Example structures for canvas and remapper info.
// Adjust these as needed for your application.
struct CanvasInfo {
  int width;
  int height;
  // We assume that positions[1].x will be used below.
  std::vector<cv::Point> positions;
};

struct Remapper {
  int width;
};

class MaskConverter {
 public:
  // Members that hold canvas, blending, and remapper information.
  CanvasInfo _canvas_info;
  bool _minimize_blend;
  int _overlap_pad;
  Remapper _remapper_1;

  // The function converts a mask cv::Mat so that it has the same size as
  // _canvas_info, padding if necessary (using replication) and then, if
  // _minimize_blend is true, returns a horizontally cropped region.
  cv::Mat convertMaskMat(const cv::Mat& mask) {
    int padw = 0, padh = 0;
    // In OpenCV, image width is the number of columns and height is the number of rows.
    int mwidth = mask.cols;
    int mheight = mask.rows;

    // Ensure that the incoming mask is not larger than the canvas.
    assert(mwidth <= _canvas_info.width);
    assert(mheight <= _canvas_info.height);

    if (mwidth < _canvas_info.width)
      padw = _canvas_info.width - mwidth;
    if (mheight < _canvas_info.height)
      padh = _canvas_info.height - mheight;

    cv::Mat paddedMask;
    if (padw > 0 || padh > 0) {
      // In PyTorch the code uses replication padding on the right (width)
      // and bottom (height). In OpenCV, we use copyMakeBorder with:
      // top = 0, bottom = padh, left = 0, right = padw, and BORDER_REPLICATE.
      cv::copyMakeBorder(mask, paddedMask, 0, padh, 0, padw, cv::BORDER_REPLICATE);
    } else {
      paddedMask = mask;
    }

    // Check that the padded mask now matches the canvas dimensions.
    assert(paddedMask.cols == _canvas_info.width);
    assert(paddedMask.rows == _canvas_info.height);

    // If we are not minimizing the blend, return the full mask.
    if (!_minimize_blend)
      return paddedMask;

    // Otherwise, take a slice of the mask.
    // In the original Python code the slice is taken along the width dimension:
    // mask[..., x_start : x_end] where:
    //   x_start = self._canvas_info.positions[1].x - self._overlap_pad
    //   x_end   = self._remapper_1.width + self._overlap_pad
    int x_start = _canvas_info.positions[1].x - _overlap_pad;
    int x_end = _remapper_1.width + _overlap_pad;
    // For safety, you may want to add additional bounds checks.
    assert(x_start >= 0 && x_end <= paddedMask.cols);

    // Define a ROI (region of interest) covering all rows and columns [x_start, x_end)
    cv::Rect roi(x_start, 0, x_end - x_start, paddedMask.rows);
    return paddedMask(roi);
  }
};

int main(int argc, char** argv) {
  // Usage check.
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2> <mask> <output>" << std::endl;
    return -1;
  }

  RenderSet display;

  std::string game_id = "stitch-fix";
  std::string game_dir = std::string(::getenv("HOME")) + "/Videos/" + game_id + "/";

  std::string mapping_0_pos = game_dir + "mapping_0000.tif";
  std::string mapping_0_x = game_dir + "mapping_0000_x.tif";
  std::string mapping_0_y = game_dir + "mapping_0000_y.tif";
  std::string mapping_1_pos = game_dir + "mapping_0001.tif";
  std::string mapping_1_x = game_dir + "mapping_0001_x.tif";
  std::string mapping_1_y = game_dir + "mapping_0001_y.tif";
  std::string whole_seam_mask = game_dir + "seam_file.png";

  std::string sample_img_left_path = game_dir + "GX010100.png";
  std::string sample_img_right_path = game_dir + "GX010019.png";

  // Normalize
  std::vector<SpatialTiff> positions{get_geo_tiff(mapping_0_pos), get_geo_tiff(mapping_1_pos)};
  positions = normalize(std::move(positions));

  cv::Mat img1_col = cv::imread(mapping_0_x, cv::IMREAD_ANYDEPTH);
  assert(img1_col.type() == CV_16U);
  cv::Mat img1_row = cv::imread(mapping_0_y, cv::IMREAD_ANYDEPTH);
  cv::Mat img2_col = cv::imread(mapping_1_x, cv::IMREAD_ANYDEPTH);
  cv::Mat img2_row = cv::imread(mapping_1_y, cv::IMREAD_ANYDEPTH);
  cv::Mat whole_seam_mask_image = cv::imread(whole_seam_mask, cv::IMREAD_ANYDEPTH);

  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  sample_img_left.convertTo(sample_img_left, CV_32FC3, 1.0 / 255.0);
  sample_img_right.convertTo(sample_img_right, CV_32FC3, 1.0 / 255.0);

  // Compute canvas size
  const size_t canvas_width = std::max(positions[0].xpos + img1_col.cols, positions[1].xpos + img2_col.cols);
  const size_t canvas_height = std::max(positions[0].ypos + img1_col.rows, positions[1].ypos + img2_col.rows);
  std::cout << "Canvas size: " << canvas_width << " x " << canvas_height << std::endl;

  // compute overlap size
  // if self._minimize_blend:
  //     self._x1, self._y1, self._x2, self._y2 = (
  //         self._canvas_info.positions[0].x,
  //         self._canvas_info.positions[0].y,
  //         self._canvas_info.positions[1].x,
  //         self._canvas_info.positions[1].y,
  //     )

  //     self._remapper_1.xpos = self._x1
  //     self._remapper_2.xpos = self._x1 + self._overlap_pad  # start overlapping right away
  //     width_1 = self._remapper_1.width
  //     self._overlapping_width = width_1 - self._x2
  //     assert width_1 > self._x2
  //     # seam tensor box (box we'll be blending)
  //     self._padded_blended_tlbr = [
  //         self._x2 - self._overlap_pad,  # x1
  //         max(0, min(self._y1, self._y2) - self._overlap_pad),  # y1
  //         width_1 + self._overlap_pad,  # x2
  //         min(
  //             self._canvas_info.height,
  //             max(self._y1 + self._remapper_1.height, self._y2 + self._remapper_2.height)
  //             + self._overlap_pad,
  //         ),  # y2
  //     ]
  //     assert self._x2 - self._overlap_pad >= 0
  //     assert width_1 + self._overlap_pad <= self._canvas_info.width

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

  CudaMat remap_1_x(img1_col), remap_1_y(img1_row);
  CudaMat remap_2_x(img2_col), remap_2_y(img2_row);

  cv::Mat remapped_1(img1_col.size(), CV_32FC3);
  cv::Mat remapped_2(img2_col.size(), CV_32FC3);
  CudaMat cudaRemapped_1(remapped_1, /*copy=*/false);
  CudaMat cudaRemapped_2(remapped_2, /*copy=*/false);

  CudaMat sampleImage1(sample_img_left);
  CudaMat sampleImage2(sample_img_right);

  CudaMat cudaImage1Float(img1_float);
  CudaMat cudaImage2Float(img2_float);
  CudaMat cudaMask(seam_mask);

  // Prepare the output image (as float).
  cv::Mat blended_float(img1.size(), CV_32FC3);
  CudaMat cudaBlendedFloat(blended_float);

  cudaDeviceSynchronize();

  // Set default color for unmapped pixels.
  float defaultR = 128.0f, defaultG = 128.0f, defaultB = 128.0f;

  // Launch the remap kernel.
  remap_kernel(
      (float*)sampleImage1.data(),
      sampleImage1.width(),
      sampleImage1.height(),
      (float*)cudaRemapped_1.data(),
      cudaRemapped_1.width(),
      cudaRemapped_1.height(),
      (uint16_t*)remap_1_x.data(),
      (uint16_t*)remap_1_y.data(),
      defaultR,
      defaultG,
      defaultB);

  remap_kernel(
      (float*)sampleImage2.data(),
      sampleImage2.width(),
      sampleImage2.height(),
      (float*)cudaRemapped_2.data(),
      cudaRemapped_2.width(),
      cudaRemapped_2.height(),
      (uint16_t*)remap_2_x.data(),
      (uint16_t*)remap_2_y.data(),
      defaultR,
      defaultG,
      defaultB);

  // cudaDeviceSynchronize();

  // auto disp = cudaRemapped_1.download();
  // auto disp = cudaRemapped_2.download();
  // auto disp = sampleImage2.download();
  // disp.convertTo(disp, CV_8UC3, 255.0);
  // cv::imshow("remapped", disp);
  // cv::waitKey(0);

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
#if 1
    remap_kernel(
        (float*)sampleImage1.data(),
        sampleImage1.width(),
        sampleImage1.height(),
        (float*)cudaRemapped_1.data(),
        cudaRemapped_1.width(),
        cudaRemapped_1.height(),
        (uint16_t*)remap_1_x.data(),
        (uint16_t*)remap_1_y.data(),
        defaultR,
        defaultG,
        defaultB);
    remap_kernel(
        (float*)sampleImage2.data(),
        sampleImage2.width(),
        sampleImage2.height(),
        (float*)cudaRemapped_2.data(),
        cudaRemapped_2.width(),
        cudaRemapped_2.height(),
        (uint16_t*)remap_2_x.data(),
        (uint16_t*)remap_2_y.data(),
        defaultR,
        defaultG,
        defaultB);
#endif

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
