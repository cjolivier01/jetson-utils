#include "cudaBlend.h"
#include "cudaMakeFull.h"
#include "cudaMat.h"
#include "cudaRemap.h"
#include "cudaStatus.h"
#include "glDisplay.h"
#include "imageFormat.h"
#include "imageFormat.h" // Assumed to define the jetson‑utils imageFormat enum
#include "videoOutput.h"

#include <cuda_runtime.h> // for CUDA vector types
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/imgcodecs.hpp>
#include <tiffio.h>

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

// Function parameters:
//   - image1, image2: the two RGB images to be adjusted.
//   - seam: the seam mask image (CV_8U) that is larger than both images.
//   - N: number of pixels to sample on each side of the seam.
//   - topLeft1, topLeft2: the (x,y) coordinates (relative to the seam mask)
//                           of the top left corners of image1 and image2, respectively.
void matchSeamImages(
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam,
    int N,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2) {
  // Ensure the seam mask is of type CV_8U.
  if (seam.type() != CV_8U) {
    std::cerr << "Error: Seam mask must be of type CV_8U." << std::endl;
    return;
  }

  // Accumulators for summing per-channel pixel values.
  cv::Scalar sumLeft(0, 0, 0); // For image1 samples (left of the seam).
  cv::Scalar sumRight(0, 0, 0); // For image2 samples (right of the seam).
  int countLeft = 0, countRight = 0;

  // ----- Process image1 (sampling from the left side of the seam) -----
  // Only examine the middle 50% of image1's rows.
  int startRow1 = image1.rows / 4;
  int endRow1 = (3 * image1.rows) / 4;
  for (int r = startRow1; r < endRow1; r++) {
    // Map image1’s local row (r) to the seam mask’s row coordinate.
    int globalRow = topLeft1.y + r;
    if (globalRow < 0 || globalRow >= seam.rows)
      continue; // Row is outside the seam mask.

    // Define the horizontal span of image1 in the seam mask.
    int colStart = topLeft1.x;
    int colEnd = topLeft1.x + image1.cols;

    // Find the seam boundary: the first column (within image1’s span)
    // where the seam mask pixel equals 1.
    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols)
        continue;
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1)
      continue; // No seam boundary found for this row.

    // Convert the global seam column to image1’s local coordinate.
    int seamLocalCol = seamGlobalCol - topLeft1.x;

    // Sample up to N pixels immediately to the left of the seam boundary.
    int sampleStart = std::max(0, seamLocalCol - N);
    for (int c = sampleStart; c < seamLocalCol; c++) {
      // Safety check.
      if (c < 0 || c >= image1.cols)
        continue;

      // Depending on the image depth, read the pixel appropriately.
      if (image1.depth() == CV_8U) {
        cv::Vec3b pixel = image1.at<cv::Vec3b>(r, c);
        sumLeft[0] += pixel[0];
        sumLeft[1] += pixel[1];
        sumLeft[2] += pixel[2];
      } else if (image1.depth() == CV_32F) {
        cv::Vec3f pixel = image1.at<cv::Vec3f>(r, c);
        sumLeft[0] += pixel[0];
        sumLeft[1] += pixel[1];
        sumLeft[2] += pixel[2];
      }
      countLeft++;
    }
  }

  // ----- Process image2 (sampling from the right side of the seam) -----
  // Only examine the middle 50% of image2's rows.
  int startRow2 = image2.rows / 4;
  int endRow2 = (3 * image2.rows) / 4;
  for (int r = startRow2; r < endRow2; r++) {
    // Map image2’s local row (r) to the seam mask’s row coordinate.
    int globalRow = topLeft2.y + r;
    if (globalRow < 0 || globalRow >= seam.rows)
      continue;

    // Define the horizontal span of image2 in the seam mask.
    int colStart = topLeft2.x;
    int colEnd = topLeft2.x + image2.cols;

    // Find the seam boundary in image2’s region of the seam mask.
    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols)
        continue;
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1)
      continue; // No seam boundary found in this row.

    // Convert the global seam column to image2’s local coordinate.
    int seamLocalCol = seamGlobalCol - topLeft2.x;

    // Sample up to N pixels immediately to the right of the seam boundary.
    int sampleEnd = std::min(image2.cols, seamLocalCol + N);
    for (int c = seamLocalCol; c < sampleEnd; c++) {
      if (c < 0 || c >= image2.cols)
        continue;

      if (image2.depth() == CV_8U) {
        cv::Vec3b pixel = image2.at<cv::Vec3b>(r, c);
        sumRight[0] += pixel[0];
        sumRight[1] += pixel[1];
        sumRight[2] += pixel[2];
      } else if (image2.depth() == CV_32F) {
        cv::Vec3f pixel = image2.at<cv::Vec3f>(r, c);
        sumRight[0] += pixel[0];
        sumRight[1] += pixel[1];
        sumRight[2] += pixel[2];
      }
      countRight++;
    }
  }

  // Check that we have collected samples from both images.
  if (countLeft == 0 || countRight == 0) {
    std::cerr << "Error: Not enough seam samples collected for adjustment." << std::endl;
    return;
  }

  // Compute per-channel averages.
  cv::Scalar avgLeft = sumLeft * (1.0 / countLeft);
  cv::Scalar avgRight = sumRight * (1.0 / countRight);
  std::cout << "Average values (Image1, left side): " << avgLeft << std::endl;
  std::cout << "Average values (Image2, right side): " << avgRight << std::endl;

  // Compute an offset per channel (half the difference).
  // The idea is to subtract this offset from image1 and add it to image2.
  cv::Scalar offset = (avgLeft - avgRight) * 0.5;
  std::cout << "Offset: " << offset << std::endl;

  // Helper lambda: adjusts an image by a per-channel amount.
  auto adjustImage = [&](cv::Mat& img, cv::Scalar adjustment) {
    if (img.depth() == CV_8U) {
      cv::Mat floatImg;
      img.convertTo(floatImg, CV_32F);
      std::vector<cv::Mat> channels;
      cv::split(floatImg, channels);
      for (int i = 0; i < 3; i++) {
        channels[i] += static_cast<float>(adjustment[i]);
      }
      cv::merge(channels, floatImg);
      // Clamp the adjusted values to the valid range [0,255].
      cv::min(floatImg, 255.0, floatImg);
      cv::max(floatImg, 0.0, floatImg);
      floatImg.convertTo(img, CV_8U);
    } else if (img.depth() == CV_32F) {
      std::vector<cv::Mat> channels;
      cv::split(img, channels);
      for (int i = 0; i < 3; i++) {
        channels[i] += static_cast<float>(adjustment[i]);
      }
      cv::merge(channels, img);
    }
  };

  // Adjust the images: subtract the offset from image1 and add it to image2.
  adjustImage(image1, -offset);
  adjustImage(image2, offset);
}

void show_image(const std::string& label, const cv::Mat& img, bool wait = true) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

void displayScaledImage(const std::string& label, cv::Mat image, float scale = 1.0, bool wait = true) {
  if (scale != 1.0f) {
    // Calculate new dimensions
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);

    // Resize the image
    cv::resize(image, image, cv::Size(newWidth, newHeight));
  }

  // Display the image
  cv::imshow(label, image);
  cv::waitKey(wait ? 0 : 1); // Wait for a keystroke in the window
}

#define SHOW_IMAGE(_mat$)                                                \
  do {                                                                   \
    show_image(std::string(#_mat$), (_mat$)->download(), /*wait=*/true); \
  } while (false)

#define SHOW_SMALL(_mat$)                                                              \
  do {                                                                                 \
    displayScaledImage(std::string(#_mat$), (_mat$)->download(), 0.05, /*wait=*/true); \
  } while (false)

namespace {

template <typename T>
struct CudaSurface {
  CudaSurface(int w, int h, imageFormat format, void* data)
      : width(w), height(h), image_format(format), dataptr(data) {}
  CudaSurface(const CudaMat<T>& cm)
      : width(cm.width()), height(cm.height()), image_format(get_image_format(cm.type())), dataptr((void*)cm.data()) {}
  int width{0};
  int height{0};
  imageFormat image_format;
  void* dataptr{nullptr};
};

class RenderSet {
 public:
  template <typename T>
  void render(const std::string& name, const CudaSurface<T>& surface, cudaStream_t stream = 0) {
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

cv::Mat load_seam_mask(const std::string& filename) {
  cv::Mat seam_mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (!seam_mask.empty()) {
    // show_image("seam_mask", seam_mask);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    // Get the minimum and maximum values and their locations
    cv::minMaxLoc(seam_mask, &minVal, &maxVal, &minLoc, &maxLoc);

    // Create masks for min and max values
    cv::Mat minMask = (seam_mask == (int)minVal); // Mask for min value
    cv::Mat maxMask = (seam_mask == (int)maxVal); // Mask for max value

    // Set all min values to 0 and max values to 1
    // 1's to left, 0's to right (invert)
    seam_mask.setTo(0, maxMask); // Set min value locations to 0
    seam_mask.setTo(1, minMask); // Set max value locations to 1
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

template <typename T>
std::set<T> get_unique_values(const cv::Mat& mat, const std::unordered_set<T>& ignore = {}) {
  std::set<T> unique_values;

  // Check if the data type of the matrix matches the template type
  if (mat.type() != cv::DataType<T>::type) {
    throw std::invalid_argument("Matrix data type does not match the template type T");
  }

  // Iterate over each element in the matrix
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      T value = mat.at<T>(i, j);
      // Add to set if not in ignore set
      if (ignore.find(value) == ignore.end()) {
        unique_values.insert(value);
      }
    }
  }

  return unique_values;
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

// Structure to hold canvas information.
struct CanvasInfo {
  int width{0};
  int height{0};
  // Assume positions[0] and positions[1] are valid cv::Point's with x and y coordinates.
  std::vector<cv::Point> positions;
};

// Structure to hold remapper parameters.
struct Remapper {
  int width{0};
  int height{0};
  int xpos{0}; // This will be set by the blend logic.
};

class CanvasManager {
 public:
  // Canvas and blending parameters.

  // Two remappers (for example, for two image streams).
  Remapper _remapper_1;
  Remapper _remapper_2;

  // Additional members for blending logic.
  int _x1{0}, _y1{0}, _x2{0}, _y2{0};
  // The padded blended box, stored as [x1, y1, x2, y2].
  std::vector<int> _padded_blended_tlbr;

  // Constructor (if needed)
  CanvasManager(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad = 128)
      : _x1(0),
        _y1(0),
        _x2(0),
        _y2(0),
        canvas_info_(canvas_info),
        _overlapping_width(0),
        _minimize_blend(minimize_blend),
        _overlap_pad(overlap_pad) {}

  // This function updates blending parameters if _minimize_blend is true.
  void updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2) {
    // Ensure that canvas positions are available.
    assert(canvas_info_.positions.size() >= 2);

    // Unpack positions from the canvas.
    _x1 = canvas_info_.positions[0].x;
    _y1 = canvas_info_.positions[0].y;
    _x2 = canvas_info_.positions[1].x;
    _y2 = canvas_info_.positions[1].y;

    int width_1 = _remapper_1.width;
    _overlapping_width = width_1 - _x2;
    // The first remapper's width must be greater than _x2.
    assert(width_1 > _x2);

    if (_minimize_blend) {
      // Set remapper x positions.
      _remapper_1.xpos = _x1;
      _remapper_2.xpos = _x1 + _overlap_pad; // Start overlapping right away.

      // Define the seam box (the region to be blended).
      int box_x1 = _x2 - _overlap_pad;
      int box_y1 = std::max(0, std::min(_y1, _y2) - _overlap_pad);
      int box_x2 = width_1 + _overlap_pad;
      int box_y2 =
          std::min(canvas_info_.height, std::max(_y1 + _remapper_1.height, _y2 + _remapper_2.height) + _overlap_pad);
      _padded_blended_tlbr = {box_x1, box_y1, box_x2, box_y2};

      // Validate the computed coordinates.
      assert(box_x1 >= 0);
      assert(box_x2 <= canvas_info_.width);

      // Compute ROIs
      partial_size_1 = cv::Size(_x2 + _overlap_pad, _remapper_1.height);
      roi_partial_1 = {0, 0, _x2 + _overlap_pad, partial_size_1.height};
      roi_blend_1 = {_x2 - _overlap_pad, 0, remapped_size_1.width, remapped_size_1.height};

      partial_size_2 = cv::Size{_remapper_2.width - (_overlapping_width - _overlap_pad), _remapper_2.height};
      roi_partial_2 = {
          _overlapping_width - _overlap_pad,
          0,
          _overlapping_width - _overlap_pad + partial_size_2.width,
          partial_size_2.height};
      roi_blend_2 = {0, 0, _overlapping_width + _overlap_pad, remapped_size_2.height};
    }
  }

  // Example conversion function that returns a cv::Mat with the same size as the canvas.
  // If _minimize_blend is true, it also updates the blend parameters and returns a cropped region.
  cv::Mat convertMaskMat(const cv::Mat& mask) {
    int padw = 0, padh = 0;
    int mwidth = mask.cols;
    int mheight = mask.rows;

    // The mask should not be larger than the canvas.
    assert(mwidth <= canvas_info_.width);
    assert(mheight <= canvas_info_.height);

    if (mwidth < canvas_info_.width)
      padw = canvas_info_.width - mwidth;
    if (mheight < canvas_info_.height)
      padh = canvas_info_.height - mheight;

    cv::Mat paddedMask;
    if (padw > 0 || padh > 0) {
      // Replicate border pixels on the right and bottom.
      cv::copyMakeBorder(mask, paddedMask, 0, padh, 0, padw, cv::BORDER_REPLICATE);
    } else {
      paddedMask = mask;
    }

    // Check that the padded mask matches the canvas dimensions.
    assert(paddedMask.cols == canvas_info_.width);
    assert(paddedMask.rows == canvas_info_.height);

    if (_minimize_blend) {
      // Update blending parameters.
      // updateMinimizeBlend();
      // In the original Python code, the mask is cropped horizontally:
      //   mask[..., positions[1].x - overlap_pad : remapper_1.width + overlap_pad]
      int x_start = canvas_info_.positions[1].x - _overlap_pad;
      int x_end = _remapper_1.width + _overlap_pad;
      // Validate the crop region.
      assert(x_start >= 0 && x_end <= paddedMask.cols);
      cv::Rect roi(x_start, 0, x_end - x_start, paddedMask.rows);
      return paddedMask(roi);
    }
    return paddedMask;
  }

  cv::Size partial_size_1;
  cv::Size partial_size_2;
  int4 roi_partial_1{
      0,
  };
  int4 roi_partial_2{
      0,
  };
  int4 roi_blend_1{
      0,
  };
  int4 roi_blend_2{
      0,
  };

  constexpr int overlap_padding() const {
    return _overlap_pad;
  }

  constexpr int overlapping_width() const {
    return _overlapping_width;
  }

  constexpr int canvas_width() const {
    return canvas_info_.width;
  }

  constexpr int canvas_height() const {
    return canvas_info_.height;
  }

 private:
  CanvasInfo canvas_info_;
  int _overlapping_width{0};
  bool _minimize_blend{false};
  int _overlap_pad{0};
};

cv::Mat make_fake_mask_like(const cv::Mat& mask) {
  cv::Mat img(mask.rows, mask.cols, CV_32FC1, cv::Scalar(0));

  // Define a region of interest (ROI) for the left half of the image.
  cv::Rect leftHalfROI(0, 0, mask.cols / 2, mask.rows);

  // Set all pixels in the left half to 1.
  img(leftHalfROI).setTo(1.0f);
  return img;
}

template <typename T, typename T_compute>
struct StitchingContext {
  StitchingContext(int batch_size, bool is_hard_seam) : batch_size_(batch_size), is_hard_seam_(is_hard_seam) {}
  // Static buffers
  std::unique_ptr<CudaMat<uint16_t>> remap_1_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_1_y;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_y;

  std::unique_ptr<CudaMat<T_compute>> cudaBlendSoftSeam;
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam;

  // Scratch buffers
  std::unique_ptr<CudaMat<T_compute>> cudaFull1;
  std::unique_ptr<CudaMat<T_compute>> cudaFull2;

  // Laplacian Blend Scratch context
  std::unique_ptr<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>> laplacian_blend_context;

  constexpr int batch_size() const {
    return batch_size_;
  }
  constexpr bool is_hard_seam() const {
    return is_hard_seam_;
  }

 private:
  int batch_size_;
  bool is_hard_seam_;
};

template <typename T, typename T_compute>
class CudaStitchPano {
 public:
  CudaStitchPano(int batch_size) {}

  static CudaStatusOr<std::unique_ptr<CudaMat<T>>> process(
      const CudaMat<T>& sampleImage1,
      const CudaMat<T>& sampleImage2,
      StitchingContext<T, T_compute>& stitch_context,
      CanvasManager& canvas_manager,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T>>&& canvas) {
    CudaStatus cuerr;

    assert(canvas);

    int zero = 0;
    // int y1 = canvas_manager._y1;
    // int y2 = canvas_manager._y2;

    auto roi_width = [](const int4& roi) { return roi.z - roi.x; };
    // auto roi_height = [](const int4& roi) { return roi.w - roi.y; };

    if (!stitch_context.is_hard_seam()) {
      //
      // SOFT SEAM LEFT
      //
#if 1
      //
      // Image 1
      //
      // Remap image 1 onto the canvas
      //
      cuerr = batched_remap_kernel_ex_offset(
          sampleImage1.data(),
          sampleImage1.width(),
          sampleImage1.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          stream);
      // CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_SMALL(canvas);
#endif

#if 1
      //
      // Now copy the blending portion of remapped image 1 from the canvas onto the blend image
      //
      cuerr = simple_make_full_batch<BaseScalar_t<T_compute>, BaseScalar_t<T_compute>, unsigned char>(
          // Image 1 (float image)
          canvas->data_raw(),
          canvas->width(),
          canvas->height(),
          /*region_width=*/roi_width(canvas_manager.roi_blend_1),
          /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_1)*/,
          /*channels=*/3,
          // Batch of masks (optional)
          nullptr,
          0,
          0,
          0,
          canvas_manager.roi_blend_1.x,
          // canvas_manager.roi_blend_1.y,
          0 /* we've already applied our Y offset */,
          canvas_manager._remapper_1.xpos,
          // y1,
          zero,
          stitch_context.cudaBlendSoftSeam->width(),
          stitch_context.cudaBlendSoftSeam->height(),
          /*adjust_origin=*/false,
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.cudaFull1->data_raw(),
          /*d_full_masks=*/nullptr,
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(stitch_context.cudaFull1);
#endif
    } else {
      //
      // HARD SEAM LEFT
      //
#if 1
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          sampleImage1.data(),
          sampleImage1.width(),
          sampleImage1.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          1 /* <-- we inverted the mask at load-time to make it a weight, so image 0 is actually 1 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          stream);
      // SHOW_SMALL(canvas);
#endif
    }
    //
    // Image 2
    //
    if (!stitch_context.is_hard_seam()) {
      //
      // SOFT SEAM RIGHT
      //
#if 1
      //
      // Remap image 2 directly onto the canvas (will overwrite the overlappign portion of image 1)
      //
      cuerr = batched_remap_kernel_ex_offset(
          (const float3*)sampleImage2.data(),
          sampleImage2.width(),
          sampleImage2.height(),
          (float3*)canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_SMALL(canvas);
#endif

#if 1
      //
      // Now copy the blending portion of remapped image 2 from the canvas onto the blend image
      //
      // assert(stitch_context.cudaBlendSoftSeam->height() == roi_height(canvas_manager.roi_blend_2));
      cuerr = simple_make_full_batch<BaseScalar_t<T_compute>, BaseScalar_t<T_compute>, unsigned char>(
          // Image 1 (float image)
          canvas->data_raw(),
          canvas->width(),
          canvas->height(),
          /*region_width=*/roi_width(canvas_manager.roi_blend_2),
          /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_2)*/,
          /*channels=*/3,
          // Batch of masks (optional)
          nullptr,
          0,
          0,
          0,
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          canvas_manager._remapper_2.xpos,
          // y2,
          zero,
          stitch_context.cudaBlendSoftSeam->width(),
          stitch_context.cudaBlendSoftSeam->height(),
          /*adjust_origin=*/false,
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.cudaFull2->data_raw(),
          /*d_full_masks=*/nullptr,
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(stitch_context.cudaFull2);
#endif
    } else {
      //
      // HARD SEAM RIGHT
      //
#if 1
      assert(canvas_manager._x2 + stitch_context.remap_2_x->width() <= canvas->width());
      assert(canvas_manager._y2 + stitch_context.remap_2_x->height() <= canvas->height());
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          sampleImage2.data(),
          sampleImage2.width(),
          sampleImage2.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          0 /* <-- we inverted the mask at load-time to make it a weight, so image 1 is actually 0 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          stream);
      // SHOW_SMALL(&sampleImage2);
      // SHOW_SMALL(canvas);
      // SHOW_SMALL(stitch_context.cudaBlendHardSeam);
#endif
    }
    if (!stitch_context.is_hard_seam()) {
      CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull1;
#if 1
      //
      // BLEND THE IMAGES (overlapping portions + some padding)
      //
      cuerr = cudaBatchedLaplacianBlendWithContext(
          stitch_context.cudaFull1->data_raw(),
          stitch_context.cudaFull2->data_raw(),
          stitch_context.cudaBlendSoftSeam->data_raw(),
          // Put output in full-1 memory
          cudaBlendedFull.data_raw(),
          *stitch_context.laplacian_blend_context,
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(&cudaBlendedFull);
#endif

#if 1
      //
      // Copy the blended portion (overlapping portion + some padding) onto
      // the canvas over some of the remapped image 1 and image 2
      //
      cuerr = copyRoiBatchedInterface(
          cudaBlendedFull.data(),
          cudaBlendedFull.width(),
          cudaBlendedFull.height(),
          cudaBlendedFull.width(),
          cudaBlendedFull.height(),
          0,
          0,
          canvas->data(),
          canvas->width(),
          canvas->height(),
          /*offsetX=*/canvas_manager._x2 - canvas_manager.overlap_padding(),
          /*offsetY=*/0,
          /*channels=*/1, // <-- 1 when using stuff like float3
          /*batchSize=*/stitch_context.batch_size(),
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(canvas);
#endif
    }
    return std::move(canvas);
  };
};

std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  return std::vector<cv::Mat>(batch_size, mat);
}

struct ControlMasks {
  bool load(std::string game_dir) {
    if (!game_dir.empty() && game_dir.back() != '/') {
      game_dir += '/';
    }
    std::string mapping_0_pos = game_dir + "mapping_0000.tif";
    std::string mapping_0_x = game_dir + "mapping_0000_x.tif";
    std::string mapping_0_y = game_dir + "mapping_0000_y.tif";
    std::string mapping_1_pos = game_dir + "mapping_0001.tif";
    std::string mapping_1_x = game_dir + "mapping_0001_x.tif";
    std::string mapping_1_y = game_dir + "mapping_0001_y.tif";
    std::string whole_seam_mask = game_dir + "seam_file.png";

    img1_col = cv::imread(mapping_0_x, cv::IMREAD_ANYDEPTH);
    assert(img1_col.type() == CV_16U);
    if (img1_col.empty()) {
      return false;
    }
    img1_row = cv::imread(mapping_0_y, cv::IMREAD_ANYDEPTH);
    if (img1_row.empty()) {
      return false;
    }
    img2_col = cv::imread(mapping_1_x, cv::IMREAD_ANYDEPTH);
    if (img2_col.empty()) {
      return false;
    }
    img2_row = cv::imread(mapping_1_y, cv::IMREAD_ANYDEPTH);
    if (img2_row.empty()) {
      return false;
    }

    whole_seam_mask_image = load_seam_mask(whole_seam_mask);
    if (whole_seam_mask_image.empty()) {
      return false;
    }
    // whole_seam_mask_image.convertTo(whole_seam_mask_image, CV_32FC1);
#if 0
      whole_seam_mask_image = make_fake_mask_like(whole_seam_mask_image);
#endif
    positions = normalize_positions(std::vector<SpatialTiff>{get_geo_tiff(mapping_0_pos), get_geo_tiff(mapping_1_pos)});
    return true;
  }

 private:
  struct SpatialTiff {
    // position in pixels
    float xpos;
    float ypos;
  };

  static std::vector<SpatialTiff> normalize_positions(std::vector<SpatialTiff>&& positions) {
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

  static SpatialTiff get_geo_tiff(const std::string& filename) {
    TiffInfo info = getTiffInfo(filename);
    return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
  }

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
  static TiffInfo getTiffInfo(const std::string& filename) {
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

 public:
  cv::Mat img1_col;
  cv::Mat img1_row;
  cv::Mat img2_col;
  cv::Mat img2_row;
  cv::Mat whole_seam_mask_image;
  std::vector<SpatialTiff> positions;
};

int main(int argc, char** argv) {
  // Usage check.
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <game-id>" << std::endl;
    return -1;
  }

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  RenderSet display;

  std::string game_id = argv[1];
  std::string game_dir = std::string(::getenv("HOME")) + "/Videos/" + game_id + "/";

  // std::string sample_img_left_path = game_dir + "GX010100.png";
  // std::string sample_img_right_path = game_dir + "GX010019.png";

  // std::string sample_img_left_path = game_dir + "GX010097.png";
  // std::string sample_img_right_path = game_dir + "GX010016.png";

  // PDP
  std::string sample_img_left_path = game_dir + "GX010087.png";
  std::string sample_img_right_path = game_dir + "GX010003.png";

  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  sample_img_left.convertTo(sample_img_left, CV_32FC3, 1.0 / 255.0);
  sample_img_right.convertTo(sample_img_right, CV_32FC3, 1.0 / 255.0);

  ControlMasks control_masks;
  control_masks.load(game_dir);

  // Compute canvas size
  const int canvas_width = std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols);
  const int canvas_height = std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows);
  std::cout << "Canvas size: " << canvas_width << " x " << canvas_height << std::endl;

// Configurable parameter: number of pyramid levels.
#ifdef __aarch64__
  // Lower compute, quick and dirty
  int numLevels = 0;
  // int numLevels = 6;
#else
  // int numLevels = 6;
  // int numLevels = 2;
  // int numLevels = 6;
  int numLevels = 0;
#endif

#if 1
#if 0
  using T = uchar3;
  using T_compute = uchar3;
#define CV_T_PIPELINE CV_8UC3
#define CV_T_COMPUTE3 CV_8UC3
#else
  using T = float3;
  using T_compute = float3;
#define CV_T_PIPELINE CV_32FC3
#define CV_T_COMPUTE3 CV_32FC3
#endif
#else
  using T = float;
  using T_compute = __half;
#define CV_T_PIPELINE CV_32FC3
#define CV_T_COMPUTE3 CV_16FC3
#endif

  // constexpr int kBatchSize = 1;
  constexpr int kBatchSize = 2;

  StitchingContext<T, T_compute> stitch_context(/*batch_size=*/kBatchSize, /*is_hard_seam=*/numLevels == 0);

  //
  // CanvasManager
  //
  CanvasManager canvas_manager(
      CanvasInfo{
          .width = canvas_width,
          .height = canvas_height,
          .positions =
              {cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
               cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos)}},
      /*minimize_blend=*/!stitch_context.is_hard_seam());
  canvas_manager._remapper_1.width = control_masks.img1_col.cols;
  canvas_manager._remapper_1.height = control_masks.img1_col.rows;
  canvas_manager._remapper_2.width = control_masks.img2_col.cols;
  canvas_manager._remapper_2.height = control_masks.img2_col.rows;

  canvas_manager.updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  cv::Mat blend_seam = canvas_manager.convertMaskMat(control_masks.whole_seam_mask_image);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  auto canvas = std::make_unique<CudaMat<T>>(
      stitch_context.batch_size(), canvas_manager.canvas_width(), canvas_manager.canvas_height());

  assert(control_masks.img1_col.type() == CV_16U);
  stitch_context.remap_1_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_col);
  stitch_context.remap_1_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_row);

  stitch_context.remap_2_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_col);
  stitch_context.remap_2_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_row);

  if (!stitch_context.is_hard_seam()) {
    blend_seam.convertTo(blend_seam, CV_T_COMPUTE3);
    stitch_context.cudaFull1 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context.cudaFull2 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);

    stitch_context.cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(blend_seam);
    stitch_context.laplacian_blend_context = std::make_unique<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>>(
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        numLevels,
        /*batch_size=*/stitch_context.batch_size());
  } else {
    assert(blend_seam.type() == CV_8U);
    stitch_context.cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(blend_seam);
  }

  //
  // The actual incoming images
  //

  // matchSeamImages(
  //     sample_img_left,
  //     sample_img_left,
  //     control_masks.whole_seam_mask_image,
  //     /*N=*/10,
  //     cv::Point(canvas_manager.canvas_info_.positions[0].x, canvas_manager.canvas_info_.positions[0].y),
  //     cv::Point(canvas_manager.canvas_info_.positions[1].x, canvas_manager.canvas_info_.positions[1].y));

  CudaMat<T> sampleImage1(as_batch(sample_img_left, kBatchSize));
  CudaMat<T> sampleImage2(as_batch(sample_img_right, kBatchSize));

  auto blendedCanvasResult = CudaStitchPano<T, T_compute>::process(
      sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  auto blendedCanvas = blendedCanvasResult.ConsumeValueOrDie();
  SHOW_SMALL(blendedCanvas);

  // blendedCanvas = process(sampleImage1, sampleImage2, stitch_context, canvas_manager, stream);
  // SHOW_IMAGE(blendedCanvas);

  // cudaStreamSynchronize(stream);

  // display.render("cudaBlendedFull", CudaSurface(cudaBlendedFull), stream);

#if 0 /* perf test */
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    blendedCanvas = CudaStitchPano<T, T_compute>::process(
                        sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(blendedCanvas))
                        .ConsumeValueOrDie();
    cudaStreamSynchronize(stream);
  }

  auto stop_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  float ms = stop_ms - start_ms;
  float sec_per_frame = (ms / 1000) / (frame_count * stitch_context.batch_size());
  std::cout << "Blend speed: " << (1.0 / sec_per_frame) << "fps" << std::endl;
#endif

  cudaStreamDestroy(stream);

  return cudaSuccess;
}
