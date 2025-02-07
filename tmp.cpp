#include "canvasManager.h"
#include "controlMasks.h"
// Additional CUDA/OpenGL/utility headers:
#include "cudaBlend.h"
#include "cudaMakeFull.h"
#include "cudaMat.h"
#include "cudaPano.h"
#include "cudaRemap.h"
#include "cudaStatus.h"
#include "glDisplay.h"
#include "imageFormat.h"
#include "videoOutput.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <unordered_set>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

/**
 * @brief Checks whether a keyboard press is available in a non-blocking manner.
 *
 * This function temporarily switches stdin to noncanonical mode (no buffering)
 * and checks if a character is waiting. If yes, it returns 1; otherwise 0.
 *
 * @return 1 if a character is available, 0 otherwise.
 */
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

/**
 * @brief Waits until a keyboard press is detected.
 *
 * Internally uses `kbhit()` in a loop with a small sleep to avoid busy-waiting.
 *
 * @return The keyboard code read from stdin (if needed).
 */
int wait_key() {
  int c;
  while (!(c = kbhit())) {
    usleep(100);
  }
  return c;
}

/**
 * @brief Adjusts two images so that their color along a seam boundary aligns.
 *
 * It samples up to N pixels to the left side of the seam in `image1` and
 * up to N pixels to the right side of the seam in `image2`, computes average
 * color, and applies an offset to minimize color difference across the seam.
 *
 * @param image1    The first (left) image to be adjusted in-place.
 * @param image2    The second (right) image to be adjusted in-place.
 * @param seam      A binary mask (CV_8U) indicating seam boundary (1 at the boundary).
 * @param N         Number of pixels on either side of the seam to sample.
 * @param topLeft1  The top-left corner of `image1` in the seam mask.
 * @param topLeft2  The top-left corner of `image2` in the seam mask.
 */
void matchSeamImages(
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam,
    int N,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2) {
  // Ensure seam is 8-bit single channel.
  if (seam.type() != CV_8U) {
    std::cerr << "Error: Seam mask must be of type CV_8U." << std::endl;
    return;
  }

  // Accumulators for averaging color.
  cv::Scalar sumLeft(0, 0, 0);
  cv::Scalar sumRight(0, 0, 0);
  int countLeft = 0, countRight = 0;

  // ----- Sample the left side in image1 -----
  int startRow1 = image1.rows / 4;
  int endRow1 = (3 * image1.rows) / 4;
  for (int r = startRow1; r < endRow1; r++) {
    int globalRow = topLeft1.y + r;
    if (globalRow < 0 || globalRow >= seam.rows) {
      continue;
    }
    int colStart = topLeft1.x;
    int colEnd = topLeft1.x + image1.cols;

    // Find boundary where seam == 1 in the global seam mask space.
    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols) {
        continue;
      }
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1) {
      continue;
    }

    // Convert global boundary to local coords in image1.
    int seamLocalCol = seamGlobalCol - topLeft1.x;

    // Sample up to N pixels left of the boundary.
    int sampleStart = std::max(0, seamLocalCol - N);
    for (int c = sampleStart; c < seamLocalCol; c++) {
      if (c < 0 || c >= image1.cols) {
        continue;
      }
      // Read the pixel according to depth.
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

  // ----- Sample the right side in image2 -----
  int startRow2 = image2.rows / 4;
  int endRow2 = (3 * image2.rows) / 4;
  for (int r = startRow2; r < endRow2; r++) {
    int globalRow = topLeft2.y + r;
    if (globalRow < 0 || globalRow >= seam.rows) {
      continue;
    }
    int colStart = topLeft2.x;
    int colEnd = topLeft2.x + image2.cols;

    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols) {
        continue;
      }
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1) {
      continue;
    }

    int seamLocalCol = seamGlobalCol - topLeft2.x;

    // Sample up to N pixels right of the boundary.
    int sampleEnd = std::min(image2.cols, seamLocalCol + N);
    for (int c = seamLocalCol; c < sampleEnd; c++) {
      if (c < 0 || c >= image2.cols) {
        continue;
      }
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

  // If no samples were gathered, warn and exit.
  if (countLeft == 0 || countRight == 0) {
    std::cerr << "Error: Not enough seam samples collected for adjustment." << std::endl;
    return;
  }

  // Compute average color on each side.
  cv::Scalar avgLeft = sumLeft * (1.0 / countLeft);
  cv::Scalar avgRight = sumRight * (1.0 / countRight);
  std::cout << "Average values (Image1, left side): " << avgLeft << std::endl;
  std::cout << "Average values (Image2, right side): " << avgRight << std::endl;

  // Compute a per-channel offset (half the difference).
  cv::Scalar offset = (avgLeft - avgRight) * 0.5;
  std::cout << "Offset: " << offset << std::endl;

  // Lambda to shift an image by a color offset.
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
      // Clamp to [0, 255].
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

  // Subtract offset from image1, add offset to image2 to meet in the middle.
  adjustImage(image1, -offset);
  adjustImage(image2, offset);
}

/**
 * @brief Displays an image in a named OpenCV window, optionally waiting for a keypress.
 *
 * @param label The name of the window.
 * @param img   The image to display.
 * @param wait  Whether to wait indefinitely for a keypress (default true).
 */
void show_image(const std::string& label, const cv::Mat& img, bool wait = true) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

/**
 * @brief Displays a scaled version of an image in a named OpenCV window.
 *
 * @param label The name of the window.
 * @param image The image to display.
 * @param scale The scale factor (1.0 leaves the image as-is).
 * @param wait  Whether to wait indefinitely for a keypress (default true).
 */
void displayScaledImage(const std::string& label, cv::Mat image, float scale = 1.0, bool wait = true) {
  if (scale != 1.0f) {
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    cv::resize(image, image, cv::Size(newWidth, newHeight));
  }
  cv::imshow(label, image);
  cv::waitKey(wait ? 0 : 1);
}

/**
 * @brief Macro for showing a CudaMat by downloading it and displaying in a window (blocking).
 */
#define SHOW_IMAGE(_mat$)                                                \
  do {                                                                   \
    show_image(std::string(#_mat$), (_mat$)->download(), /*wait=*/true); \
  } while (false)

/**
 * @brief Macro for showing a scaled CudaMat by downloading and resizing before display (blocking).
 */
#define SHOW_SCALED(_mat$, _scale$)                                                       \
  do {                                                                                    \
    displayScaledImage(std::string(#_mat$), (_mat$)->download(), _scale$, /*wait=*/true); \
  } while (false)

/**
 * @brief Macro for showing a small (5%) scaled CudaMat.
 */
#define SHOW_SMALL(_mat$)     \
  do {                        \
    SHOW_SCALED(_mat$, 0.05); \
  } while (false)

namespace {

/**
 * @brief Simple structure to encapsulate a GPU surface for rendering via glDisplay.
 *
 * @tparam T The pixel type in CUDA memory (e.g., uchar3, float3, etc.).
 */
template <typename T>
struct CudaSurface {
  CudaSurface(int w, int h, imageFormat format, void* data)
      : width(w), height(h), image_format(format), dataptr(data) {}

  CudaSurface(const CudaMat<T>& cm)
      : width(cm.width()), height(cm.height()), image_format(get_image_format(cm.type())), dataptr((void*)cm.data()) {}

  int width{0}; ///< Surface width.
  int height{0}; ///< Surface height.
  imageFormat image_format; ///< Format enum (e.g., IMAGE_RGB8, IMAGE_FLOAT32).
  void* dataptr{nullptr}; ///< Pointer to GPU memory.
};

/**
 * @class RenderSet
 * @brief Manages multiple `glDisplay` output windows, each identified by a name.
 *
 * Calls to `render()` will create a new window if one by that name does not already exist.
 * This allows multiple GPU surfaces to be rendered simultaneously to separate windows.
 */
class RenderSet {
 public:
  /**
   * @brief Renders a given CUDA surface in a named OpenGL display window.
   *
   * If the named window does not exist yet, it is automatically created.
   *
   * @tparam T The pixel type for the surface (e.g., `uchar3`, `float3`).
   * @param name The identifier for the display window.
   * @param surface The `CudaSurface` to render.
   * @param stream  Optional CUDA stream for asynchronous rendering.
   */
  template <typename T>
  void render(const std::string& name, const CudaSurface<T>& surface, cudaStream_t stream = 0) {
    get_video_output(name, surface.width, surface.height)
        ->Render((void*)surface.dataptr, surface.width, surface.height, surface.image_format, stream);
  }

 private:
  /**
   * @brief Creates a new video output using glDisplay with the given dimensions.
   *
   * @param name   The display window name.
   * @param width  The initial width of the window.
   * @param height The initial height of the window.
   * @return A `std::unique_ptr<glDisplay>` pointing to the new display.
   */
  static std::unique_ptr<glDisplay> create_video_output(const std::string& name, int width, int height) {
    videoOptions vo;
    vo.width = width;
    vo.height = height;
    auto video_output = std::unique_ptr<glDisplay>(glDisplay::Create(vo));
    video_output->SetTitle(name.c_str());
    return video_output;
  }

  /**
   * @brief Returns the display by name, creating it if necessary.
   *
   * @param name   The display window name.
   * @param width  The desired width of the window.
   * @param height The desired height of the window.
   */
  videoOutput* get_video_output(const std::string& name, int width, int height) {
    std::unique_lock<std::mutex> lk(mu_);
    auto found = video_outputs_.find(name);
    if (found == video_outputs_.end()) {
      found = video_outputs_.emplace(name, create_video_output(name, width, height)).first;
    }
    return found->second.get();
  }

  std::mutex mu_; ///< Protects the map of `glDisplay` outputs.
  std::map<std::string, std::unique_ptr<glDisplay>> video_outputs_; ///< Map: name -> display instance.
};

} // namespace

/**
 * @brief Finds the minimum and maximum values in an OpenCV matrix.
 *
 * @param mat Input matrix (single or multi-channel).
 * @return A pair (minVal, maxVal).
 */
std::pair<double, double> get_min_max(const cv::Mat& mat) {
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
  return std::make_pair(minVal, maxVal);
}

/**
 * @brief Returns all unique values of type T in an OpenCV matrix, excluding any in `ignore`.
 *
 * @tparam T The data type we expect in the matrix.
 * @param mat The input matrix.
 * @param ignore A set of values to skip.
 * @return A sorted `std::set` of unique values found.
 * @throws std::invalid_argument If the matrix type does not match `cv::DataType<T>::type`.
 */
template <typename T>
std::set<T> get_unique_values(const cv::Mat& mat, const std::unordered_set<T>& ignore = {}) {
  std::set<T> unique_values;
  if (mat.type() != cv::DataType<T>::type) {
    throw std::invalid_argument("Matrix data type does not match the template type T");
  }
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      T value = mat.at<T>(i, j);
      if (ignore.find(value) == ignore.end()) {
        unique_values.insert(value);
      }
    }
  }
  return unique_values;
}

/**
 * @brief Loads a position mask (e.g., a row/col mapping) from disk with optional min/max reporting.
 *
 * @param filename The file to read.
 * @param minVal   Optional pointer to store the minimum pixel value found.
 * @param maxVal   Optional pointer to store the maximum pixel value found.
 * @return The loaded `cv::Mat`. If empty, minVal/maxVal will be set to NaN.
 */
cv::Mat load_position_mask(const std::string& filename, double* minVal, double* maxVal) {
  cv::Mat pos_mask = cv::imread(filename, cv::IMREAD_ANYDEPTH);
  if (!pos_mask.empty()) {
    if (minVal || maxVal) {
      double min, max;
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(pos_mask, &min, &max, &minLoc, &maxLoc);
      if (minVal)
        *minVal = min;
      if (maxVal)
        *maxVal = max;
    }
  } else {
    if (minVal)
      *minVal = std::nan("");
    if (maxVal)
      *maxVal = std::nan("");
  }
  return pos_mask;
}

/**
 * @brief Generates a synthetic 1/0 mask with a vertical split.
 *
 * Given a `mask` to define dimensions, returns a float matrix with the left half = 1,
 * right half = 0. Useful for testing blending logic.
 *
 * @param mask A reference mask from which to copy dimensions.
 * @return A new CV_32FC1 mask with a half-and-half split.
 */
cv::Mat make_fake_mask_like(const cv::Mat& mask) {
  cv::Mat img(mask.rows, mask.cols, CV_32FC1, cv::Scalar(0));
  cv::Rect leftHalfROI(0, 0, mask.cols / 2, mask.rows);
  img(leftHalfROI).setTo(1.0f);
  return img;
}

namespace hm {
namespace cuda {

/**
 * @class CudaStitchPano
 * @brief Example stitching pipeline that demonstrates how two images might be remapped and blended into a canvas.
 *
 * This templated class is designed to work with various data types (e.g., `uchar3`, `float3`, or half-precision).
 *
 * @tparam T         The source pixel type for the input images (e.g., `uchar3`).
 * @tparam T_compute The computation pixel type (e.g., `float3`) used for blending.
 */
template <typename T, typename T_compute>
class CudaStitchPano {
 public:
  /**
   * @brief Constructs a CudaStitchPano object for a given batch size.
   *
   * @param batch_size How many images are processed concurrently in a single GPU call.
   */
  CudaStitchPano(int batch_size) {}

  /**
   * @brief Main entry point for the stitching process: remap images, copy overlapping regions, and blend if needed.
   *
   * @param sampleImage1       The first input image on the GPU.
   * @param sampleImage2       The second input image on the GPU.
   * @param stitch_context     Holds GPU memory needed for the remap/ blending (e.g., x/y maps, seam mask).
   * @param canvas_manager     Coordinates for final placement (canvas offsets, ROI, etc.).
   * @param stream             The CUDA stream on which to enqueue operations.
   * @param canvas             A pointer to the destination canvas on which images will be composited.
   * @return A new pointer to the updated canvas if success, or an error code otherwise.
   */
  static CudaStatusOr<std::unique_ptr<CudaMat<T>>> process(
      const CudaMat<T>& sampleImage1,
      const CudaMat<T>& sampleImage2,
      StitchingContext<T, T_compute>& stitch_context,
      const hm::pano::CanvasManager& canvas_manager,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T>>&& canvas);
};

template <typename T, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T>>> CudaStitchPano<T, T_compute>::process(
    const CudaMat<T>& sampleImage1,
    const CudaMat<T>& sampleImage2,
    StitchingContext<T, T_compute>& stitch_context,
    const hm::pano::CanvasManager& canvas_manager,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T>>&& canvas) {
  CudaStatus cuerr;

  // Utility lambdas for ROI dimensions.
  auto roi_width = [](const cv::Rect2i& roi) { return roi.width; };

  // If using a soft seam, we do a multi-step blending; if hard seam, just map them with a binary mask.
  if (!stitch_context.is_hard_seam()) {
    // --- SOFT SEAM, LEFT IMAGE ---

    // 1) Remap image1 onto the canvas.
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

    // 2) Copy the overlapping region from the canvas into a dedicated blend buffer (cudaFull1).
    cuerr = simple_make_full_batch<BaseScalar_t<T>, BaseScalar_t<T_compute>, unsigned char>(
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        /*region_width=*/roi_width(canvas_manager.roi_blend_1),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height(),
        /*channels=*/3,
        nullptr,
        0,
        0,
        0,
        canvas_manager.roi_blend_1.x,
        0,
        /*destOffsetX=*/canvas_manager._remapper_1.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull1->data_raw(),
        nullptr,
        stream);
    // Error checks omitted for brevity.
  } else {
    // --- HARD SEAM, LEFT IMAGE ---
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
        /*this_image_index=*/1,
        stitch_context.cudaBlendHardSeam->data(),
        stitch_context.batch_size(),
        stitch_context.remap_1_x->width(),
        stitch_context.remap_1_x->height(),
        canvas_manager._x1,
        canvas_manager._y1,
        stream);
  }

  // --- RIGHT IMAGE ---
  if (!stitch_context.is_hard_seam()) {
    // SOFT SEAM, RIGHT IMAGE
    cuerr = batched_remap_kernel_ex_offset(
        sampleImage2.data(),
        sampleImage2.width(),
        sampleImage2.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_2_x->data(),
        stitch_context.remap_2_y->data(),
        {0, 0, 0},
        stitch_context.batch_size(),
        stitch_context.remap_2_x->width(),
        stitch_context.remap_2_x->height(),
        canvas_manager._x2,
        canvas_manager._y2,
        stream);

    // Copy region from canvas to blend buffer (cudaFull2).
    cuerr = simple_make_full_batch<BaseScalar_t<T>, BaseScalar_t<T_compute>, unsigned char>(
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        roi_width(canvas_manager.roi_blend_2),
        stitch_context.cudaBlendSoftSeam->height(),
        3,
        nullptr,
        0,
        0,
        0,
        canvas_manager._x2,
        canvas_manager._y2,
        /*destOffsetX=*/canvas_manager._remapper_2.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        false,
        stitch_context.batch_size(),
        stitch_context.cudaFull2->data_raw(),
        nullptr,
        stream);

    // Perform the Laplacian blend on the overlapping region:
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull1;
    cuerr = cudaBatchedLaplacianBlendWithContext(
        stitch_context.cudaFull1->data_raw(),
        stitch_context.cudaFull2->data_raw(),
        stitch_context.cudaBlendSoftSeam->data_raw(),
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stream);

    // Copy the result back onto the canvas:
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
        canvas_manager._x2 - canvas_manager.overlap_padding(),
        0,
        /*channels=*/1,
        stitch_context.batch_size(),
        stream);
  } else {
    // HARD SEAM, RIGHT IMAGE
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
        /*this_image_index=*/0,
        stitch_context.cudaBlendHardSeam->data(),
        stitch_context.batch_size(),
        stitch_context.remap_2_x->width(),
        stitch_context.remap_2_x->height(),
        canvas_manager._x2,
        canvas_manager._y2,
        stream);
  }

  // Return the updated canvas.
  return std::move(canvas);
}

} // namespace cuda
} // namespace hm

/**
 * @brief Helper to produce a repeated batch of the same Mat, if your pipeline is batch-based.
 *
 * @param mat The single `cv::Mat`.
 * @param batch_size Number of copies to produce.
 * @return Vector of identical `cv::Mat` objects.
 */
std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  return std::vector<cv::Mat>(batch_size, mat);
}

/**
 * @brief Main entry point demonstrating how to load control masks, create a CanvasManager,
 * and run a stitching pipeline with Laplacian blending or a hard seam.
 *
 * Usage:
 *   ./myProgram <game-id>
 *
 * The code assumes a directory structure and file naming convention for input images.
 */
int main(int argc, char** argv) {
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

  // Example input images for left and right.
  std::string sample_img_left_path = game_dir + "GX010100.png";
  std::string sample_img_right_path = game_dir + "GX010019.png";

  // Load test images in OpenCV.
  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  // Load control masks containing row/col maps and a seam mask.
  hm::pano::ControlMasks control_masks;
  control_masks.load(game_dir);

  // Compute canvas size from image positions.
  const int canvas_width = std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols);
  const int canvas_height = std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows);
  std::cout << "Canvas size: " << canvas_width << " x " << canvas_height << std::endl;

  // Decide how many pyramid levels for Laplacian blending (if not zero, we do soft seam).
  int numLevels = 6;

  // Choose types for input and computation.
  using T = uchar3; // e.g., for GPU-based BGR or RGB in 8u
  using T_compute = float3; // use float3 for blending

  // Possibly scale input images to floats if you want 0..1 range:
  // sample_img_left.convertTo(sample_img_left, CV_32FC3, 1.0/255.0);
  // sample_img_right.convertTo(sample_img_right, CV_32FC3, 1.0/255.0);

  constexpr int kBatchSize = 1;

  // Create a stitching context: stores GPU buffers for remap, seam, etc.
  hm::cuda::StitchingContext<T, T_compute> stitch_context(kBatchSize, /*is_hard_seam=*/(numLevels == 0));

  // Build a CanvasManager to define how images fit on the canvas.
  hm::pano::CanvasManager canvas_manager(
      hm::pano::CanvasInfo{
          .width = canvas_width,
          .height = canvas_height,
          .positions =
              {cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
               cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos)}},
      /*minimize_blend=*/!stitch_context.is_hard_seam(),
      /*overlap_pad=*/128);

  canvas_manager._remapper_1.width = control_masks.img1_col.cols;
  canvas_manager._remapper_1.height = control_masks.img1_col.rows;
  canvas_manager._remapper_2.width = control_masks.img2_col.cols;
  canvas_manager._remapper_2.height = control_masks.img2_col.rows;

  // Update internal blend logic in CanvasManager.
  canvas_manager.updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  // Crop/expand the seam mask to match the canvas region if needed.
  cv::Mat blend_seam = canvas_manager.convertMaskMat(control_masks.whole_seam_mask_image);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  // Create a canvas on the GPU, sized to the entire output.
  auto canvas = std::make_unique<CudaMat<T>>(
      stitch_context.batch_size(), canvas_manager.canvas_width(), canvas_manager.canvas_height());

  // Copy control data (x/y transformations, seam mask) onto GPU.
  stitch_context.remap_1_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_col);
  stitch_context.remap_1_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_row);
  stitch_context.remap_2_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_col);
  stitch_context.remap_2_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_row);

  if (!stitch_context.is_hard_seam()) {
    // Convert seam mask to float3 (for multi-channel blending).
    blend_seam.convertTo(blend_seam, CV_32FC3); // or CV_T_COMPUTE3
    stitch_context.cudaFull1 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context.cudaFull2 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context.cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(blend_seam);

    // Create a Laplacian blending context with `numLevels`.
    stitch_context.laplacian_blend_context = std::make_unique<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>>(
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        numLevels,
        stitch_context.batch_size());
  } else {
    // Hard seam: keep it as a single-channel 8-bit mask.
    assert(blend_seam.type() == CV_8U);
    stitch_context.cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(blend_seam);
  }

  // Convert the input images into batch form, then upload them to GPU.
  CudaMat<T> sampleImage1(as_batch(sample_img_left, kBatchSize));
  CudaMat<T> sampleImage2(as_batch(sample_img_right, kBatchSize));

  // Run the stitching/blending pipeline once:
  auto blendedCanvasResult = hm::cuda::CudaStitchPano<T, T_compute>::process(
      sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  auto blendedCanvas = blendedCanvasResult.ConsumeValueOrDie();

  // Optional: show the result in a scaled window (blocking).
  // SHOW_SCALED(blendedCanvas, 0.25);

  // Basic performance test: run the blend multiple times and measure throughput.
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    blendedCanvas = hm::cuda::CudaStitchPano<T, T_compute>::process(
                        sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(blendedCanvas))
                        .ConsumeValueOrDie();
    cudaStreamSynchronize(stream);
  }

  auto stop_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  float ms = stop_ms - start_ms;
  float sec_per_frame = (ms / 1000.0f) / (frame_count * stitch_context.batch_size());
  std::cout << "Blend speed: " << (1.0f / sec_per_frame) << " fps" << std::endl;

  cudaStreamDestroy(stream);
  return cudaSuccess;
}
