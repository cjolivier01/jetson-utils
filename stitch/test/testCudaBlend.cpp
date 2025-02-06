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

void show_image(const std::string& label, const cv::Mat& img, bool wait = true) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

#define SHOW_IMAGE(_mat$)                                                 \
  do {                                                                    \
    show_image(std::string(#_mat$), (_mat$)->download(), /*waitr=*/true); \
  } while (false)

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

class MaskConverter {
 public:
  // Canvas and blending parameters.
  CanvasInfo _canvas_info;
  bool _minimize_blend{true};
  int _overlap_pad{0};

  // Two remappers (for example, for two image streams).
  Remapper _remapper_1;
  Remapper _remapper_2;

  // Additional members for blending logic.
  int _x1, _y1, _x2, _y2;
  int _overlapping_width;
  // The padded blended box, stored as [x1, y1, x2, y2].
  std::vector<int> _padded_blended_tlbr;

  // Constructor (if needed)
  MaskConverter() : _minimize_blend(false), _overlap_pad(128), _x1(0), _y1(0), _x2(0), _y2(0), _overlapping_width(0) {}

  // This function updates blending parameters if _minimize_blend is true.
  void updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2) {
    if (_minimize_blend) {
      // Ensure that canvas positions are available.
      assert(_canvas_info.positions.size() >= 2);

      // Unpack positions from the canvas.
      _x1 = _canvas_info.positions[0].x;
      _y1 = _canvas_info.positions[0].y;
      _x2 = _canvas_info.positions[1].x;
      _y2 = _canvas_info.positions[1].y;

      // Set remapper x positions.
      _remapper_1.xpos = _x1;
      _remapper_2.xpos = _x1 + _overlap_pad; // Start overlapping right away.

      int width_1 = _remapper_1.width;
      _overlapping_width = width_1 - _x2;
      // The first remapper's width must be greater than _x2.
      assert(width_1 > _x2);

      // Define the seam box (the region to be blended).
      int box_x1 = _x2 - _overlap_pad;
      int box_y1 = std::max(0, std::min(_y1, _y2) - _overlap_pad);
      int box_x2 = width_1 + _overlap_pad;
      int box_y2 =
          std::min(_canvas_info.height, std::max(_y1 + _remapper_1.height, _y2 + _remapper_2.height) + _overlap_pad);
      _padded_blended_tlbr = {box_x1, box_y1, box_x2, box_y2};

      // Validate the computed coordinates.
      assert(box_x1 >= 0);
      assert(box_x2 <= _canvas_info.width);

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
    assert(mwidth <= _canvas_info.width);
    assert(mheight <= _canvas_info.height);

    if (mwidth < _canvas_info.width)
      padw = _canvas_info.width - mwidth;
    if (mheight < _canvas_info.height)
      padh = _canvas_info.height - mheight;

    cv::Mat paddedMask;
    if (padw > 0 || padh > 0) {
      // Replicate border pixels on the right and bottom.
      cv::copyMakeBorder(mask, paddedMask, 0, padh, 0, padw, cv::BORDER_REPLICATE);
    } else {
      paddedMask = mask;
    }

    // Check that the padded mask matches the canvas dimensions.
    assert(paddedMask.cols == _canvas_info.width);
    assert(paddedMask.rows == _canvas_info.height);

    if (_minimize_blend) {
      // Update blending parameters.
      // updateMinimizeBlend();
      // In the original Python code, the mask is cropped horizontally:
      //   mask[..., positions[1].x - overlap_pad : remapper_1.width + overlap_pad]
      int x_start = _canvas_info.positions[1].x - _overlap_pad;
      int x_end = _remapper_1.width + _overlap_pad;
      // Validate the crop region.
      assert(x_start >= 0 && x_end <= paddedMask.cols);
      cv::Rect roi(x_start, 0, x_end - x_start, paddedMask.rows);
      return paddedMask(roi);
    }
    return paddedMask.clone();
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

  // std::vector<cv::Mat> canvas_mat;
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
      MaskConverter& mask_converter,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T>>&& canvas) {
    CudaStatus cuerr;

    assert(canvas);

    int zero = 0;
    // int y1 = mask_converter._y1;
    // int y2 = mask_converter._y2;

    auto roi_width = [](const int4& roi) { return roi.z - roi.x; };
    // auto roi_height = [](const int4& roi) { return roi.w - roi.y; };

    if (!stitch_context.is_hard_seam()) {
#if 1
      //
      // Image 1
      //
      // Remap image 1 ontp the canvas
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
          make_float3(0, 0, 0),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/mask_converter._x1,
          /*offsetY=*/mask_converter._y1,
          stream);
      // CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(canvas);
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
          /*region_width=*/roi_width(mask_converter.roi_blend_1),
          /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(mask_converter.roi_blend_1)*/,
          /*channels=*/3,
          // Batch of masks (optional)
          nullptr,
          0,
          0,
          0,
          mask_converter.roi_blend_1.x,
          // mask_converter.roi_blend_1.y,
          0 /* we've already applied our Y offset */,
          mask_converter._remapper_1.xpos,
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
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          sampleImage1.data(),
          sampleImage1.width(),
          sampleImage1.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          make_float3(0, 0, 0),
          /*this_image_index=*/
          1 /* <-- we inverted the mask at load-time to make it a weight, so image 0 is actually 1 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/mask_converter._x1,
          /*offsetY=*/mask_converter._y1,
          stream);
      // SHOW_IMAGE(canvas);
    }
    //
    // Image 2
    //
    if (!stitch_context.is_hard_seam()) {
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
          make_float3(0, 0, 0),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/mask_converter._x2,
          /*offsetY=*/mask_converter._y2,
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
      // SHOW_IMAGE(stitch_context.cudaFull1);
#endif

#if 1
      //
      // Now copy the blending portion of remapped image 2 from the canvas onto the blend image
      //
      // assert(stitch_context.cudaBlendSoftSeam->height() == roi_height(mask_converter.roi_blend_2));
      cuerr = simple_make_full_batch<BaseScalar_t<T_compute>, BaseScalar_t<T_compute>, unsigned char>(
          // Image 1 (float image)
          canvas->data_raw(),
          canvas->width(),
          canvas->height(),
          /*region_width=*/roi_width(mask_converter.roi_blend_2),
          /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(mask_converter.roi_blend_2)*/,
          /*channels=*/3,
          // Batch of masks (optional)
          nullptr,
          0,
          0,
          0,
          /*offsetX=*/mask_converter._x2,
          /*offsetY=*/mask_converter._y2,
          mask_converter._remapper_2.xpos,
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
      // Hard seam
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          sampleImage2.data(),
          sampleImage2.width(),
          sampleImage2.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          make_float3(0, 0, 0),
          /*this_image_index=*/
          0 /* <-- we inverted the mask at load-time to make it a weight, so image 1 is actually 0 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/mask_converter._x2,
          /*offsetY=*/mask_converter._y2,
          stream);
      // SHOW_IMAGE(canvas);
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
          /*offsetX=*/mask_converter._x2 - mask_converter._overlap_pad,
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

  RenderSet display;

  std::string game_id = argv[1];
  std::string game_dir = std::string(::getenv("HOME")) + "/Videos/" + game_id + "/";

  // std::string sample_img_left_path = game_dir + "GX010100.png";
  // std::string sample_img_right_path = game_dir + "GX010019.png";

  std::string sample_img_left_path = game_dir + "GX010097.png";
  std::string sample_img_right_path = game_dir + "GX010016.png";

  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  sample_img_left.convertTo(sample_img_left, CV_32FC3, 1.0 / 255.0);
  sample_img_right.convertTo(sample_img_right, CV_32FC3, 1.0 / 255.0);

  ControlMasks control_masks;
  control_masks.load(game_dir);

  // Compute canvas size
  const size_t canvas_width = std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols);
  const size_t canvas_height = std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows);
  std::cout << "Canvas size: " << canvas_width << " x " << canvas_height << std::endl;

  //
  // MaskConverter
  //
  MaskConverter mask_converter;
  mask_converter._minimize_blend = true;
  mask_converter._canvas_info.width = canvas_width;
  mask_converter._canvas_info.height = canvas_height;
  mask_converter._canvas_info.positions.emplace_back(
      cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos));
  mask_converter._canvas_info.positions.emplace_back(
      cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos));
  mask_converter._remapper_1.width = control_masks.img1_col.cols;
  mask_converter._remapper_1.height = control_masks.img1_col.rows;
  mask_converter._remapper_2.width = control_masks.img2_col.cols;
  mask_converter._remapper_2.height = control_masks.img2_col.rows;

  mask_converter.updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  cv::Mat blend_seam = mask_converter.convertMaskMat(control_masks.whole_seam_mask_image);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

// Configurable parameter: number of pyramid levels.
#ifdef __aarch64__
  // Lower compute, quick and dirty
  int numLevels = 1;
#else
  //int numLevels = 6;
  int numLevels = 1;
  //int numLevels = 6;
#endif

#if 1
  using T = float3;
  using T_compute = float3;
#define CV_T_PIPELINE CV_32FC3
#define CV_T_COMPUTE3 CV_32FC3
#else
  using T = float;
  using T_compute = __half;
#define CV_T_PIPELINE CV_32FC3
#define CV_T_COMPUTE3 CV_16FC3
#endif

  // constexpr int kBatchSize = 1;
  constexpr int kBatchSize = 2;

  StitchingContext<T, T_compute> stitch_context(/*batch_size=*/kBatchSize, /*is_hard_seam=*/numLevels == 0);

  auto canvas = std::make_unique<CudaMat<T>>(
      stitch_context.batch_size(), control_masks.whole_seam_mask_image.cols, control_masks.whole_seam_mask_image.rows);

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
    assert(control_masks.whole_seam_mask_image.type() == CV_8U);
    stitch_context.cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(control_masks.whole_seam_mask_image);
  }

  //
  // The actual incoming imaged
  //
  CudaMat<T> sampleImage1(as_batch(sample_img_left, kBatchSize));
  CudaMat<T> sampleImage2(as_batch(sample_img_right, kBatchSize));

  auto blendedCanvasResult = CudaStitchPano<T, T_compute>::process(
      sampleImage1, sampleImage2, stitch_context, mask_converter, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  auto blendedCanvas = blendedCanvasResult.ConsumeValueOrDie();
  SHOW_IMAGE(blendedCanvas);

  // blendedCanvas = process(sampleImage1, sampleImage2, stitch_context, mask_converter, stream);
  // SHOW_IMAGE(blendedCanvas);

  // cudaStreamSynchronize(stream);

  // display.render("cudaBlendedFull", CudaSurface(cudaBlendedFull), stream);

#if 1 /* perf test */
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    blendedCanvas = CudaStitchPano<T, T_compute>::process(
                        sampleImage1, sampleImage2, stitch_context, mask_converter, stream, std::move(blendedCanvas))
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
