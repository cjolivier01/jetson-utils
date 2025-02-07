#pragma once

#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include <vector>

namespace hm {
namespace pano {

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

/**
 *   _____                               __  __
 *  / ____|                             |  \/  |
 * | |      __ _ _ __ __   __ __ _  ___ | \  / | __ _ _ __   __ _  __ _  ___  _ __
 * | |     / _` | '_ \\ \ / // _` |/ __|| |\/| |/ _` | '_ \ / _` |/ _` |/ _ \| '__|
 * | |____| (_| | | | |\ V /| (_| |\__ \| |  | | (_| | | | | (_| | (_| |  __/| |
 *  \_____|\__,_|_| |_| \_/  \__,_||___/|_|  |_|\__,_|_| |_|\__,_|\__, |\___||_|
 *                                                                 __/ |
 *                                                                |___/
 */
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
  CanvasManager(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad = 128);

  // This function updates blending parameters if _minimize_blend is true.
  void updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2);

  // Example conversion function that returns a cv::Mat with the same size as the canvas.
  // If _minimize_blend is true, it also updates the blend parameters and returns a cropped region.
  cv::Mat convertMaskMat(const cv::Mat& mask);

  cv::Rect2i roi_blend_1;
  cv::Rect2i roi_blend_2;

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

} // namespace pano
} // namespace hm
