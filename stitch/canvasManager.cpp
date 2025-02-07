#include "canvasManager.h"

namespace hm {
namespace pano {

// Constructor (if needed)
CanvasManager::CanvasManager(CanvasInfo canvas_info, bool minimize_blend, int overlap_pad)
    : _x1(0),
      _y1(0),
      _x2(0),
      _y2(0),
      canvas_info_(canvas_info),
      _overlapping_width(0),
      _minimize_blend(minimize_blend),
      _overlap_pad(overlap_pad) {}

// This function updates blending parameters if _minimize_blend is true.
void CanvasManager::updateMinimizeBlend(const cv::Size& remapped_size_1, const cv::Size& remapped_size_2) {
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
    roi_blend_1 = {_x2 - _overlap_pad, 0, remapped_size_1.width - _x2 - _overlap_pad, remapped_size_1.height};
    roi_blend_2 = {0, 0, _overlapping_width + _overlap_pad, remapped_size_2.height};
  }
}

// Example conversion function that returns a cv::Mat with the same size as the canvas.
// If _minimize_blend is true, it also updates the blend parameters and returns a cropped region.
cv::Mat CanvasManager::convertMaskMat(const cv::Mat& mask) {
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

} // namespace pano
} // namespace hm
