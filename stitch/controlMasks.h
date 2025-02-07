#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace hm {
namespace pano {

struct SpatialTiff {
  // position in pixels
  float xpos;
  float ypos;
};

struct ControlMasks {
  bool load(std::string game_dir);

 public:
  cv::Mat img1_col;
  cv::Mat img1_row;
  cv::Mat img2_col;
  cv::Mat img2_row;
  cv::Mat whole_seam_mask_image;
  std::vector<SpatialTiff> positions;
};

} // namespace pano
} // namespace hm
