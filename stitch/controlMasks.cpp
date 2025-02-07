#include "controlMasks.h"

#include <tiffio.h>

namespace hm {
namespace pano {

namespace {

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

std::vector<SpatialTiff> normalize_positions(std::vector<SpatialTiff>&& positions) {
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

// A structure to hold TIFF information
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

SpatialTiff get_geo_tiff(const std::string& filename) {
  TiffInfo info = getTiffInfo(filename);
  return SpatialTiff{.xpos = info.xPosition * info.xResolution, .ypos = info.yPosition * info.yResolution};
}

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

} // namespace

bool ControlMasks::load(std::string game_dir) {
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
#if 0
      whole_seam_mask_image = make_fake_mask_like(whole_seam_mask_image);
#endif
  positions = normalize_positions(std::vector<SpatialTiff>{get_geo_tiff(mapping_0_pos), get_geo_tiff(mapping_1_pos)});
  return true;
}

} // namespace pano
} // namespace hm
