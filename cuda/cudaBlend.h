#pragma once

#include <cuda_runtime.h>

#include <vector>

struct CudaLaplacianBlendContext {
  CudaLaplacianBlendContext(int image_width, int image_height, int num_levels)
      : numLevels(num_levels),
        imageWidth(image_width),
        imageHeight(image_height),
        widths(num_levels),
        heights(num_levels),
        d_gauss1(num_levels, nullptr),
        d_gauss2(num_levels, nullptr),
        d_maskPyr(num_levels, nullptr),
        d_lap1(num_levels, nullptr),
        d_lap2(num_levels, nullptr),
        d_blend(num_levels, nullptr) {}

  static constexpr void maybeCudaFree(void* p) {
    if (p) {
      cudaFree(p);
    }
  }

  ~CudaLaplacianBlendContext() {
    for (int level = 0; level < numLevels; level++) {
      maybeCudaFree(d_gauss1[level]);
      maybeCudaFree(d_gauss2[level]);
      maybeCudaFree(d_maskPyr[level]);
      maybeCudaFree(d_lap1[level]);
      maybeCudaFree(d_lap2[level]);
      maybeCudaFree(d_blend[level]);
    }
  }
  const int numLevels;
  const int imageWidth;
  const int imageHeight;
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<float*> d_gauss1;
  std::vector<float*> d_gauss2;
  std::vector<float*> d_maskPyr;
  std::vector<float*> d_lap1;
  std::vector<float*> d_lap2;
  std::vector<float*> d_blend;
  bool initialized{false};
};

cudaError_t cudaLaplacianBlend(
    const float* h_image1,
    const float* h_image2,
    const float* h_mask,
    float* h_output,
    int imageWidth,
    int imageHeight,
    int numLevels);

cudaError_t cudaLaplacianBlendWithContext(
    const float* d_image1,
    const float* d_image2,
    const float* d_mask,
    float* d_output,
    CudaLaplacianBlendContext& context);

/**
 * @brief Remaps a source image to a destination image using provided mapping arrays.
 *
 * For each pixel in the destination image, this kernel uses the mapping arrays
 * (`mapX` and `mapY`) to determine the corresponding pixel in the source image.
 * If the mapped coordinates fall within the valid range of the source image,
 * the source pixel (consisting of three floats for RGB) is copied to the destination.
 * Otherwise, the destination pixel is set to the specified default color.
 *
 * The mapping arrays contain unsigned short values representing the x and y coordinates
 * in the source image.
 *
 * @param src   Pointer to the source image data (float array, 3 channels per pixel).
 * @param srcW  Width of the source image.
 * @param srcH  Height of the source image.
 * @param dest  Pointer to the destination image data (float array, 3 channels per pixel).
 * @param destW Width of the destination image.
 * @param destH Height of the destination image.
 * @param mapX  Pointer to the mapping array for the x-coordinate (one unsigned short per destination pixel).
 * @param mapY  Pointer to the mapping array for the y-coordinate (one unsigned short per destination pixel).
 * @param defR  Default red channel value for pixels mapped out-of-range.
 * @param defG  Default green channel value for pixels mapped out-of-range.
 * @param defB  Default blue channel value for pixels mapped out-of-range.
 */
cudaError_t remap_kernel(
    const float* src,
    int srcW,
    int srcH,
    float* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    float defR,
    float defG,
    float defB);
