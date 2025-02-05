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

// =============================================================================
// Batched Context Structure (for images only; the mask is shared across the
// batch)
// =============================================================================
struct CudaBatchLaplacianBlendContext {
  CudaBatchLaplacianBlendContext(int image_width, int image_height, int num_levels, int batch_size)
      : numLevels(num_levels),
        imageWidth(image_width),
        imageHeight(image_height),
        batchSize(batch_size),
        widths(num_levels),
        heights(num_levels),
        d_gauss1(num_levels, nullptr),
        d_gauss2(num_levels, nullptr),
        d_maskPyr(num_levels, nullptr),
        d_lap1(num_levels, nullptr),
        d_lap2(num_levels, nullptr),
        d_blend(num_levels, nullptr) {}

  // Helper: free pointer if non-null.
  static constexpr void maybeCudaFree(void* p) {
    if (p) {
      cudaFree(p);
    }
  }

  ~CudaBatchLaplacianBlendContext() {
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
  const int batchSize;
  std::vector<int> widths;
  std::vector<int> heights;
  std::vector<float*> d_gauss1;
  std::vector<float*> d_gauss2;
  // Note: The mask is shared, so each level’s allocation is only for one image.
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

cudaError_t cudaBatchedLaplacianBlendWithContext(
    const float* d_image1,
    const float* d_image2,
    const float* d_mask,
    float* d_output,
    CudaBatchLaplacianBlendContext& context);

