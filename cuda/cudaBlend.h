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

/**
 * @brief Batched Laplacian blending.
 *
 * Copies host images (batched layout) and a shared mask to device memory, builds Gaussian and Laplacian pyramids,
 * blends the Laplacian pyramids, reconstructs the final blended images, and copies the result back to host.
 *
 * @param h_image1 Host pointer to the first set of full-resolution images.
 * @param h_image2 Host pointer to the second set of full-resolution images.
 * @param h_mask Host pointer to the full-resolution shared mask.
 * @param h_output Host pointer where the final blended images will be copied.
 * @param imageWidth Width of each full-resolution image.
 * @param imageHeight Height of each full-resolution image.
 * @param numLevels Number of pyramid levels.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
cudaError_t cudaBatchedLaplacianBlend(
    const float* h_image1,
    const float* h_image2,
    const float* h_mask,
    float* h_output,
    int imageWidth,
    int imageHeight,
    int numLevels,
    int batchSize,
    cudaStream_t stream);
/**
 * @brief Batched Laplacian blending with a preallocated context.
 *
 * Uses a preallocated context to store intermediate pyramid arrays, builds Gaussian and Laplacian pyramids,
 * blends the Laplacian pyramids, reconstructs the final blended image, and stores the result in d_output.
 *
 * @param d_image1 Device pointer to the first set of full-resolution images.
 * @param d_image2 Device pointer to the second set of full-resolution images.
 * @param d_mask Device pointer to the shared mask.
 * @param d_output Device pointer where the final blended images will be stored.
 * @param context Reference to a CudaBatchLaplacianBlendContext structure that holds preallocated arrays and blending
 * parameters.
 * @param stream CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
cudaError_t cudaBatchedLaplacianBlendWithContext(
    const float* d_image1,
    const float* d_image2,
    const float* d_mask,
    float* d_output,
    CudaBatchLaplacianBlendContext& context,
    cudaStream_t stream);
