#include "cudaBlend.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// =============================================================================
// Batched Kernels (for image data) and Shared-Mask Kernels
// =============================================================================

// ---------------------
// Batched downsample kernel for RGB images.
// ---------------------
// For each output pixel (x,y), average a 2x2 block from the input image.
__global__ void BatchedDownsampleKernelRGB(
    const float* input,
    int inWidth,
    int inHeight,
    float* output,
    int outWidth,
    int outHeight,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * 3;
  int outImageSize = outWidth * outHeight * 3;
  const float* inImage = input + b * inImageSize;
  float* outImage = output + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;
  float sumR = 0, sumG = 0, sumB = 0;
  int count = 0;
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * 3;
        sumR += inImage[idx + 0];
        sumG += inImage[idx + 1];
        sumB += inImage[idx + 2];
        count++;
      }
    }
  }
  int outIdx = (y * outWidth + x) * 3;
  outImage[outIdx + 0] = sumR / count;
  outImage[outIdx + 1] = sumG / count;
  outImage[outIdx + 2] = sumB / count;
}

// ---------------------
// Shared (non-batched) downsample kernel for a single–channel mask.
// ---------------------
// Note: Since the seam mask is shared among the entire batch, no batch index is used.
__global__ void BatchedDownsampleKernelMask(
    const float* input,
    int inWidth,
    int inHeight,
    float* output,
    int outWidth,
    int outHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inX = x * 2;
  int inY = y * 2;
  float sum = 0.0f;
  int count = 0;
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        sum += input[iy * inWidth + ix];
        count++;
      }
    }
  }
  output[y * outWidth + x] = sum / count;
}

// ---------------------
// Batched upsample kernel for RGB images.
// ---------------------
// Upsamples a low-resolution image by a factor of 2 using bilinear interpolation.
__global__ void BatchedUpsampleKernelRGB(
    const float* input,
    int inWidth,
    int inHeight,
    float* output,
    int outWidth,
    int outHeight,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * 3;
  int outImageSize = outWidth * outHeight * 3;
  const float* inImage = input + b * inImageSize;
  float* outImage = output + b * outImageSize;

  float gx = (float)x / 2.0f;
  float gy = (float)y / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  float dx = gx - gxi;
  float dy = gy - gyi;
  int gxi1 = min(gxi + 1, inWidth - 1);
  int gyi1 = min(gyi + 1, inHeight - 1);

  int idx00 = (gyi * inWidth + gxi) * 3;
  int idx10 = (gyi * inWidth + gxi1) * 3;
  int idx01 = (gyi1 * inWidth + gxi) * 3;
  int idx11 = (gyi1 * inWidth + gxi1) * 3;

  float outR, outG, outB;
  { // R channel interpolation.
    float val00 = inImage[idx00 + 0];
    float val10 = inImage[idx10 + 0];
    float val01 = inImage[idx01 + 0];
    float val11 = inImage[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // G channel interpolation.
    float val00 = inImage[idx00 + 1];
    float val10 = inImage[idx10 + 1];
    float val01 = inImage[idx01 + 1];
    float val11 = inImage[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // B channel interpolation.
    float val00 = inImage[idx00 + 2];
    float val10 = inImage[idx10 + 2];
    float val01 = inImage[idx01 + 2];
    float val11 = inImage[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxOut = (y * outWidth + x) * 3;
  outImage[idxOut + 0] = outR;
  outImage[idxOut + 1] = outG;
  outImage[idxOut + 2] = outB;
}

// ---------------------
// Batched compute Laplacian kernel for RGB images.
// ---------------------
// For each pixel in the high-resolution Gaussian level, compute:
// Laplacian = Gaussian_high - upsample(Gaussian_low)
__global__ void BatchedComputeLaplacianKernelRGB(
    const float* gaussHigh,
    int highWidth,
    int highHeight,
    const float* gaussLow,
    int lowWidth,
    int lowHeight,
    float* laplacian,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int highImageSize = highWidth * highHeight * 3;
  int lowImageSize = lowWidth * lowHeight * 3;
  const float* highImage = gaussHigh + b * highImageSize;
  const float* lowImage = gaussLow + b * lowImageSize;
  float* lapImage = laplacian + b * highImageSize;

  float gx = (float)x / 2.0f;
  float gy = (float)y / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  float dx = gx - gxi;
  float dy = gy - gyi;
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idx00 = (gyi * lowWidth + gxi) * 3;
  int idx10 = (gyi * lowWidth + gxi1) * 3;
  int idx01 = (gyi1 * lowWidth + gxi) * 3;
  int idx11 = (gyi1 * lowWidth + gxi1) * 3;

  float upR, upG, upB;
  { // R channel.
    float val00 = lowImage[idx00 + 0];
    float val10 = lowImage[idx10 + 0];
    float val01 = lowImage[idx01 + 0];
    float val11 = lowImage[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // G channel.
    float val00 = lowImage[idx00 + 1];
    float val10 = lowImage[idx10 + 1];
    float val01 = lowImage[idx01 + 1];
    float val11 = lowImage[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // B channel.
    float val00 = lowImage[idx00 + 2];
    float val10 = lowImage[idx10 + 2];
    float val01 = lowImage[idx01 + 2];
    float val11 = lowImage[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  lapImage[idxHigh + 0] = highImage[idxHigh + 0] - upR;
  lapImage[idxHigh + 1] = highImage[idxHigh + 1] - upG;
  lapImage[idxHigh + 2] = highImage[idxHigh + 2] - upB;
}

// ---------------------
// Batched blend kernel for RGB images.
// ---------------------
// Blends two Laplacian images using a single–channel mask.
// The shared mask is not batched, so no batch offset is applied to it.
__global__ void BatchedBlendKernelRGB(
    const float* lap1,
    const float* lap2,
    const float* mask,
    float* blended,
    int width,
    int height,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int imageSizeRGB = width * height * 3;
  const float* lap1Image = lap1 + b * imageSizeRGB;
  const float* lap2Image = lap2 + b * imageSizeRGB;
  // Use the shared mask (no batch offset)
  const float* maskImage = mask;
  float* blendImage = blended + b * imageSizeRGB;

  int idx = (y * width + x) * 3;
  float m = maskImage[y * width + x];
  float mm1 = 1.0f - m;
  blendImage[idx + 0] = m * lap1Image[idx + 0] + mm1 * lap2Image[idx + 0];
  blendImage[idx + 1] = m * lap1Image[idx + 1] + mm1 * lap2Image[idx + 1];
  blendImage[idx + 2] = m * lap1Image[idx + 2] + mm1 * lap2Image[idx + 2];
}

// ---------------------
// Batched reconstruction kernel for RGB images.
// ---------------------
// Reconstructs the higher-resolution image by upsampling the lower-resolution
// reconstruction and adding the blended Laplacian.
__global__ void BatchedReconstructKernelRGB(
    const float* lowerRes,
    int lowWidth,
    int lowHeight,
    const float* lap,
    int highWidth,
    int highHeight,
    float* reconstruction,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int lowImageSize = lowWidth * lowHeight * 3;
  int highImageSize = highWidth * highHeight * 3;
  const float* lowImage = lowerRes + b * lowImageSize;
  const float* lapImage = lap + b * highImageSize;
  float* reconImage = reconstruction + b * highImageSize;

  float gx = (float)x / 2.0f;
  float gy = (float)y / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  float dx = gx - gxi;
  float dy = gy - gyi;
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idx00 = (gyi * lowWidth + gxi) * 3;
  int idx10 = (gyi * lowWidth + gxi1) * 3;
  int idx01 = (gyi1 * lowWidth + gxi) * 3;
  int idx11 = (gyi1 * lowWidth + gxi1) * 3;

  float upR, upG, upB;
  { // R channel.
    float val00 = lowImage[idx00 + 0];
    float val10 = lowImage[idx10 + 0];
    float val01 = lowImage[idx01 + 0];
    float val11 = lowImage[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // G channel.
    float val00 = lowImage[idx00 + 1];
    float val10 = lowImage[idx10 + 1];
    float val01 = lowImage[idx01 + 1];
    float val11 = lowImage[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // B channel.
    float val00 = lowImage[idx00 + 2];
    float val10 = lowImage[idx10 + 2];
    float val01 = lowImage[idx01 + 2];
    float val11 = lowImage[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  reconImage[idxHigh + 0] = upR + lapImage[idxHigh + 0];
  reconImage[idxHigh + 1] = upG + lapImage[idxHigh + 1];
  reconImage[idxHigh + 2] = upB + lapImage[idxHigh + 2];
}

// =============================================================================
// Host Functions: Batched Laplacian Blending
// =============================================================================

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
    cudaStream_t stream) {
  // For RGB images (3 channels)
  size_t imageSize = imageWidth * imageHeight * 3 * sizeof(float);
  // For mask (single channel, not batched)
  size_t maskSize = imageWidth * imageHeight * sizeof(float);

  // Allocate device memory for level-0 images (batched).
  std::vector<float*> d_gauss1(numLevels);
  std::vector<float*> d_gauss2(numLevels);
  std::vector<float*> d_maskPyr(numLevels);
  std::vector<float*> d_lap1(numLevels);
  std::vector<float*> d_lap2(numLevels);
  std::vector<float*> d_blend(numLevels);

  std::vector<int> widths(numLevels), heights(numLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int i = 1; i < numLevels; i++) {
    widths[i] = (widths[i - 1] + 1) / 2;
    heights[i] = (heights[i - 1] + 1) / 2;
  }

  // Allocate level 0 arrays.
  size_t sizeRGB0 = widths[0] * heights[0] * 3 * batchSize * sizeof(float);
  // Mask level 0: single mask only.
  size_t sizeMask0 = widths[0] * heights[0] * sizeof(float);
  cudaMalloc((void**)&d_gauss1[0], sizeRGB0);
  cudaMalloc((void**)&d_gauss2[0], sizeRGB0);
  cudaMalloc((void**)&d_maskPyr[0], sizeMask0);
  // Copy host images into device memory for level 0.
  cudaMemcpyAsync(d_gauss1[0], h_image1, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_gauss2[0], h_image2, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_maskPyr[0], h_mask, maskSize, cudaMemcpyHostToDevice, stream);

  // Allocate device memory for higher pyramid levels.
  for (int level = 1; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(float);
    size_t sizeMask = widths[level] * heights[level] * sizeof(float); // single mask
    cudaMalloc((void**)&d_gauss1[level], sizeRGB);
    cudaMalloc((void**)&d_gauss2[level], sizeRGB);
    cudaMalloc((void**)&d_maskPyr[level], sizeMask);
  }

  dim3 block(16, 16, 1);

  // 1. Build Gaussian pyramids.
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 gridRGB((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, batchSize);
    // Downsample image1.
    BatchedDownsampleKernelRGB<<<gridRGB, block, 0, stream>>>(
        d_gauss1[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        widths[level + 1],
        heights[level + 1],
        batchSize);
    // Downsample image2.
    BatchedDownsampleKernelRGB<<<gridRGB, block, 0, stream>>>(
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        batchSize);
    // Downsample mask: use a 2D grid since mask is shared.
    {
      dim3 gridMask((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, 1);
      BatchedDownsampleKernelMask<<<gridMask, block, 0, stream>>>(
          d_maskPyr[level], widths[level], heights[level], d_maskPyr[level + 1], widths[level + 1], heights[level + 1]);
    }
  }

  // 2. Build Laplacian pyramids.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(float);
    cudaMalloc((void**)&d_lap1[level], sizeRGB);
    cudaMalloc((void**)&d_lap2[level], sizeRGB);
  }
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedComputeLaplacianKernelRGB<<<grid, block, 0, stream>>>(
        d_gauss1[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap1[level],
        batchSize);
    BatchedComputeLaplacianKernelRGB<<<grid, block, 0, stream>>>(
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap2[level],
        batchSize);
  }
  int last = numLevels - 1;
  {
    size_t lastSize = widths[last] * heights[last] * 3 * batchSize * sizeof(float);
    cudaMemcpyAsync(d_lap1[last], d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_lap2[last], d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream);
  }

  // 3. Blend the Laplacian pyramids.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(float);
    cudaMalloc((void**)&d_blend[level], sizeRGB);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedBlendKernelRGB<<<grid, block, 0, stream>>>(
        d_lap1[level],
        d_lap2[level],
        d_maskPyr[level], // shared mask
        d_blend[level],
        widths[level],
        heights[level],
        batchSize);
  }

  // 4. Reconstruct the final blended image.
  float* d_reconstruct = nullptr;
  cudaMalloc((void**)&d_reconstruct, widths[last] * heights[last] * 3 * batchSize * sizeof(float));
  cudaMemcpyAsync(
      d_reconstruct,
      d_blend[last],
      widths[last] * heights[last] * 3 * batchSize * sizeof(float),
      cudaMemcpyDeviceToDevice,
      stream);
  for (int level = numLevels - 2; level >= 0; level--) {
    float* d_temp = nullptr;
    size_t highSize = widths[level] * heights[level] * 3 * batchSize * sizeof(float);
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedReconstructKernelRGB<<<grid, block, 0, stream>>>(
        d_reconstruct,
        widths[level + 1],
        heights[level + 1],
        d_blend[level],
        widths[level],
        heights[level],
        d_temp,
        batchSize);
    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }
  cudaMemcpyAsync(h_output, d_reconstruct, imageSize * batchSize, cudaMemcpyDeviceToHost, stream);
  cudaFree(d_reconstruct);

  // Cleanup all allocated device memory.
  for (int level = 0; level < numLevels; level++) {
    cudaFree(d_gauss1[level]);
    cudaFree(d_gauss2[level]);
    cudaFree(d_maskPyr[level]);
    cudaFree(d_lap1[level]);
    cudaFree(d_lap2[level]);
    cudaFree(d_blend[level]);
  }

  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  size_t imageSize = context.imageWidth * context.imageHeight * 3 * sizeof(float);
  size_t maskSize = context.imageWidth * context.imageHeight * sizeof(float);

  if (!context.initialized) {
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < context.numLevels; i++) {
      context.widths[i] = (context.widths[i - 1] + 1) / 2;
      context.heights[i] = (context.heights[i - 1] + 1) / 2;
      assert(context.widths[i] && context.heights[i]);
    }
    // Allocate device memory for each level.
    for (int level = 0; level < context.numLevels; level++) {
      size_t sizeRGB = context.widths[level] * context.heights[level] * 3 * sizeof(float);
      size_t sizeMask = context.widths[level] * context.heights[level] * sizeof(float);
      cudaMalloc((void**)&context.d_gauss1[level], sizeRGB);
      cudaMalloc((void**)&context.d_gauss2[level], sizeRGB);
      cudaMalloc((void**)&context.d_maskPyr[level], sizeMask);
      cudaMalloc((void**)&context.d_lap1[level], sizeRGB);
      cudaMalloc((void**)&context.d_lap2[level], sizeRGB);
      cudaMalloc((void**)&context.d_blend[level], sizeRGB);
    }
    // Copy level 0 mask (shared) from d_mask.
    cudaMemcpyAsync(context.d_maskPyr[0], d_mask, maskSize, cudaMemcpyDeviceToDevice, stream);
  }
  // Set level 0 images.
  cudaMemcpyAsync(context.d_gauss1[0], d_image1, imageSize * context.batchSize, cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(context.d_gauss2[0], d_image2, imageSize * context.batchSize, cudaMemcpyDeviceToDevice, stream);

  dim3 block(16, 16, 1);
  // 1. Build Gaussian pyramid for images and mask.
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 grid(
        (context.widths[level + 1] + block.x - 1) / block.x,
        (context.heights[level + 1] + block.y - 1) / block.y,
        context.batchSize);
    BatchedDownsampleKernelRGB<<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.batchSize);
    BatchedDownsampleKernelRGB<<<grid, block, 0, stream>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.batchSize);
    {
      dim3 gridMask(
          (context.widths[level + 1] + block.x - 1) / block.x, (context.heights[level + 1] + block.y - 1) / block.y, 1);
      BatchedDownsampleKernelMask<<<gridMask, block, 0, stream>>>(
          context.d_maskPyr[level],
          context.widths[level],
          context.heights[level],
          context.d_maskPyr[level + 1],
          context.widths[level + 1],
          context.heights[level + 1]);
    }
  }
  // 2. Build Laplacian pyramids.
  for (int level = 0; level < context.numLevels; level++) {
    size_t sizeRGB = context.widths[level] * context.heights[level] * 3 * context.batchSize * sizeof(float);
    cudaMalloc((void**)&context.d_lap1[level], sizeRGB);
    cudaMalloc((void**)&context.d_lap2[level], sizeRGB);
  }
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedComputeLaplacianKernelRGB<<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap1[level],
        context.batchSize);
    BatchedComputeLaplacianKernelRGB<<<grid, block, 0, stream>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap2[level],
        context.batchSize);
  }
  int last = context.numLevels - 1;
  cudaMemcpyAsync(
      context.d_lap1[last],
      context.d_gauss1[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream);
  cudaMemcpyAsync(
      context.d_lap2[last],
      context.d_gauss2[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream);
  // 3. Blend pyramids.
  for (int level = 0; level < context.numLevels; level++) {
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedBlendKernelRGB<<<grid, block, 0, stream>>>(
        context.d_lap1[level],
        context.d_lap2[level],
        context.d_maskPyr[level],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        context.batchSize);
  }
  // 4. Reconstruct final image.
  float* d_reconstruct = nullptr;
  cudaMalloc(
      (void**)&d_reconstruct, context.widths[last] * context.heights[last] * 3 * sizeof(float) * context.batchSize);
  cudaMemcpyAsync(
      d_reconstruct,
      context.d_blend[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream);
  for (int level = context.numLevels - 2; level >= 0; level--) {
    float* d_temp = nullptr;
    size_t highSize = context.widths[level] * context.heights[level] * 3 * sizeof(float) * context.batchSize;
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedReconstructKernelRGB<<<grid, block, 0, stream>>>(
        d_reconstruct,
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        d_temp,
        context.batchSize);
    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }
  cudaMemcpyAsync(d_output, d_reconstruct, imageSize * context.batchSize, cudaMemcpyDeviceToDevice, stream);
  cudaFree(d_reconstruct);
  context.initialized = true;
  return cudaGetLastError();
}
