// laplacian_blend_pyramid_rgb.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> // for std::min
#include "cudaBlend.h"

#include <vector>

// ---------------------------------------------------------------------
// Downsample kernel for RGB images.
// For each output pixel (x,y), a 2x2 block of the input image is averaged.
// The image is assumed to be stored in interleaved RGB order.
// ---------------------------------------------------------------------
__global__ void downsampleKernelRGB(
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
  float sumR = 0, sumG = 0, sumB = 0;
  int count = 0;
  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * 3;
        sumR += input[idx + 0];
        sumG += input[idx + 1];
        sumB += input[idx + 2];
        count++;
      }
    }
  }
  int outIdx = (y * outWidth + x) * 3;
  output[outIdx + 0] = sumR / count;
  output[outIdx + 1] = sumG / count;
  output[outIdx + 2] = sumB / count;
}

// ---------------------------------------------------------------------
// Downsample kernel for a single–channel mask.
// ---------------------------------------------------------------------
__global__ void downsampleKernelMask(
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

// ---------------------------------------------------------------------
// Upsample kernel for RGB images.
// Upsamples a lower–resolution image by a factor of 2 using bilinear interpolation.
// ---------------------------------------------------------------------
__global__ void upsampleKernelRGB(
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

  // Map output coordinate to input (low–res) coordinate space.
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
  { // Interpolate for R channel.
    float val00 = input[idx00 + 0];
    float val10 = input[idx10 + 0];
    float val01 = input[idx01 + 0];
    float val11 = input[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Interpolate for G channel.
    float val00 = input[idx00 + 1];
    float val10 = input[idx10 + 1];
    float val01 = input[idx01 + 1];
    float val11 = input[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Interpolate for B channel.
    float val00 = input[idx00 + 2];
    float val10 = input[idx10 + 2];
    float val01 = input[idx01 + 2];
    float val11 = input[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    outB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxOut = (y * outWidth + x) * 3;
  output[idxOut + 0] = outR;
  output[idxOut + 1] = outG;
  output[idxOut + 2] = outB;
}

// ---------------------------------------------------------------------
// Compute Laplacian kernel for RGB images.
// For each pixel in the higher–resolution Gaussian level (level L),
// compute Laplacian = Gaussian[L] – upsample(Gaussian[L+1]).
// ---------------------------------------------------------------------
__global__ void computeLaplacianKernelRGB(
    const float* gaussHigh,
    int highWidth,
    int highHeight,
    const float* gaussLow,
    int lowWidth,
    int lowHeight,
    float* laplacian) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

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
  { // Upsample for R channel.
    float val00 = gaussLow[idx00 + 0];
    float val10 = gaussLow[idx10 + 0];
    float val01 = gaussLow[idx01 + 0];
    float val11 = gaussLow[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Upsample for G channel.
    float val00 = gaussLow[idx00 + 1];
    float val10 = gaussLow[idx10 + 1];
    float val01 = gaussLow[idx01 + 1];
    float val11 = gaussLow[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Upsample for B channel.
    float val00 = gaussLow[idx00 + 2];
    float val10 = gaussLow[idx10 + 2];
    float val01 = gaussLow[idx01 + 2];
    float val11 = gaussLow[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  laplacian[idxHigh + 0] = gaussHigh[idxHigh + 0] - upR;
  laplacian[idxHigh + 1] = gaussHigh[idxHigh + 1] - upG;
  laplacian[idxHigh + 2] = gaussHigh[idxHigh + 2] - upB;
}

// ---------------------------------------------------------------------
// Blend kernel for RGB images.
// Blends two Laplacian images using a single–channel mask.
// ---------------------------------------------------------------------
__global__ void blendKernelRGB(
    const float* lap1,
    const float* lap2,
    const float* mask,
    float* blended,
    int width,
    int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int idx = (y * width + x) * 3;
  float m = mask[y * width + x];
  blended[idx + 0] = m * lap1[idx + 0] + (1.0f - m) * lap2[idx + 0];
  blended[idx + 1] = m * lap1[idx + 1] + (1.0f - m) * lap2[idx + 1];
  blended[idx + 2] = m * lap1[idx + 2] + (1.0f - m) * lap2[idx + 2];
}

// ---------------------------------------------------------------------
// Reconstruction kernel for RGB images.
// For a given level, reconstruct the higher–resolution image by upsampling
// the lower–resolution reconstruction and adding the blended Laplacian.
// ---------------------------------------------------------------------
__global__ void reconstructKernelRGB(
    const float* lowerRes,
    int lowWidth,
    int lowHeight,
    const float* lap,
    int highWidth,
    int highHeight,
    float* reconstruction) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

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
  { // Upsample for R channel.
    float val00 = lowerRes[idx00 + 0];
    float val10 = lowerRes[idx10 + 0];
    float val01 = lowerRes[idx01 + 0];
    float val11 = lowerRes[idx11 + 0];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Upsample for G channel.
    float val00 = lowerRes[idx00 + 1];
    float val10 = lowerRes[idx10 + 1];
    float val01 = lowerRes[idx01 + 1];
    float val11 = lowerRes[idx11 + 1];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // Upsample for B channel.
    float val00 = lowerRes[idx00 + 2];
    float val10 = lowerRes[idx10 + 2];
    float val01 = lowerRes[idx01 + 2];
    float val11 = lowerRes[idx11 + 2];
    float val0 = val00 * (1.0f - dx) + val10 * dx;
    float val1 = val01 * (1.0f - dx) + val11 * dx;
    upB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  reconstruction[idxHigh + 0] = upR + lap[idxHigh + 0];
  reconstruction[idxHigh + 1] = upG + lap[idxHigh + 1];
  reconstruction[idxHigh + 2] = upB + lap[idxHigh + 2];
}

// ---------------------------------------------------------------------
// Host code: main blending pipeline.
// ---------------------------------------------------------------------

cudaError_t cudaLaplacianBlend(
    const float* h_image1,
    const float* h_image2,
    const float* h_mask,
    float* h_output,
    int imageWidth,
    int imageHeight,
    int numLevels) {
  // -----------------------------------------------------------------
  // Configurable parameters:
  //   numLevels: the number of pyramid levels (including the full–resolution level)
  //   imageWidth, imageHeight: dimensions of the full–resolution images.
  //   The images are assumed to be RGB (interleaved) and the mask single–channel.
  // -----------------------------------------------------------------
  // const int numLevels = 4; // <-- Configurable number of pyramid levels
  // const int imageWidth = 1024;
  // const int imageHeight = 1024;

  // For RGB images (3 channels)
  size_t imageSize = imageWidth * imageHeight * 3 * sizeof(float);
  // For mask (single channel)
  size_t maskSize = imageWidth * imageHeight * sizeof(float);

  // Allocate host memory (in real use, load your images and mask)
  // float* h_image1 = new float[imageWidth * imageHeight * 3];
  // float* h_image2 = new float[imageWidth * imageHeight * 3];
  // float* h_mask = new float[imageWidth * imageHeight];
  // float* h_output = new float[imageWidth * imageHeight * 3];

  // ... Initialize h_image1, h_image2, and h_mask as needed ...

  // -----------------------------------------------------------------
  // Allocate device arrays for the pyramids.
  // For each pyramid level we will allocate storage for:
  //   - Gaussian pyramid for image1 and image2 (RGB)
  //   - Gaussian pyramid for mask (single channel)
  //   - Laplacian pyramid for image1 and image2 (RGB)
  //   - Blended Laplacian pyramid (RGB)
  // -----------------------------------------------------------------
  float* d_gauss1[numLevels];
  float* d_gauss2[numLevels];
  float* d_maskPyr[numLevels];
  float* d_lap1[numLevels];
  float* d_lap2[numLevels];
  float* d_blend[numLevels];

  // Store dimensions for each pyramid level.
  int widths[numLevels];
  int heights[numLevels];
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int i = 1; i < numLevels; i++) {
    // Use ceiling division in case dimensions are not even.
    widths[i] = (widths[i - 1] + 1) / 2;
    heights[i] = (heights[i - 1] + 1) / 2;
  }

  // Allocate device memory for each pyramid level.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * sizeof(float);
    size_t sizeMask = widths[level] * heights[level] * sizeof(float);
    cudaMalloc((void**)&d_gauss1[level], sizeRGB);
    cudaMalloc((void**)&d_gauss2[level], sizeRGB);
    cudaMalloc((void**)&d_maskPyr[level], sizeMask);
    cudaMalloc((void**)&d_lap1[level], sizeRGB);
    cudaMalloc((void**)&d_lap2[level], sizeRGB);
    cudaMalloc((void**)&d_blend[level], sizeRGB);
  }

  // Copy full–resolution images and mask to level 0 of the Gaussian pyramids.
  cudaMemcpy(d_gauss1[0], h_image1, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gauss2[0], h_image2, imageSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_maskPyr[0], h_mask, maskSize, cudaMemcpyHostToDevice);

  // Kernel launch parameters.
  dim3 block(16, 16);

  // -----------------------------------------------------------------
  // 1. Build Gaussian pyramids by downsampling each level.
  // -----------------------------------------------------------------
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 gridRGB((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y);
    // Downsample image1.
    downsampleKernelRGB<<<gridRGB, block>>>(
        d_gauss1[level], widths[level], heights[level], d_gauss1[level + 1], widths[level + 1], heights[level + 1]);
    // Downsample image2.
    downsampleKernelRGB<<<gridRGB, block>>>(
        d_gauss2[level], widths[level], heights[level], d_gauss2[level + 1], widths[level + 1], heights[level + 1]);
    // Downsample mask.
    downsampleKernelMask<<<gridRGB, block>>>(
        d_maskPyr[level], widths[level], heights[level], d_maskPyr[level + 1], widths[level + 1], heights[level + 1]);
  }

  // -----------------------------------------------------------------
  // 2. Build Laplacian pyramids for both images.
  // For levels 0 to numLevels-2, compute: Lap = Gaussian - upsample(Gaussian_next).
  // The last level Laplacian is simply the Gaussian image.
  // -----------------------------------------------------------------
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y);
    computeLaplacianKernelRGB<<<grid, block>>>(
        d_gauss1[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap1[level]);
    computeLaplacianKernelRGB<<<grid, block>>>(
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap2[level]);
  }
  int last = numLevels - 1;
  cudaMemcpy(d_lap1[last], d_gauss1[last], widths[last] * heights[last] * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_lap2[last], d_gauss2[last], widths[last] * heights[last] * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

  // -----------------------------------------------------------------
  // 3. Blend the Laplacian pyramids at each level using the corresponding mask pyramid.
  // -----------------------------------------------------------------
  for (int level = 0; level < numLevels; level++) {
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y);
    blendKernelRGB<<<grid, block>>>(
        d_lap1[level], d_lap2[level], d_maskPyr[level], d_blend[level], widths[level], heights[level]);
  }

  // -----------------------------------------------------------------
  // 4. Reconstruct the final blended image from the blended pyramid.
  // Start from the smallest level and iteratively upsample and add.
  // -----------------------------------------------------------------
  float* d_reconstruct;
  cudaMalloc((void**)&d_reconstruct, widths[last] * heights[last] * 3 * sizeof(float));
  cudaMemcpy(d_reconstruct, d_blend[last], widths[last] * heights[last] * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

  float* d_temp = nullptr;
  for (int level = numLevels - 2; level >= 0; level--) {
    size_t highSize = widths[level] * heights[level] * 3 * sizeof(float);
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y);
    reconstructKernelRGB<<<grid, block>>>(
        d_reconstruct, widths[level + 1], heights[level + 1], d_blend[level], widths[level], heights[level], d_temp);
    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }

  // d_reconstruct now holds the final blended image at full resolution.
  cudaMemcpy(h_output, d_reconstruct, imageSize, cudaMemcpyDeviceToHost);

  // ... h_output now contains your final blended RGB image.

  // -----------------------------------------------------------------
  // Cleanup: free all allocated device and host memory.
  // -----------------------------------------------------------------
  for (int level = 0; level < numLevels; level++) {
    cudaFree(d_gauss1[level]);
    cudaFree(d_gauss2[level]);
    cudaFree(d_maskPyr[level]);
    cudaFree(d_lap1[level]);
    cudaFree(d_lap2[level]);
    cudaFree(d_blend[level]);
  }
  cudaFree(d_reconstruct);
  // delete[] h_image1;
  // delete[] h_image2;
  // delete[] h_mask;
  // delete[] h_output;

  return cudaGetLastError();
}

//
//
//
//
//
//

cudaError_t cudaLaplacianBlendWithContext(
    const float* d_image1,
    const float* d_image2,
    const float* d_mask,
    float* d_output,
    CudaLaplacianBlendContext& context) {
  // -----------------------------------------------------------------
  // Configurable parameters:
  //   numLevels: the number of pyramid levels (including the full–resolution level)
  //   imageWidth, imageHeight: dimensions of the full–resolution images.
  //   The images are assumed to be RGB (interleaved) and the mask single–channel.
  // -----------------------------------------------------------------
  // const int numLevels = 4; // <-- Configurable number of pyramid levels
  // const int imageWidth = 1024;
  // const int imageHeight = 1024;

  // For RGB images (3 channels)
  size_t imageSize = context.imageWidth * context.imageHeight * 3 * sizeof(float);
  // For mask (single channel)
  size_t maskSize = context.imageWidth * context.imageHeight * sizeof(float);

  // Allocate host memory (in real use, load your images and mask)
  // float* h_image1 = new float[imageWidth * imageHeight * 3];
  // float* h_image2 = new float[imageWidth * imageHeight * 3];
  // float* h_mask = new float[imageWidth * imageHeight];
  // float* h_output = new float[imageWidth * imageHeight * 3];

  // ... Initialize h_image1, h_image2, and h_mask as needed ...

  // -----------------------------------------------------------------
  // Allocate device arrays for the pyramids.
  // For each pyramid level we will allocate storage for:
  //   - Gaussian pyramid for image1 and image2 (RGB)
  //   - Gaussian pyramid for mask (single channel)
  //   - Laplacian pyramid for image1 and image2 (RGB)
  //   - Blended Laplacian pyramid (RGB)
  // -----------------------------------------------------------------
  // float* d_gauss1[numLevels];
  // float* d_gauss2[numLevels];
  // float* d_maskPyr[numLevels];
  // float* d_lap1[numLevels];
  // float* d_lap2[numLevels];
  // float* d_blend[numLevels];

  // Store dimensions for each pyramid level.
  // int widths[numLevels];
  // int heights[numLevels];

  if (!context.initialized) {
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < context.numLevels; i++) {
      // Use ceiling division in case dimensions are not even.
      context.widths[i] = (context.widths[i - 1] + 1) / 2;
      context.heights[i] = (context.heights[i - 1] + 1) / 2;
    }

    // Allocate device memory for each pyramid level.
    for (int level = 0; level < context.numLevels; level++) {
      size_t sizeRGB = context.widths[level] * context.heights[level] * 3 * sizeof(float);
      size_t sizeMask = context.widths[level] * context.heights[level] * sizeof(float);
      // TODO: won't need to allocate 0-level gauss when we just reassign the pointer to incoming image
      cudaMalloc((void**)&context.d_gauss1[level], sizeRGB);
      cudaMalloc((void**)&context.d_gauss2[level], sizeRGB);
      cudaMalloc((void**)&context.d_maskPyr[level], sizeMask);
      cudaMalloc((void**)&context.d_lap1[level], sizeRGB);
      cudaMalloc((void**)&context.d_lap2[level], sizeRGB);
      cudaMalloc((void**)&context.d_blend[level], sizeRGB);
    }
    cudaMemcpy(context.d_maskPyr[0], d_mask, maskSize, cudaMemcpyDeviceToDevice);
  }

  // Copy full–resolution images and mask to level 0 of the Gaussian pyramids.
  // TODO: Just set the pointer instead and don;t allocate this level
  cudaMemcpy(context.d_gauss1[0], d_image1, imageSize, cudaMemcpyDeviceToDevice);
  cudaMemcpy(context.d_gauss2[0], d_image2, imageSize, cudaMemcpyDeviceToDevice);

  // Kernel launch parameters.
  dim3 block(16, 16);

  // -----------------------------------------------------------------
  // 1. Build Gaussian pyramids by downsampling each level.
  // -----------------------------------------------------------------
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 gridRGB(
        (context.widths[level + 1] + block.x - 1) / block.x, (context.heights[level + 1] + block.y - 1) / block.y);
    // Downsample image1.
    downsampleKernelRGB<<<gridRGB, block>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1]);
    // Downsample image2.
    downsampleKernelRGB<<<gridRGB, block>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1]);
    if (!context.initialized) {
      // Downsample mask. (only needs to hbe done the first time)
      downsampleKernelMask<<<gridRGB, block>>>(
          context.d_maskPyr[level],
          context.widths[level],
          context.heights[level],
          context.d_maskPyr[level + 1],
          context.widths[level + 1],
          context.heights[level + 1]);
    }
  }

  // -----------------------------------------------------------------
  // 2. Build Laplacian pyramids for both images.
  // For levels 0 to numLevels-2, compute: Lap = Gaussian - upsample(Gaussian_next).
  // The last level Laplacian is simply the Gaussian image.
  // -----------------------------------------------------------------
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 grid((context.widths[level] + block.x - 1) / block.x, (context.heights[level] + block.y - 1) / block.y);
    computeLaplacianKernelRGB<<<grid, block>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap1[level]);
    computeLaplacianKernelRGB<<<grid, block>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap2[level]);
  }
  int last = context.numLevels - 1;
  cudaMemcpy(
      context.d_lap1[last],
      context.d_gauss1[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float),
      cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      context.d_lap2[last],
      context.d_gauss2[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float),
      cudaMemcpyDeviceToDevice);

  // -----------------------------------------------------------------
  // 3. Blend the Laplacian pyramids at each level using the corresponding mask pyramid.
  // -----------------------------------------------------------------
  for (int level = 0; level < context.numLevels; level++) {
    dim3 grid((context.widths[level] + block.x - 1) / block.x, (context.heights[level] + block.y - 1) / block.y);
    blendKernelRGB<<<grid, block>>>(
        context.d_lap1[level],
        context.d_lap2[level],
        context.d_maskPyr[level],
        context.d_blend[level],
        context.widths[level],
        context.heights[level]);
  }

  // -----------------------------------------------------------------
  // 4. Reconstruct the final blended image from the blended pyramid.
  // Start from the smallest level and iteratively upsample and add.
  // -----------------------------------------------------------------
  float* d_reconstruct;
  cudaMalloc((void**)&d_reconstruct, context.widths[last] * context.heights[last] * 3 * sizeof(float));
  cudaMemcpy(
      d_reconstruct,
      context.d_blend[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(float),
      cudaMemcpyDeviceToDevice);

  // TODO: cache these allocations/frees for reconstruct?
  float* d_temp = nullptr;
  for (int level = context.numLevels - 2; level >= 0; level--) {
    size_t highSize = context.widths[level] * context.heights[level] * 3 * sizeof(float);
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid((context.widths[level] + block.x - 1) / block.x, (context.heights[level] + block.y - 1) / block.y);
    reconstructKernelRGB<<<grid, block>>>(
        d_reconstruct,
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        d_temp);
    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }

  // d_reconstruct now holds the final blended image at full resolution.
  cudaMemcpy(d_output, d_reconstruct, imageSize, cudaMemcpyDeviceToDevice);

  // ... h_output now contains your final blended RGB image.

  // -----------------------------------------------------------------
  // Cleanup: free all allocated device and host memory.
  // -----------------------------------------------------------------
  // for (int level = 0; level < numLevels; level++) {
  //   cudaFree(d_gauss1[level]);
  //   cudaFree(d_gauss2[level]);
  //   cudaFree(d_maskPyr[level]);
  //   cudaFree(d_lap1[level]);
  //   cudaFree(d_lap2[level]);
  //   cudaFree(d_blend[level]);
  // }
  cudaFree(d_reconstruct);
  // delete[] h_image1;
  // delete[] h_image2;
  // delete[] h_mask;
  // delete[] h_output;
  context.initialized = true;
  return cudaGetLastError();
}
