#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <math.h>
// #include <stdlib.h>
#include "cudaRemap.h"

namespace {
__global__ void remapKernel(
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
    float defB) {
  // Compute destination pixel coordinates.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Check destination bounds.
  if (x >= destW || y >= destH)
    return;

  // Compute the linear index for the destination pixel.
  int destIdx = y * destW + x;

  // Retrieve mapping coordinates from unsigned short arrays and cast to int.
  int srcX = static_cast<int>(mapX[destIdx]);
  int srcY = static_cast<int>(mapY[destIdx]);

  // Check if the mapping is within the source image bounds.
  if (srcX < srcW && srcY < srcH) {
    // Compute index into the source array (3 floats per pixel).
    int srcIdx = (srcY * srcW + srcX) * 3;
    dest[destIdx * 3 + 0] = src[srcIdx + 0];
    dest[destIdx * 3 + 1] = src[srcIdx + 1];
    dest[destIdx * 3 + 2] = src[srcIdx + 2];
  } else {
    // If out-of-range, set the destination pixel to the default color.
    dest[destIdx * 3 + 0] = defR;
    dest[destIdx * 3 + 1] = defG;
    dest[destIdx * 3 + 2] = defB;
  }
}

// ---------------------
// Batched remap kernel.
// ---------------------
// Processes each image in the batch; the mask is not used here.
__global__ void BatchedRemapKernel(
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
    float defB,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int srcImageSize = srcW * srcH * 3;
  int destImageSize = destW * destH * 3;
  int mapSize = destW * destH; // mapping arrays match destination size
  const float* srcImage = src + b * srcImageSize;
  float* destImage = dest + b * destImageSize;
  const unsigned short* mapXImage = mapX + b * mapSize;
  const unsigned short* mapYImage = mapY + b * mapSize;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destW || y >= destH)
    return;
  int destIdx = y * destW + x;
  int srcX = static_cast<int>(mapXImage[destIdx]);
  int srcY = static_cast<int>(mapYImage[destIdx]);
  if (srcX < srcW && srcY < srcH) {
    int srcIdx = (srcY * srcW + srcX) * 3;
    destImage[destIdx * 3 + 0] = srcImage[srcIdx + 0];
    destImage[destIdx * 3 + 1] = srcImage[srcIdx + 1];
    destImage[destIdx * 3 + 2] = srcImage[srcIdx + 2];
  } else {
    destImage[destIdx * 3 + 0] = defR;
    destImage[destIdx * 3 + 1] = defG;
    destImage[destIdx * 3 + 2] = defB;
  }
}

} // namespace

cudaError_t remap_kernel(
    const float* d_src,
    int srcW,
    int srcH,
    float* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    float defR,
    float defG,
    float defB) {
  // Define kernel launch configuration.
  dim3 blockDim(16, 16);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y);

  // Set default color for unmapped pixels.
  float defaultR = 100.0f, defaultG = 100.0f, defaultB = 100.0f;

  // Launch the remap kernel.
  remapKernel<<<gridDim, blockDim>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defaultR, defaultG, defaultB);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------
// Batched remap host function.
// d_src, d_dest, d_mapX, d_mapY are device pointers; batchSize determines
// how many images to process.
// ---------------------------------------------------------------------
cudaError_t batched_remap_kernel(
    const float* d_src,
    int srcW,
    int srcH,
    float* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    float defR,
    float defG,
    float defB,
    int batchSize) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernel<<<gridDim, blockDim>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB, batchSize);
  return cudaGetLastError();
}
