#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaRemap.h"

namespace {

/**
 * @brief Remap kernel for a single image.
 *
 * Computes the destination pixel coordinates and, using mapping arrays, copies the corresponding
 * pixel from the source image. If the mapped coordinates are out-of-bounds, a default color is used.
 *
 * @param src Pointer to the source image.
 * @param srcW Width of the source image.
 * @param srcH Height of the source image.
 * @param dest Pointer to the destination image.
 * @param destW Width of the destination image.
 * @param destH Height of the destination image.
 * @param mapX Pointer to the X-coordinate mapping array.
 * @param mapY Pointer to the Y-coordinate mapping array.
 * @param defR Default red component.
 * @param defG Default green component.
 * @param defB Default blue component.
 */
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

/**
 * @brief Batched remap kernel for RGB images.
 *
 * Processes each image in the batch. For each image, a mapping array is used to determine the
 * corresponding source pixel for each destination pixel. Out-of-bounds mappings use the default color.
 *
 * @param src Pointer to the batch of source images.
 * @param srcW Width of each source image.
 * @param srcH Height of each source image.
 * @param dest Pointer to the batch of destination images.
 * @param destW Width of each destination image.
 * @param destH Height of each destination image.
 * @param mapX Pointer to the batch of X-coordinate mapping arrays.
 * @param mapY Pointer to the batch of Y-coordinate mapping arrays.
 * @param defR Default red component.
 * @param defG Default green component.
 * @param defB Default blue component.
 * @param batchSize Number of images in the batch.
 */
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

/**
 * @brief Remap a single image using mapping arrays.
 *
 * This host function launches the remapKernel on the default or specified CUDA stream.
 *
 * @param d_src Device pointer to the source image.
 * @param srcW Width of the source image.
 * @param srcH Height of the source image.
 * @param d_dest Device pointer to the destination image.
 * @param destW Width of the destination image.
 * @param destH Height of the destination image.
 * @param d_mapX Device pointer to the mapping array for X coordinates.
 * @param d_mapY Device pointer to the mapping array for Y coordinates.
 * @param defR Default red component (if mapping is out-of-range).
 * @param defG Default green component.
 * @param defB Default blue component.
 * @param stream CUDA stream to use for the kernel launch (default is 0).
 * @return cudaError_t The status returned by cudaGetLastError.
 */
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
    float defB,
    cudaStream_t stream) {
  // Define kernel launch configuration.
  dim3 blockDim(16, 16);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x,
               (destH + blockDim.y - 1) / blockDim.y);

  // Launch the remap kernel on the specified stream.
  remapKernel<<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB);
  return cudaGetLastError();
}

/**
 * @brief Batched remap host function.
 *
 * Launches the BatchedRemapKernel to process a batch of images using mapping arrays.
 *
 * @param d_src Device pointer to the batch of source images.
 * @param srcW Width of each source image.
 * @param srcH Height of each source image.
 * @param d_dest Device pointer to the batch of destination images.
 * @param destW Width of each destination image.
 * @param destH Height of each destination image.
 * @param d_mapX Device pointer to the batch of mapping arrays for X coordinates.
 * @param d_mapY Device pointer to the batch of mapping arrays for Y coordinates.
 * @param defR Default red component.
 * @param defG Default green component.
 * @param defB Default blue component.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for the kernel launch (default is 0).
 * @return cudaError_t The status returned by cudaGetLastError.
 */
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
    int batchSize,
    cudaStream_t stream = 0) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x,
               (destH + blockDim.y - 1) / blockDim.y,
               batchSize);
  BatchedRemapKernel<<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY,
      defR, defG, defB, batchSize);
  return cudaGetLastError();
}
