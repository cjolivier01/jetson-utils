#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaRemap.h"

// If you wish to support 16‐bit and bfloat16 types:
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

//--------------------------------------------------------------------------
// Templated remap kernel for a single image.
//--------------------------------------------------------------------------
template <typename T>
__global__ void remapKernel(
    const T* src,
    int srcW,
    int srcH,
    T* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T defR,
    T defG,
    T defB) {
  // Compute destination pixel coordinates.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destW || y >= destH)
    return;

  // Compute the linear index for the destination pixel.
  int destIdx = y * destW + x;

  // Retrieve mapping coordinates (stored as unsigned short) and cast to int.
  int srcX = static_cast<int>(mapX[destIdx]);
  int srcY = static_cast<int>(mapY[destIdx]);

  // Check bounds: if the mapped coordinates are in range, copy the pixel.
  if (srcX < srcW && srcY < srcH) {
    int srcIdx = (srcY * srcW + srcX) * 3;
    dest[destIdx * 3 + 0] = src[srcIdx + 0];
    dest[destIdx * 3 + 1] = src[srcIdx + 1];
    dest[destIdx * 3 + 2] = src[srcIdx + 2];
  } else {
    // Out-of-range: use the default color.
    dest[destIdx * 3 + 0] = defR;
    dest[destIdx * 3 + 1] = defG;
    dest[destIdx * 3 + 2] = defB;
  }
}

//--------------------------------------------------------------------------
// Templated batched remap kernel for RGB images.
//--------------------------------------------------------------------------
template <typename T>
__global__ void BatchedRemapKernel(
    const T* src,
    int srcW,
    int srcH,
    T* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T defR,
    T defG,
    T defB,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int srcImageSize = srcW * srcH * 3;
  int destImageSize = destW * destH * 3;
  int mapSize = destW * destH; // mapping arrays match destination size

  // Get pointers to the b-th source image, destination image, and mapping arrays.
  const T* srcImage = src + b * srcImageSize;
  T* destImage = dest + b * destImageSize;
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

//--------------------------------------------------------------------------
// Templated host functions
//--------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Single-image remap host function.
//------------------------------------------------------------------------------
template <typename T>
cudaError_t remap_kernel(
    const T* d_src,
    int srcW,
    int srcH,
    T* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T defR,
    T defG,
    T defB,
    cudaStream_t stream) {
  // Define kernel launch configuration.
  dim3 blockDim(16, 16);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y);

  // Launch the remap kernel.
  remapKernel<T>
      <<<gridDim, blockDim, 0, stream>>>(d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// Batched remap host function.
//------------------------------------------------------------------------------
template <typename T>
cudaError_t batched_remap_kernel(
    const T* d_src,
    int srcW,
    int srcH,
    T* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T defR,
    T defG,
    T defB,
    int batchSize,
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y, batchSize);

  // Launch the batched remap kernel.
  BatchedRemapKernel<T><<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB, batchSize);
  return cudaGetLastError();
}

//
// Explicit instantiation declarations for T = float, __half, and __nv_bfloat16.
//

template cudaError_t remap_kernel<float>(
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
    cudaStream_t stream);

template cudaError_t remap_kernel<__half>(
    const __half* d_src,
    int srcW,
    int srcH,
    __half* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    __half defR,
    __half defG,
    __half defB,
    cudaStream_t stream);

template cudaError_t remap_kernel<__nv_bfloat16>(
    const __nv_bfloat16* d_src,
    int srcW,
    int srcH,
    __nv_bfloat16* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    __nv_bfloat16 defR,
    __nv_bfloat16 defG,
    __nv_bfloat16 defB,
    cudaStream_t stream);

template cudaError_t batched_remap_kernel<float>(
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
    cudaStream_t stream);

template cudaError_t batched_remap_kernel<__half>(
    const __half* d_src,
    int srcW,
    int srcH,
    __half* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    __half defR,
    __half defG,
    __half defB,
    int batchSize,
    cudaStream_t stream);

template cudaError_t batched_remap_kernel<__nv_bfloat16>(
    const __nv_bfloat16* d_src,
    int srcW,
    int srcH,
    __nv_bfloat16* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    __nv_bfloat16 defR,
    __nv_bfloat16 defG,
    __nv_bfloat16 defB,
    int batchSize,
    cudaStream_t stream);
