#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaRemap.h"  // Assumed to declare these host functions
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

//------------------------------------------------------------------------------
// Templated Remap Kernel for a Single Image
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void remapKernel(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T_out defR,
    T_out defG,
    T_out defB)
{
  // Compute destination pixel coordinates.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destW || y >= destH)
    return;

  int destIdx = y * destW + x;

  // Get mapping coordinates (stored as unsigned shorts) and cast them to int.
  int srcX = static_cast<int>(mapX[destIdx]);
  int srcY = static_cast<int>(mapY[destIdx]);

  if (srcX < srcW && srcY < srcH) {
    // Compute index into the source array (assumes 3 channels per pixel).
    int srcIdx = (srcY * srcW + srcX) * 3;
    dest[destIdx * 3 + 0] = static_cast<T_out>( src[srcIdx + 0] );
    dest[destIdx * 3 + 1] = static_cast<T_out>( src[srcIdx + 1] );
    dest[destIdx * 3 + 2] = static_cast<T_out>( src[srcIdx + 2] );
  } else {
    // Out-of-bounds: use default color.
    dest[destIdx * 3 + 0] = defR;
    dest[destIdx * 3 + 1] = defG;
    dest[destIdx * 3 + 2] = defB;
  }
}

//------------------------------------------------------------------------------
// Templated Batched Remap Kernel for RGB Images
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernel(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T_out defR,
    T_out defG,
    T_out defB,
    int batchSize)
{
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int srcImageSize  = srcW * srcH * 3;
  int destImageSize = destW * destH * 3;
  int mapSize       = destW * destH; // mapping arrays match destination size

  const T_in*  srcImage  = src  + b * srcImageSize;
  T_out*       destImage = dest + b * destImageSize;
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
    destImage[destIdx * 3 + 0] = static_cast<T_out>( srcImage[srcIdx + 0] );
    destImage[destIdx * 3 + 1] = static_cast<T_out>( srcImage[srcIdx + 1] );
    destImage[destIdx * 3 + 2] = static_cast<T_out>( srcImage[srcIdx + 2] );
  } else {
    destImage[destIdx * 3 + 0] = defR;
    destImage[destIdx * 3 + 1] = defG;
    destImage[destIdx * 3 + 2] = defB;
  }
}

template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelEx(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T_out deflt,
    int batchSize)
{
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int srcImageSize  = srcW * srcH;
  int destImageSize = destW * destH;
  int mapSize       = destW * destH; // mapping arrays match destination size

  const T_in*  srcImage  = src  + b * srcImageSize;
  T_out*       destImage = dest + b * destImageSize;
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
    int srcIdx = srcY * srcW + srcX;
    destImage[destIdx] = static_cast<T_out>( srcImage[srcIdx] );
  } else {
    destImage[destIdx] = deflt;
  }
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Host Function: Remap a Single Image
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
cudaError_t remap_kernel(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_out defR,
    T_out defG,
    T_out defB,
    cudaStream_t stream)
{
  dim3 blockDim(16, 16);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x,
               (destH + blockDim.y - 1) / blockDim.y);

  remapKernel<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// Host Function: Batched Remap
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in defR,
    T_in defG,
    T_in defB,
    int batchSize,
    cudaStream_t stream)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x,
               (destH + blockDim.y - 1) / blockDim.y,
               batchSize);

  BatchedRemapKernel<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY,
      defR, defG, defB, batchSize);
  return cudaGetLastError();
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in dflt,
    int batchSize,
    cudaStream_t stream)
{
  dim3 blockDim(10, 10, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x,
               (destH + blockDim.y - 1) / blockDim.y,
               batchSize);

  BatchedRemapKernelEx<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY,
      dflt, batchSize);
  return cudaGetLastError();
}

//
// Explicit Template Instantiations
//


template cudaError_t batched_remap_kernel<float, float>(
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

template cudaError_t batched_remap_kernel<float, __half>(
    const float* d_src,
    int srcW,
    int srcH,
    __half* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    float defR,
    float defG,
    float defB,
    int batchSize,
    cudaStream_t stream);

template cudaError_t batched_remap_kernel<__half, float>(
    const __half* d_src,
    int srcW,
    int srcH,
    float* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    __half defR,
    __half defG,
    __half defB,
    int batchSize,
    cudaStream_t stream);

template cudaError_t batched_remap_kernel<__half, __half>(
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
