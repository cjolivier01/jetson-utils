#ifndef __CUDA_OVERLAY_H__
#define __CUDA_OVERLAY_H__

#include "cudaUtility.h"
#include "imageFormat.h"

//---------------------------------------------------------------------
// Original functions (unchanged)
//---------------------------------------------------------------------

/**
 * Overlay the input image onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended.
 * @ingroup overlay
 */
cudaError_t cudaOverlay(
    void* input,
    size_t inputWidth,
    size_t inputHeight,
    void* output,
    size_t outputWidth,
    size_t outputHeight,
    imageFormat format,
    int x,
    int y,
    cudaStream_t stream = 0);

/**
 * Overlay the input image composted onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended.
 * @ingroup overlay
 */
template <typename T>
cudaError_t cudaOverlay(
    T* input,
    size_t inputWidth,
    size_t inputHeight,
    T* output,
    size_t outputWidth,
    size_t outputHeight,
    int x,
    int y,
    cudaStream_t stream = 0) {
  return cudaOverlay(
      input, inputWidth, inputHeight, output, outputWidth, outputHeight, imageFormatFromType<T>(), x, y, stream);
}

/**
 * Overlay the input image composted onto the output image at location (x,y)
 * If the composted image doesn't entirely fit in the output, it will be cropped.
 * If the images have an alpha channel, they will be alpha blended.
 * @ingroup overlay
 */
template <typename T>
cudaError_t cudaOverlay(
    T* input,
    const int2& inputDims,
    T* output,
    const int2& outputDims,
    int x,
    int y,
    cudaStream_t stream = 0) {
  return cudaOverlay(
      input, inputDims.x, inputDims.y, output, outputDims.x, outputDims.y, imageFormatFromType<T>(), x, y, stream);
}

/**
 * cudaRectFill
 * @deprecated please use the functions from cudaDraw.h instead
 * @ingroup overlay
 */
// cudaError_t cudaRectFill(
//     void* input,
//     void* output,
//     size_t width,
//     size_t height,
//     imageFormat format,
//     float4* rects,
//     int numRects,
//     const float4& color,
//     cudaStream_t stream = 0);

/**
 * cudaRectFill
 * @deprecated please use the functions from cudaDraw.h instead
 * @ingroup overlay
 */
// template <typename T>
// cudaError_t cudaRectFill(
//     T* input,
//     T* output,
//     size_t width,
//     size_t height,
//     float4* rects,
//     int numRects,
//     const float4& color,
//     cudaStream_t stream = 0) {
//   return cudaRectFill(input, output, width, height, imageFormatFromType<T>(), rects, numRects, color, stream);
// }

//---------------------------------------------------------------------
// New functions with pitch support
//---------------------------------------------------------------------

/**
 * Overlay the input image onto the output image at location (x,y) using
 * pitched memory (row pitch in bytes) for both input and output.
 */
cudaError_t cudaOverlayPitch(
    void* input,
    size_t inputWidth,
    size_t inputHeight,
    size_t inputPitch,
    void* output,
    size_t outputWidth,
    size_t outputHeight,
    size_t outputPitch,
    imageFormat format,
    int x,
    int y,
    cudaStream_t stream = 0);

template <typename T>
cudaError_t cudaOverlayPitch(
    T* input,
    size_t inputWidth,
    size_t inputHeight,
    size_t inputPitch,
    T* output,
    size_t outputWidth,
    size_t outputHeight,
    size_t outputPitch,
    int x,
    int y,
    cudaStream_t stream = 0) {
  return cudaOverlayPitch(
      input,
      inputWidth,
      inputHeight,
      inputPitch,
      output,
      outputWidth,
      outputHeight,
      outputPitch,
      imageFormatFromType<T>(),
      x,
      y,
      stream);
}

template <typename T>
cudaError_t cudaOverlayPitch(
    T* input,
    const int2& inputDims,
    size_t inputPitch,
    T* output,
    const int2& outputDims,
    size_t outputPitch,
    int x,
    int y,
    cudaStream_t stream = 0) {
  return cudaOverlayPitch(
      input,
      inputDims.x,
      inputDims.y,
      inputPitch,
      output,
      outputDims.x,
      outputDims.y,
      outputPitch,
      imageFormatFromType<T>(),
      x,
      y,
      stream);
}

/**
 * Fill one or more rectangular regions of an image using pitched memory.
 */
cudaError_t cudaRectFillPitch(
    void* input,
    void* output,
    size_t width,
    size_t height,
    size_t inputPitch,
    size_t outputPitch,
    imageFormat format,
    float4* rects,
    int numRects,
    const float4& color,
    cudaStream_t stream = 0);

template <typename T>
cudaError_t cudaRectFillPitch(
    T* input,
    T* output,
    size_t width,
    size_t height,
    size_t inputPitch,
    size_t outputPitch,
    float4* rects,
    int numRects,
    const float4& color,
    cudaStream_t stream = 0) {
  return cudaRectFillPitch(
      input, output, width, height, inputPitch, outputPitch, imageFormatFromType<T>(), rects, numRects, color, stream);
}

#endif
