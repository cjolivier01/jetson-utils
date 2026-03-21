#include "cudaAlphaBlend.cuh"
#include "cudaOverlay.h"

//---------------------------------------------------------------------
// New Overlay Kernels with Pitch
//---------------------------------------------------------------------
template <typename T>
__global__ void gpuOverlayPitch(T *input, int inputWidth, int inputHeight,
                                size_t inputPitch, T *output, int outputWidth,
                                int outputHeight, size_t outputPitch, int x0,
                                int y0) {
  const int input_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int input_y = blockIdx.y * blockDim.y + threadIdx.y;

  const int x = input_x + x0;
  const int y = input_y + y0;

  if (input_x >= inputWidth || input_y >= inputHeight || x >= outputWidth ||
      y >= outputHeight)
    return;

  // Compute row pointers using pitch (in bytes)
  T *inputRow = (T *)((char *)input + input_y * inputPitch);
  T *outputRow = (T *)((char *)output + y * outputPitch);
  outputRow[x] = inputRow[input_x];
}

template <typename T>
__global__ void gpuOverlayAlphaPitch(T *input, int inputWidth, int inputHeight,
                                     size_t inputPitch, T *output,
                                     int outputWidth, int outputHeight,
                                     size_t outputPitch, int x0, int y0) {
  const int input_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int input_y = blockIdx.y * blockDim.y + threadIdx.y;

  const int x = input_x + x0;
  const int y = input_y + y0;

  if (input_x >= inputWidth || input_y >= inputHeight || x >= outputWidth ||
      y >= outputHeight)
    return;

  T *inputRow = (T *)((char *)input + input_y * inputPitch);
  T *outputRow = (T *)((char *)output + y * outputPitch);
  outputRow[x] = cudaAlphaBlend(outputRow[x], inputRow[input_x]);
}

//---------------------------------------------------------------------
// Base wrapper to provide the legacy no-pitch entry point.
//---------------------------------------------------------------------
cudaError_t cudaOverlay(void *input, size_t inputWidth, size_t inputHeight,
                        void *output, size_t outputWidth, size_t outputHeight,
                        imageFormat format, int x, int y, cudaStream_t stream) {
  if (!input || !output)
    return cudaErrorInvalidValue;

  const size_t inputPitch = imageFormatSize(format, inputWidth, 1);
  const size_t outputPitch = imageFormatSize(format, outputWidth, 1);

  return cudaOverlayPitch(input, inputWidth, inputHeight, inputPitch, output,
                          outputWidth, outputHeight, outputPitch, format, x, y,
                          stream);
}

//---------------------------------------------------------------------
// New Host Function: cudaOverlayPitch
//---------------------------------------------------------------------
cudaError_t cudaOverlayPitch(void *input, size_t inputWidth, size_t inputHeight,
                             size_t inputPitch, void *output,
                             size_t outputWidth, size_t outputHeight,
                             size_t outputPitch, imageFormat format, int x,
                             int y, cudaStream_t stream) {
  if (!input || !output || inputWidth == 0 || inputHeight == 0 ||
      outputWidth == 0 || outputHeight == 0)
    return cudaErrorInvalidValue;

  if (x < 0 || y < 0 || x >= outputWidth || y >= outputHeight)
    return cudaErrorInvalidValue;

  if (!imageFormatIsRGB(format) && !imageFormatIsBGR(format) &&
      !imageFormatIsGray(format))
    return cudaErrorInvalidValue;

  int overlayWidth = inputWidth;
  int overlayHeight = inputHeight;

  if (x + overlayWidth >= outputWidth)
    overlayWidth = outputWidth - x;

  if (y + overlayHeight >= outputHeight)
    overlayHeight = outputHeight - y;

  const dim3 blockDim(8, 8);
  const dim3 gridDim(iDivUp(overlayWidth, blockDim.x),
                     iDivUp(overlayHeight, blockDim.y));

#define launch_overlay_pitch(kernel, type)                                     \
  kernel<type><<<gridDim, blockDim, 0, stream>>>(                              \
      (type *)input, inputWidth, inputHeight, inputPitch, (type *)output,      \
      outputWidth, outputHeight, outputPitch, x, y)

  if (format == IMAGE_RGB8 || format == IMAGE_BGR8)
    launch_overlay_pitch(gpuOverlayPitch, uchar3);
  else if (format == IMAGE_RGBA8 || format == IMAGE_BGRA8)
    launch_overlay_pitch(gpuOverlayAlphaPitch, uchar4);
  else if (format == IMAGE_RGB32F || format == IMAGE_BGR32F)
    launch_overlay_pitch(gpuOverlayPitch, float3);
  else if (format == IMAGE_RGBA32F || format == IMAGE_BGRA32F)
    launch_overlay_pitch(gpuOverlayAlphaPitch, float4);
  else if (format == IMAGE_GRAY8)
    launch_overlay_pitch(gpuOverlayPitch, uint8_t);
  else if (format == IMAGE_GRAY32F)
    launch_overlay_pitch(gpuOverlayPitch, float);

  return CUDA(cudaGetLastError());
}

template <typename T>
__global__ void gpuRectFill(T *input, T *output, int width, int height,
                            float4 *rects, int numRects, float4 color) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  T px = input[y * width + x];

  const float fx = x;
  const float fy = y;

  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;

  for (int nr = 0; nr < numRects; nr++) {
    const float4 r = rects[nr];

    if (fy >= r.y && fy <= r.w && fx >= r.x && fx <= r.z) {
      px.x = alpha * color.x + ialph * px.x;
      px.y = alpha * color.y + ialph * px.y;
      px.z = alpha * color.z + ialph * px.z;
    }
  }

  output[y * width + x] = px;
}

template <typename T>
__global__ void gpuRectFillBox(T *input, T *output, int imgWidth, int imgHeight,
                               int x0, int y0, int boxWidth, int boxHeight,
                               const float4 color) {
  const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (box_x >= boxWidth || box_y >= boxHeight)
    return;

  const int x = box_x + x0;
  const int y = box_y + y0;

  if (x >= imgWidth || y >= imgHeight)
    return;

  T px = input[y * imgWidth + x];

  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;

  px.x = alpha * color.x + ialph * px.x;
  px.y = alpha * color.y + ialph * px.y;
  px.z = alpha * color.z + ialph * px.z;

  output[y * imgWidth + x] = px;
}

template <typename T>
cudaError_t launchRectFill(T *input, T *output, size_t width, size_t height,
                           float4 *rects, int numRects, const float4 &color,
                           cudaStream_t stream) {
  // if input and output are the same image, then we can use the faster method
  // which draws 1 box per kernel, but doesn't copy pixels that aren't inside
  // boxes
  if (input == output) {
    for (int n = 0; n < numRects; n++) {
      const int boxWidth = (int)(rects[n].z - rects[n].x);
      const int boxHeight = (int)(rects[n].w - rects[n].y);

      // launch kernel
      const dim3 blockDim(8, 8);
      const dim3 gridDim(iDivUp(boxWidth, blockDim.x),
                         iDivUp(boxHeight, blockDim.y));

      gpuRectFillBox<T><<<gridDim, blockDim, 0, stream>>>(
          input, output, width, height, (int)rects[n].x, (int)rects[n].y,
          boxWidth, boxHeight, color);
    }
  } else {
    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));

    gpuRectFill<T><<<gridDim, blockDim, 0, stream>>>(
        input, output, width, height, rects, numRects, color);
  }

  return cudaGetLastError();
}

// cudaRectFill
cudaError_t cudaRectFill(void *input, void *output, size_t width, size_t height,
                         imageFormat format, float4 *rects, int numRects,
                         const float4 &color, cudaStream_t stream) {
  if (!input || !output || width == 0 || height == 0 || !rects || numRects == 0)
    return cudaErrorInvalidValue;

  if (format == IMAGE_RGB8)
    return launchRectFill<uchar3>((uchar3 *)input, (uchar3 *)output, width,
                                  height, rects, numRects, color, stream);
  else if (format == IMAGE_RGBA8)
    return launchRectFill<uchar4>((uchar4 *)input, (uchar4 *)output, width,
                                  height, rects, numRects, color, stream);
  else if (format == IMAGE_RGB32F)
    return launchRectFill<float3>((float3 *)input, (float3 *)output, width,
                                  height, rects, numRects, color, stream);
  else if (format == IMAGE_RGBA32F)
    return launchRectFill<float4>((float4 *)input, (float4 *)output, width,
                                  height, rects, numRects, color, stream);
  else
    return cudaErrorInvalidValue;
}

//---------------------------------------------------------------------
// New Rect Fill Kernels with Pitch
//---------------------------------------------------------------------
template <typename T>
__global__ void gpuRectFillPitch(T *input, T *output, int width, int height,
                                 size_t inputPitch, size_t outputPitch,
                                 float4 *rects, int numRects, float4 color) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  T *inputRow = (T *)((char *)input + y * inputPitch);
  T *outputRow = (T *)((char *)output + y * outputPitch);
  T px = inputRow[x];

  const float fx = x;
  const float fy = y;

  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;

  for (int nr = 0; nr < numRects; nr++) {
    const float4 r = rects[nr];
    if (fy >= r.y && fy <= r.w && fx >= r.x && fx <= r.z) {
      px.x = alpha * color.x + ialph * px.x;
      px.y = alpha * color.y + ialph * px.y;
      px.z = alpha * color.z + ialph * px.z;
    }
  }

  outputRow[x] = px;
}

template <typename T>
__global__ void
gpuRectFillBoxPitch(T *input, T *output, int imgWidth, int imgHeight,
                    size_t inputPitch, size_t outputPitch, int x0, int y0,
                    int boxWidth, int boxHeight, const float4 color) {
  const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (box_x >= boxWidth || box_y >= boxHeight)
    return;

  const int x = box_x + x0;
  const int y = box_y + y0;

  if (x >= imgWidth || y >= imgHeight)
    return;

  T *inputRow = (T *)((char *)input + y * inputPitch);
  T *outputRow = (T *)((char *)output + y * outputPitch);
  T px = inputRow[x];

  const float alpha = color.w / 255.0f;
  const float ialph = 1.0f - alpha;

  px.x = alpha * color.x + ialph * px.x;
  px.y = alpha * color.y + ialph * px.y;
  px.z = alpha * color.z + ialph * px.z;

  outputRow[x] = px;
}

//---------------------------------------------------------------------
// New Launch Function for Rect Fill with Pitch
//---------------------------------------------------------------------
template <typename T>
cudaError_t launchRectFillPitch(T *input, T *output, size_t width,
                                size_t height, size_t inputPitch,
                                size_t outputPitch, float4 *rects, int numRects,
                                const float4 &color, cudaStream_t stream) {
  // if input and output are the same image, then draw one box per kernel
  if (input == output) {
    for (int n = 0; n < numRects; n++) {
      const int boxWidth = (int)(rects[n].z - rects[n].x);
      const int boxHeight = (int)(rects[n].w - rects[n].y);

      const dim3 blockDim(8, 8);
      const dim3 gridDim(iDivUp(boxWidth, blockDim.x),
                         iDivUp(boxHeight, blockDim.y));

      gpuRectFillBoxPitch<T><<<gridDim, blockDim, 0, stream>>>(
          input, output, width, height, inputPitch, outputPitch,
          (int)rects[n].x, (int)rects[n].y, boxWidth, boxHeight, color);
    }
  } else {
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));

    gpuRectFillPitch<T><<<gridDim, blockDim, 0, stream>>>(
        input, output, width, height, inputPitch, outputPitch, rects, numRects,
        color);
  }

  return cudaGetLastError();
}

//---------------------------------------------------------------------
// New Host Function: cudaRectFillPitch
//---------------------------------------------------------------------
cudaError_t cudaRectFillPitch(void *input, void *output, size_t width,
                              size_t height, size_t inputPitch,
                              size_t outputPitch, imageFormat format,
                              float4 *rects, int numRects, const float4 &color,
                              cudaStream_t stream) {
  if (!input || !output || width == 0 || height == 0 || !rects || numRects == 0)
    return cudaErrorInvalidValue;

  if (format == IMAGE_RGB8)
    return launchRectFillPitch<uchar3>((uchar3 *)input, (uchar3 *)output, width,
                                       height, inputPitch, outputPitch, rects,
                                       numRects, color, stream);
  else if (format == IMAGE_RGBA8)
    return launchRectFillPitch<uchar4>((uchar4 *)input, (uchar4 *)output, width,
                                       height, inputPitch, outputPitch, rects,
                                       numRects, color, stream);
  else if (format == IMAGE_RGB32F)
    return launchRectFillPitch<float3>((float3 *)input, (float3 *)output, width,
                                       height, inputPitch, outputPitch, rects,
                                       numRects, color, stream);
  else if (format == IMAGE_RGBA32F)
    return launchRectFillPitch<float4>((float4 *)input, (float4 *)output, width,
                                       height, inputPitch, outputPitch, rects,
                                       numRects, color, stream);
  else
    return cudaErrorInvalidValue;
}
