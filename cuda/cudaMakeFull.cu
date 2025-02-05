#include "cudaMakeFull.h"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

//--------------------------------------------------------
// Simple CUDA kernels for filling a canvas and copying an ROI
//--------------------------------------------------------

// TODO: faster to use float3, char1, etc instead of channel loops

// Kernel to fill a float image with a constant value.
__global__ void fillKernelFloat(float* dest, int destWidth, int destHeight, int channels, float value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < destWidth && y < destHeight) {
    int idx = (y * destWidth + x) * channels;
    for (int c = 0; c < channels; ++c) {
      dest[idx + c] = value;
    }
  }
}

// Kernel to fill an unsigned char image (mask) with a constant value.
__global__ void fillKernelUChar(unsigned char* dest, int destWidth, int destHeight, int channels, unsigned char value) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < destWidth && y < destHeight) {
    int idx = (y * destWidth + x) * channels;
    for (int c = 0; c < channels; ++c) {
      dest[idx + c] = value;
    }
  }
}

// Kernel to copy an ROI from a source float image into a destination image at a given offset.
__global__ void copyRoiKernel(
    const float* src,
    int srcWidth,
    int srcHeight,
    float* dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < srcWidth && y < srcHeight) {
    int srcIdx = (y * srcWidth + x) * channels;
    int destX = offsetX + x;
    int destY = offsetY + y;
    if (destX < destWidth && destY < destHeight) {
      int destIdx = (destY * destWidth + destX) * channels;
      for (int c = 0; c < channels; ++c) {
        dest[destIdx + c] = src[srcIdx + c];
      }
    }
  }
}

// Kernel to copy an ROI from a source mask (unsigned char) into a destination mask.
__global__ void copyRoiKernelUChar(
    const unsigned char* src,
    int srcWidth,
    int srcHeight,
    unsigned char* dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < srcWidth && y < srcHeight) {
    int srcIdx = (y * srcWidth + x) * channels;
    int destX = offsetX + x;
    int destY = offsetY + y;
    if (destX < destWidth && destY < destHeight) {
      int destIdx = (destY * destWidth + destX) * channels;
      for (int c = 0; c < channels; ++c) {
        dest[destIdx + c] = src[srcIdx + c];
      }
    }
  }
}

//--------------------------------------------------------
// The simple_make_full function.
// It “pads” two images (and optionally two masks) into canvases of size canvas_w x canvas_h.
// The images are placed at offsets (x1, y1) and (x2, y2) respectively.
// If adjust_origin is true, the offsets are modified so that one of the images starts at (0,0).
// (For now we require that one of the x and one of the y offsets end up being zero.)
//
// Note: The source images and masks reside in device memory (d_img_1, d_mask_1, etc.).
//       Their dimensions (width, height, channels) must be supplied.
//       The outputs (full_img_*, full_mask_*) are newly allocated device memory.
SimpleFullResult simple_make_full(
    // Image 1 (float image)
    const float* d_img_1,
    int srcOffsetX1,
    int srcOffsetY1,
    int img1_width,
    int img1_height,
    int img1_channels,
    // Optional mask 1 (1-channel unsigned char; pass nullptr if not provided)
    const unsigned char* d_mask_1,
    int mask1_width,
    int mask1_height,
    int mask1_channels,
    // Offsets for image 1
    int& x1,
    int& y1,
    // Image 2 (float image)
    const float* d_img_2,
    int srcOffsetX2,
    int srcOffsetY2,
    int img2_width,
    int img2_height,
    int img2_channels,
    // Optional mask 2
    const unsigned char* d_mask_2,
    int mask2_width,
    int mask2_height,
    int mask2_channels,
    // Offsets for image 2
    int& x2,
    int& y2,
    // Canvas dimensions
    int canvas_w,
    int canvas_h,
    float* d_full_img_1,
    unsigned char* d_full_mask_1,
    float* d_full_img_2,
    unsigned char* d_full_mask_2,
    // If true, adjust the origins so that one image is anchored at (0,0)
    bool adjust_origin,
    // Optional CUDA stream (default stream if not provided)
    cudaStream_t stream) {
  // Retrieve source image dimensions (as in the Python version)
  // (img1_width, img1_height) and (img2_width, img2_height) are assumed to be valid.
  assert(x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0);

  // If requested, adjust the origins.
  if (adjust_origin) {
    if (y1 <= y2) {
      y2 -= y1;
      y1 = 0;
    } else { // y2 < y1
      y1 -= y2;
      y2 = 0;
    }
    if (x1 <= x2) {
      x2 -= x1;
      x1 = 0;
    } else { // x2 < x1
      x1 -= x2;
      x2 = 0;
    }
  } else {
    assert(x1 == 0 || x2 == 0);
    assert(y1 == 0 || y2 == 0);
  }

  // For now, require that one of x1 or x2 is 0 and one of y1 or y2 is 0.
  assert(x1 == 0 || x2 == 0);
  assert(y1 == 0 || y2 == 0);

  // Set up CUDA kernel launch parameters.
  dim3 blockDim(16, 16);
  dim3 gridDimCanvas((canvas_w + blockDim.x - 1) / blockDim.x, (canvas_h + blockDim.y - 1) / blockDim.y);

  // ---------------------------
  // Process image 1
  // ---------------------------
  // Fill the entire canvas with 0.0f.
  fillKernelFloat<<<gridDimCanvas, blockDim, 0, stream>>>(d_full_img_1, canvas_w, canvas_h, img1_channels, 0.0f);

  // Copy the source image into the canvas at offset (x1, y1).
  dim3 gridDimCopy1((img1_width + blockDim.x - 1) / blockDim.x, (img1_height + blockDim.y - 1) / blockDim.y);
  copyRoiKernel<<<gridDimCopy1, blockDim, 0, stream>>>(
      d_img_1, img1_width, img1_height, d_full_img_1, canvas_w, canvas_h, x1, y1, img1_channels);

  // Process mask 1 (if provided)
  if (d_mask_1 != nullptr) {
    // For masks, pad with constant True (represented as 1).
    fillKernelUChar<<<gridDimCanvas, blockDim, 0, stream>>>(d_full_mask_1, canvas_w, canvas_h, mask1_channels, 1);
    // Copy the source mask into the canvas at offset (x1, y1).
    dim3 gridDimCopyMask1((mask1_width + blockDim.x - 1) / blockDim.x, (mask1_height + blockDim.y - 1) / blockDim.y);
    copyRoiKernelUChar<<<gridDimCopyMask1, blockDim, 0, stream>>>(
        d_mask_1, mask1_width, mask1_height, d_full_mask_1, canvas_w, canvas_h, x1, y1, mask1_channels);
  }

  // ---------------------------
  // Process image 2
  // ---------------------------
  fillKernelFloat<<<gridDimCanvas, blockDim, 0, stream>>>(d_full_img_2, canvas_w, canvas_h, img2_channels, 0.0f);
  dim3 gridDimCopy2((img2_width + blockDim.x - 1) / blockDim.x, (img2_height + blockDim.y - 1) / blockDim.y);
  copyRoiKernel<<<gridDimCopy2, blockDim, 0, stream>>>(
      d_img_2, img2_width, img2_height, d_full_img_2, canvas_w, canvas_h, x2, y2, img2_channels);

  // Process mask 2 (if provided)
  if (d_mask_2 != nullptr) {
    fillKernelUChar<<<gridDimCanvas, blockDim, 0, stream>>>(d_full_mask_2, canvas_w, canvas_h, mask2_channels, 1);
    dim3 gridDimCopyMask2((mask2_width + blockDim.x - 1) / blockDim.x, (mask2_height + blockDim.y - 1) / blockDim.y);
    copyRoiKernelUChar<<<gridDimCopyMask2, blockDim, 0, stream>>>(
        d_mask_2, mask2_width, mask2_height, d_full_mask_2, canvas_w, canvas_h, x2, y2, mask2_channels);
  }

  // Package the results into a struct.
  SimpleFullResult result;
  result.full_img_1 = d_full_img_1;
  result.full_mask_1 = d_full_mask_1;
  result.full_img_2 = d_full_img_2;
  result.full_mask_2 = d_full_mask_2;
  return result;
}

//--------------------------------------------------------
// Example (pseudo-)usage
//--------------------------------------------------------
/*
int main() {
    // Suppose you have already allocated and populated the following device arrays:
    // d_img_1, d_img_2: float images; d_mask_1, d_mask_2: optional unsigned char masks.
    // Their dimensions (width, height, channels) are known.
    // For demonstration, we omit the allocation and initialization code.

    // Example parameters for image1 and image2.
    int img1_width = 400, img1_height = 300, img1_channels = 3;
    int img2_width = 350, img2_height = 250, img2_channels = 3;
    // For masks we assume 1 channel.
    int mask1_width = img1_width, mask1_height = img1_height, mask1_channels = 1;
    int mask2_width = img2_width, mask2_height = img2_height, mask2_channels = 1;

    // Offsets for image1 and image2.
    int x1 = 20, y1 = 10;
    int x2 = 150, y2 = 5;

    // Canvas size.
    int canvas_w = 800, canvas_h = 600;

    // Whether to adjust origins.
    bool adjust_origin = true;

    // Device pointers (assumed allocated and filled).
    float* d_img_1 = /* ... * / nullptr;
    float* d_img_2 = /* ... * / nullptr;
    unsigned char* d_mask_1 = /* ... * / nullptr; // or nullptr if no mask
    unsigned char* d_mask_2 = /* ... * / nullptr; // or nullptr if no mask

    // Optionally, create a CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Call our function.
    SimpleFullResult full = simple_make_full(
        d_img_1, img1_width, img1_height, img1_channels,
        d_mask_1, mask1_width, mask1_height, mask1_channels,
        x1, y1,
        d_img_2, img2_width, img2_height, img2_channels,
        d_mask_2, mask2_width, mask2_height, mask2_channels,
        x2, y2,
        canvas_w, canvas_h,
        adjust_origin,
        stream
    );

    // (Now full.full_img_1, full.full_mask_1, etc., are valid device pointers for the padded images.)
    // Remember to synchronize and free device memory when done.
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Free the full images when finished:
    cudaFree(full.full_img_1);
    if (full.full_mask_1) cudaFree(full.full_mask_1);
    cudaFree(full.full_img_2);
    if (full.full_mask_2) cudaFree(full.full_mask_2);

    return 0;
}
*/
//--------------------------------------------------------
