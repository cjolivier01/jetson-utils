#include <cuda_runtime.h>
#include <cassert>

/**
 * @brief Batched kernel to fill a float image (or a batch of images) with a constant value.
 *
 * Each image in the batch is assumed to be stored in row‐major order.
 *
 * @param dest Pointer to the destination images in device memory.
 * @param destWidth Width of each destination image.
 * @param destHeight Height of each destination image.
 * @param channels Number of channels per pixel.
 * @param value Constant value to fill.
 * @param batchSize Number of images in the batch.
 */
__global__ void fillKernelFloatBatched(
    float* dest,
    int destWidth,
    int destHeight,
    int channels,
    float value,
    int batchSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (b < batchSize && x < destWidth && y < destHeight) {
    int offset = b * (destWidth * destHeight * channels);
    int idx = (y * destWidth + x) * channels;
    for (int c = 0; c < channels; ++c) {
      dest[offset + idx + c] = value;
    }
  }
}

/**
 * @brief Batched kernel to fill an unsigned char image (mask) with a constant value.
 *
 * Each mask in the batch is assumed to be stored in row‐major order.
 *
 * @param dest Pointer to the destination masks in device memory.
 * @param destWidth Width of each destination mask.
 * @param destHeight Height of each destination mask.
 * @param channels Number of channels per pixel.
 * @param value Constant value to fill.
 * @param batchSize Number of masks in the batch.
 */
__global__ void fillKernelUCharBatched(
    unsigned char* dest,
    int destWidth,
    int destHeight,
    int channels,
    unsigned char value,
    int batchSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (b < batchSize && x < destWidth && y < destHeight) {
    int offset = b * (destWidth * destHeight * channels);
    int idx = (y * destWidth + x) * channels;
    for (int c = 0; c < channels; ++c) {
      dest[offset + idx + c] = value;
    }
  }
}

/**
 * @brief Batched kernel to copy a region of interest (ROI) from a float source image to a destination canvas.
 *
 * For each image in the batch, this kernel copies a rectangular region defined by a source ROI—starting at
 * (srcROI_x, srcROI_y) with dimensions regionWidth x regionHeight—from the source image into the destination
 * canvas at position (offsetX, offsetY).
 *
 * @param src Pointer to the batch of source images in device memory.
 * @param full_src_width Full width of each source image.
 * @param full_src_height Full height of each source image.
 * @param regionWidth Width of the ROI to copy.
 * @param regionHeight Height of the ROI to copy.
 * @param srcROI_x X-coordinate of the top-left corner of the ROI in the source images.
 * @param srcROI_y Y-coordinate of the top-left corner of the ROI in the source images.
 * @param dest Pointer to the batch of destination canvases in device memory.
 * @param destWidth Width of each destination canvas.
 * @param destHeight Height of each destination canvas.
 * @param offsetX X-coordinate in the destination canvas where the ROI is pasted.
 * @param offsetY Y-coordinate in the destination canvas where the ROI is pasted.
 * @param channels Number of channels per pixel.
 * @param batchSize Number of images in the batch.
 */
__global__ void copyRoiKernelBatched(
    const float* src,
    int full_src_width,
    int full_src_height,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    float* dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < regionWidth && y < regionHeight) {
    int srcX = srcROI_x + x;
    int srcY = srcROI_y + y;
    if (srcX < full_src_width && srcY < full_src_height) {
      int srcOffset = b * (full_src_width * full_src_height * channels);
      int srcIdx = (srcY * full_src_width + srcX) * channels;
      int destX = offsetX + x;
      int destY = offsetY + y;
      if (destX < destWidth && destY < destHeight) {
        int destOffset = b * (destWidth * destHeight * channels);
        int destIdx = (destY * destWidth + destX) * channels;
        for (int c = 0; c < channels; ++c) {
          dest[destOffset + destIdx + c] = src[srcOffset + srcIdx + c];
        }
      }
    }
  }
}

/**
 * @brief Batched kernel to copy a region of interest (ROI) from an unsigned char source mask to a destination canvas.
 *
 * For each mask in the batch, this kernel copies a rectangular region defined by a source ROI from the source mask
 * into the destination canvas at position (offsetX, offsetY).
 *
 * @param src Pointer to the batch of source masks in device memory.
 * @param full_src_width Full width of each source mask.
 * @param full_src_height Full height of each source mask.
 * @param regionWidth Width of the ROI to copy.
 * @param regionHeight Height of the ROI to copy.
 * @param srcROI_x X-coordinate of the top-left corner of the ROI in the source masks.
 * @param srcROI_y Y-coordinate of the top-left corner of the ROI in the source masks.
 * @param dest Pointer to the batch of destination canvases in device memory.
 * @param destWidth Width of each destination canvas.
 * @param destHeight Height of each destination canvas.
 * @param offsetX X-coordinate in the destination canvas where the ROI is pasted.
 * @param offsetY Y-coordinate in the destination canvas where the ROI is pasted.
 * @param channels Number of channels per pixel.
 * @param batchSize Number of masks in the batch.
 */
__global__ void copyRoiKernelUCharBatched(
    const unsigned char* src,
    int full_src_width,
    int full_src_height,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    unsigned char* dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < regionWidth && y < regionHeight) {
    int srcX = srcROI_x + x;
    int srcY = srcROI_y + y;
    if (srcX < full_src_width && srcY < full_src_height) {
      int srcOffset = b * (full_src_width * full_src_height * channels);
      int srcIdx = (srcY * full_src_width + srcX) * channels;
      int destX = offsetX + x;
      int destY = offsetY + y;
      if (destX < destWidth && destY < destHeight) {
        int destOffset = b * (destWidth * destHeight * channels);
        int destIdx = (destY * destWidth + destX) * channels;
        for (int c = 0; c < channels; ++c) {
          dest[destOffset + destIdx + c] = src[srcOffset + srcIdx + c];
        }
      }
    }
  }
}

/**
 * @brief Creates full canvas images by copying specified source ROIs from a batch of images (and optional masks)
 *        into preallocated destination canvases.
 *
 * This function takes a batch of source images (and their optional masks) in device memory along with ROI
 * specifications (source offsets and region sizes) and destination offsets. It fills the preallocated destination
 * canvases with default values (0.0f for images and 1 for masks) and copies the specified ROIs from each source image
 * (or mask) into the corresponding destination canvas at the given destination offsets.
 *
 * Optionally, if adjust_origin is true, the destination offsets (x and y) are adjusted so that one image in the batch
 * is anchored at (0,0). In that case, it is required that one of the offsets becomes 0.
 *
 * @param d_imgs Pointer to the batch of source images (float) in device memory.
 * @param src_full_width Full width of each source image.
 * @param src_full_height Full height of each source image.
 * @param region_width Width of the ROI to copy from each source image.
 * @param region_height Height of the ROI to copy from each source image.
 * @param channels Number of channels in the source images.
 * @param d_masks Pointer to the batch of source masks (unsigned char) in device memory (or nullptr if not provided).
 * @param mask_width Width of each source mask.
 * @param mask_height Height of each source mask.
 * @param mask_channels Number of channels in the source masks.
 * @param src_roi_x X-coordinate of the top-left corner of the ROI in the source images/masks.
 * @param src_roi_y Y-coordinate of the top-left corner of the ROI in the source images/masks.
 * @param x Reference to destination X-offset for the ROI in the destination canvases (may be adjusted if adjust_origin
 * is true).
 * @param y Reference to destination Y-offset for the ROI in the destination canvases (may be adjusted if adjust_origin
 * is true).
 * @param canvas_w Width of the destination canvases.
 * @param canvas_h Height of the destination canvases.
 * @param adjust_origin If true, adjusts destination offsets so that one image is anchored at (0,0).
 * @param batchSize Number of images (and masks) in the batch.
 * @param d_full_imgs Preallocated pointer to the destination canvases for images in device memory.
 *                      Expected size: batchSize x canvas_w x canvas_h x channels.
 * @param d_full_masks Preallocated pointer to the destination canvases for masks in device memory (or nullptr if not
 * provided). Expected size: batchSize x canvas_w x canvas_h x mask_channels.
 * @param stream CUDA stream to use for kernel launches (default: stream 0).
 */
void simple_make_full_batch(
    // Batch of images:
    const float* d_imgs,
    int src_full_width,
    int src_full_height,
    int region_width,
    int region_height,
    int channels,
    // Batch of masks (optional):
    const unsigned char* d_masks,
    int mask_width,
    int mask_height,
    int mask_channels,
    // Source ROI offset:
    int src_roi_x,
    int src_roi_y,
    // Destination offsets (for all images in the batch):
    int& x,
    int& y,
    // Canvas dimensions:
    int canvas_w,
    int canvas_h,
    bool adjust_origin,
    // Batch size:
    int batchSize,
    // Preallocated destination canvases:
    float* d_full_imgs,
    unsigned char* d_full_masks,
    cudaStream_t stream) {
  // Ensure the destination offsets are nonnegative.
  assert(x >= 0 && y >= 0);

  // Optionally adjust origins so that one image is anchored at (0,0).
  if (adjust_origin) {
    // Here we use the same logic as before.
    if (x <= y) {
      y -= x;
      x = 0;
    } else {
      x -= y;
      y = 0;
    }
  }
  // For now, require that either x or y is 0.
  assert(x == 0 || y == 0);

  // Define kernel launch parameters for filling the destination canvases.
  dim3 blockDim(16, 16, 1);
  dim3 gridDimCanvas((canvas_w + blockDim.x - 1) / blockDim.x, (canvas_h + blockDim.y - 1) / blockDim.y, batchSize);

  // -------------------------------------------------------
  // Fill the destination canvases with default values.
  // -------------------------------------------------------
  // For images: fill with 0.0f.
  fillKernelFloatBatched<<<gridDimCanvas, blockDim, 0, stream>>>(
      d_full_imgs, canvas_w, canvas_h, channels, 0.0f, batchSize);
  // For masks (if provided): fill with 1.
  if (d_masks && d_full_masks) {
    fillKernelUCharBatched<<<gridDimCanvas, blockDim, 0, stream>>>(
        d_full_masks, canvas_w, canvas_h, mask_channels, 1, batchSize);
  }

  // -------------------------------------------------------
  // Copy the ROI from each source image/mask into the destination canvases.
  // -------------------------------------------------------
  dim3 gridDimCopy(
      (region_width + blockDim.x - 1) / blockDim.x, (region_height + blockDim.y - 1) / blockDim.y, batchSize);

  // Copy the ROI for the images.
  copyRoiKernelBatched<<<gridDimCopy, blockDim, 0, stream>>>(
      d_imgs,
      src_full_width,
      src_full_height,
      region_width,
      region_height,
      src_roi_x,
      src_roi_y,
      d_full_imgs,
      canvas_w,
      canvas_h,
      x,
      y,
      channels,
      batchSize);

  // Copy the ROI for the masks (if provided).
  if (d_masks && d_full_masks) {
    copyRoiKernelUCharBatched<<<gridDimCopy, blockDim, 0, stream>>>(
        d_masks,
        mask_width,
        mask_height,
        region_width,
        region_height,
        src_roi_x,
        src_roi_y,
        d_full_masks,
        canvas_w,
        canvas_h,
        x,
        y,
        mask_channels,
        batchSize);
  }
}
