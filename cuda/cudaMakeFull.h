#pragma once

#include <cuda_runtime.h>

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
    cudaStream_t stream = 0);
