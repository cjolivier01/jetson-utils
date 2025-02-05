#pragma once

#include <cuda_runtime.h>

/**
 * @brief Interface function for launching the batched ROI copy kernel for float images.
 *
 * This function sets up the grid and block dimensions and launches the copyRoiKernelBatched kernel,
 * which copies a rectangular region (ROI) from each source image in a batch into its corresponding
 * destination image.
 *
 * @param d_src Pointer to the batch of source images in device memory.
 * @param full_src_width Full width of each source image.
 * @param full_src_height Full height of each source image.
 * @param regionWidth Width of the ROI to copy.
 * @param regionHeight Height of the ROI to copy.
 * @param srcROI_x X-coordinate of the top-left corner of the ROI in the source images.
 * @param srcROI_y Y-coordinate of the top-left corner of the ROI in the source images.
 * @param d_dest Pointer to the batch of destination images in device memory.
 * @param destWidth Width of each destination image.
 * @param destHeight Height of each destination image.
 * @param offsetX X-coordinate in the destination image where the ROI should be pasted.
 * @param offsetY Y-coordinate in the destination image where the ROI should be pasted.
 * @param channels Number of channels per pixel.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for kernel launch (default is stream 0).
 *
 * @return cudaError_t The status of the kernel launch.
 */
template <typename T>
cudaError_t copyRoiBatchedInterface(
    const T* d_src,
    int full_src_width,
    int full_src_height,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    T* d_dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels,
    int batchSize,
    cudaStream_t stream = 0);

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
template <typename T, typename U>
cudaError_t simple_make_full_batch(
    // Batch of images:
    const T* d_imgs,
    int src_full_width,
    int src_full_height,
    int region_width,
    int region_height,
    int channels,
    // Batch of masks (optional):
    const U* d_masks,
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
    T* d_full_imgs,
    U* d_full_masks,
    cudaStream_t stream = 0);
