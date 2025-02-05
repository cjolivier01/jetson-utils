#pragma once

#include <cuda_runtime.h>

/**
 * @brief Remaps a source image to a destination image using provided mapping
 * arrays.
 *
 * For each pixel in the destination image, this kernel uses the mapping arrays
 * (`mapX` and `mapY`) to determine the corresponding pixel in the source image.
 * If the mapped coordinates fall within the valid range of the source image,
 * the source pixel (consisting of three floats for RGB) is copied to the
 * destination. Otherwise, the destination pixel is set to the specified default
 * color.
 *
 * The mapping arrays contain unsigned short values representing the x and y
 * coordinates in the source image.
 *
 * @param src   Pointer to the source image data (float array, 3 channels per
 * pixel).
 * @param srcW  Width of the source image.
 * @param srcH  Height of the source image.
 * @param dest  Pointer to the destination image data (float array, 3 channels
 * per pixel).
 * @param destW Width of the destination image.
 * @param destH Height of the destination image.
 * @param mapX  Pointer to the mapping array for the x-coordinate (one unsigned
 * short per destination pixel).
 * @param mapY  Pointer to the mapping array for the y-coordinate (one unsigned
 * short per destination pixel).
 * @param defR  Default red channel value for pixels mapped out-of-range.
 * @param defG  Default green channel value for pixels mapped out-of-range.
 * @param defB  Default blue channel value for pixels mapped out-of-range.
 */
cudaError_t remap_kernel(
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
    float defB);

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
    int batchSize);
