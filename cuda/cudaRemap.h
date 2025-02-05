#pragma once

#include <cuda_runtime.h>

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
    cudaStream_t stream = 0);

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
    cudaStream_t stream);
