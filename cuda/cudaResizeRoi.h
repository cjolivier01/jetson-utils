#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

// Include definitions for cudaFilterMode and imageFormat.
// Adjust these include paths based on your project structure.
#include "cudaFilterMode.cuh"
#include "imageFormat.h"

/* clang-format off */
//------------------------------------------------------------------------------
// ROI–based resize functions for various image data types.
//
// The following functions resize a region of interest (ROI) from the source image
// into a region of interest in the destination image. The ROI is defined by its
// x/y origin and width/height in both the source and destination images.
//
// Each function returns a cudaError_t status and uses the specified cudaStream_t.


// cudaResizeROI for 8-bit grayscale images.
cudaError_t cudaResizeROI( uint8_t* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uint8_t* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// cudaResizeROI for 32-bit floating point grayscale images.
cudaError_t cudaResizeROI( float* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// cudaResizeROI for uchar3 images (e.g. RGB8 or BGR8).
cudaError_t cudaResizeROI( uchar3* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uchar3* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// cudaResizeROI for uchar4 images (e.g. RGBA8 or BGRA8).
cudaError_t cudaResizeROI( uchar4* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uchar4* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// cudaResizeROI for float3 images (e.g. RGB32F or BGR32F).
cudaError_t cudaResizeROI( float3* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float3* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// cudaResizeROI for float4 images (e.g. RGBA32F or BGRA32F).
cudaError_t cudaResizeROI( float4* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float4* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream );

// Generic cudaResizeROI function that selects the proper overload based on imageFormat.
cudaError_t cudaResizeROI( void* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           void* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           imageFormat format, cudaFilterMode filter, cudaStream_t stream );

/* clang-format on */
