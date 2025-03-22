#include "cudaResize.h"
#include "cudaFilterMode.cuh"
/* clang-format off */
namespace {
//-----------------------------------------------------------------------------
// New device function for filtering a pixel within a source ROI
template<typename T, cudaFilterMode filter>
__device__ T cudaFilterPixelROI( T* input, float x, float y, int width, int height )
{
    // For nearest-neighbor filtering.
    #if filter == FILTER_POINT
        int ix = __float2int_rn(x);
        int iy = __float2int_rn(y);
        ix = min(max(ix, 0), width - 1);
        iy = min(max(iy, 0), height - 1);
        return input[iy * width + ix];
    // For bilinear (linear) filtering.
    #elif filter == FILTER_LINEAR
        int ix = floorf(x);
        int iy = floorf(y);
        float dx = x - ix;
        float dy = y - iy;
        int ix1 = min(ix + 1, width - 1);
        int iy1 = min(iy + 1, height - 1);
        T p00 = input[iy * width + ix];
        T p10 = input[iy * width + ix1];
        T p01 = input[iy1 * width + ix];
        T p11 = input[iy1 * width + ix1];
        return (T)(((1.0f - dx) * (1.0f - dy) * p00) +
                   (dx * (1.0f - dy) * p10) +
                   ((1.0f - dx) * dy * p01) +
                   (dx * dy * p11));
    #endif
}

//-----------------------------------------------------------------------------
// New ROI resize kernel
template<typename T, cudaFilterMode filter>
__global__ void gpuResizeROI( T* input, int inputWidth, int inputHeight, 
                              int srcX, int srcY, int srcWidth, int srcHeight,
                              T* output, int outputWidth, int outputHeight, 
                              int dstX, int dstY, int dstWidth, int dstHeight )
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if( dx >= dstWidth || dy >= dstHeight )
         return;

    // Compute the absolute destination coordinates.
    int out_x = dstX + dx;
    int out_y = dstY + dy;

    // Compute scaling factors from destination ROI to source ROI.
    float scaleX = (float)srcWidth / (float)dstWidth;
    float scaleY = (float)srcHeight / (float)dstHeight;

    // Optionally use center-aligned mapping.
    float src_coord_x = srcX + (dx + 0.5f) * scaleX - 0.5f;
    float src_coord_y = srcY + (dy + 0.5f) * scaleY - 0.5f;

    T pixel = cudaFilterPixelROI<T, filter>(input, src_coord_x, src_coord_y, inputWidth, inputHeight);
    output[out_y * outputWidth + out_x] = pixel;
}

//-----------------------------------------------------------------------------
// New launch helper for ROI resize
template<typename T>
static cudaError_t launchResizeROI( T* input, size_t inputWidth, size_t inputHeight,
                                    int srcX, int srcY, int srcWidth, int srcHeight,
                                    T* output, size_t outputWidth, size_t outputHeight,
                                    int dstX, int dstY, int dstWidth, int dstHeight,
                                    cudaFilterMode filter, cudaStream_t stream )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || inputHeight == 0 || outputWidth == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    if( srcWidth == 0 || srcHeight == 0 || dstWidth == 0 || dstHeight == 0 )
        return cudaErrorInvalidValue;

    // Validate that the source ROI is within the input image...
    if( srcX < 0 || srcY < 0 || (srcX + srcWidth) > (int)inputWidth || (srcY + srcHeight) > (int)inputHeight )
        return cudaErrorInvalidValue;

    // ...and that the destination ROI is within the output image.
    if( dstX < 0 || dstY < 0 || (dstX + dstWidth) > (int)outputWidth || (dstY + dstHeight) > (int)outputHeight )
        return cudaErrorInvalidValue;

    // Optionally force point filtering if downsizing significantly.
    if( dstWidth < srcWidth && dstHeight < srcHeight )
        filter = FILTER_POINT;

    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(dstWidth, blockDim.x), iDivUp(dstHeight, blockDim.y));

    #define launch_resize_roi(filterMode) \
        gpuResizeROI<T, filterMode><<<gridDim, blockDim, 0, stream>>>(input, inputWidth, inputHeight, \
                                                                       srcX, srcY, srcWidth, srcHeight, \
                                                                       output, outputWidth, outputHeight, \
                                                                       dstX, dstY, dstWidth, dstHeight)

    if( filter == FILTER_POINT )
        launch_resize_roi(FILTER_POINT);
    else if( filter == FILTER_LINEAR )
        launch_resize_roi(FILTER_LINEAR);

    return CUDA(cudaGetLastError());
}
}

//-----------------------------------------------------------------------------
// New cudaResizeROI functions for various types

// cudaResizeROI (uint8 grayscale)
cudaError_t cudaResizeROI( uint8_t* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uint8_t* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<uint8_t>(input, inputWidth, inputHeight,
                                    srcX, srcY, srcWidth, srcHeight,
                                    output, outputWidth, outputHeight,
                                    dstX, dstY, dstWidth, dstHeight,
                                    filter, stream);
}

// cudaResizeROI (float grayscale)
cudaError_t cudaResizeROI( float* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<float>(input, inputWidth, inputHeight,
                                  srcX, srcY, srcWidth, srcHeight,
                                  output, outputWidth, outputHeight,
                                  dstX, dstY, dstWidth, dstHeight,
                                  filter, stream);
}

// cudaResizeROI (uchar3)
cudaError_t cudaResizeROI( uchar3* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uchar3* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<uchar3>(input, inputWidth, inputHeight,
                                   srcX, srcY, srcWidth, srcHeight,
                                   output, outputWidth, outputHeight,
                                   dstX, dstY, dstWidth, dstHeight,
                                   filter, stream);
}

// cudaResizeROI (uchar4)
cudaError_t cudaResizeROI( uchar4* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           uchar4* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<uchar4>(input, inputWidth, inputHeight,
                                   srcX, srcY, srcWidth, srcHeight,
                                   output, outputWidth, outputHeight,
                                   dstX, dstY, dstWidth, dstHeight,
                                   filter, stream);
}

// cudaResizeROI (float3)
cudaError_t cudaResizeROI( float3* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float3* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<float3>(input, inputWidth, inputHeight,
                                   srcX, srcY, srcWidth, srcHeight,
                                   output, outputWidth, outputHeight,
                                   dstX, dstY, dstWidth, dstHeight,
                                   filter, stream);
}

// cudaResizeROI (float4)
cudaError_t cudaResizeROI( float4* input, size_t inputWidth, size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           float4* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           cudaFilterMode filter, cudaStream_t stream )
{
    return launchResizeROI<float4>(input, inputWidth, inputHeight,
                                   srcX, srcY, srcWidth, srcHeight,
                                   output, outputWidth, outputHeight,
                                   dstX, dstY, dstWidth, dstHeight,
                                   filter, stream);
}

//-----------------------------------------------------------------------------
// Generic cudaResizeROI for void* and imageFormat
cudaError_t cudaResizeROI( void* input,  size_t inputWidth,  size_t inputHeight,
                           int srcX, int srcY, int srcWidth, int srcHeight,
                           void* output, size_t outputWidth, size_t outputHeight,
                           int dstX, int dstY, int dstWidth, int dstHeight,
                           imageFormat format, cudaFilterMode filter, cudaStream_t stream )
{
    if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )
        return cudaResizeROI((uchar3*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (uchar3*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);
    else if( format == IMAGE_RGBA8 || format == IMAGE_BGRA8 )
        return cudaResizeROI((uchar4*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (uchar4*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);
    else if( format == IMAGE_RGB32F || format == IMAGE_BGR32F )
        return cudaResizeROI((float3*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (float3*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);
    else if( format == IMAGE_RGBA32F || format == IMAGE_BGRA32F )
        return cudaResizeROI((float4*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (float4*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);
    else if( format == IMAGE_GRAY8 )
        return cudaResizeROI((uint8_t*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (uint8_t*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);
    else if( format == IMAGE_GRAY32F )
        return cudaResizeROI((float*)input, inputWidth, inputHeight,
                             srcX, srcY, srcWidth, srcHeight,
                             (float*)output, outputWidth, outputHeight,
                             dstX, dstY, dstWidth, dstHeight, filter, stream);

    LogError(LOG_CUDA "cudaResizeROI() -- invalid image format '%s'\n", imageFormatToStr(format));
    LogError(LOG_CUDA "                  supported formats are:\n");
    LogError(LOG_CUDA "                    * gray8\n");
    LogError(LOG_CUDA "                    * gray32f\n");
    LogError(LOG_CUDA "                    * rgb8, bgr8\n");
    LogError(LOG_CUDA "                    * rgba8, bgra8\n");
    LogError(LOG_CUDA "                    * rgb32f, bgr32f\n");
    LogError(LOG_CUDA "                    * rgba32f, bgra32f\n");

    return cudaErrorInvalidValue;
}
/* clang-format on */
