#include "cudaWarp.h"

// warp_perspective_cuda.cu
//
// Pure CUDA implementation of perspective warp sampling (inverse mapping).
// - No OpenCV includes. No cv::cuda::* usage.
// - Supports channels: 1, 3, 4
// - Supports depths: 8U, 16U, 32S, 32F
// - Interpolation: nearest, linear, cubic (Keys)  [A = OPENCV_CUBIC_A]
// - Border modes: CONSTANT, REPLICATE, REFLECT, REFLECT_101, WRAP
//
// IMPORTANT: The matrix M you pass here must map DEST -> SRC (inverse map).
// If you have the forward matrix (SRC -> DEST), invert it on the host first.
//
// Build (library):
// nvcc -O3 -std=c++17 -Xcompiler -fPIC -c warp_perspective_cuda.cu -o warp_perspective_cuda.o
// ar rcs libwarp_perspective_cuda.a warp_perspective_cuda.o
//
// Or build a single binary together with your .cpp wrapper.

#include <cuda_runtime.h>
#include <math.h>


namespace {

// --------------- implementation (device/host) ---------------

#ifndef OPENCV_CUBIC_A
// Set to -0.5f (Catmull-Rom). If your OpenCV build uses -0.75f for INTER_CUBIC,
// change this to -0.75f for bit-closer parity.
#define OPENCV_CUBIC_A (-0.5f)
#endif

enum Interp { I_NEAREST=0, I_LINEAR=1, I_CUBIC=2 };
enum Border { B_CONSTANT=0, B_REPLICATE=1, B_REFLECT=2, B_WRAP=3, B_REFLECT_101=4 };
enum Depth { D_8U=0, D_16U=2, D_32S=4, D_32F=5 };

template<typename T> __device__ __forceinline__ T sat_cast_from_float(float v);
template<> __device__ __forceinline__ unsigned char sat_cast_from_float<unsigned char>(float v){
    int iv = __float2int_rn(v);
    iv = iv < 0 ? 0 : (iv > 255 ? 255 : iv);
    return static_cast<unsigned char>(iv);
}
template<> __device__ __forceinline__ unsigned short sat_cast_from_float<unsigned short>(float v){
    int iv = __float2int_rn(v);
    iv = iv < 0 ? 0 : (iv > 65535 ? 65535 : iv);
    return static_cast<unsigned short>(iv);
}
template<> __device__ __forceinline__ int sat_cast_from_float<int>(float v){
    float lo = -2147483648.0f, hi = 2147483647.0f;
    v = v < lo ? lo : (v > hi ? hi : v);
    return __float2int_rn(v);
}
template<> __device__ __forceinline__ float sat_cast_from_float<float>(float v){ return v; }

__device__ __forceinline__ int borderInterpolateDevice(int p, int len, int borderType)
{
    if ((unsigned)p < (unsigned)len) return p;

    if (borderType == B_REPLICATE) {
        return p < 0 ? 0 : len - 1;
    }
    if (borderType == B_CONSTANT) {
        return -1;
    }
    if (borderType == B_WRAP) {
        if (len == 0) return -1;
        p %= len;
        if (p < 0) p += len;
        return p;
    }
    if (borderType == B_REFLECT || borderType == B_REFLECT_101) {
        if (len == 1) return 0;
        int delta = (borderType == B_REFLECT_101);
        // Reflect until inside range
        while ((unsigned)p >= (unsigned)len) {
            if (p < 0) p = -p - 1 + delta;
            else       p = (len - 1) - (p - len) - delta;
        }
        return p;
    }
    // Fallback
    return -1;
}

__device__ __forceinline__ void cubicCoeffs(float t, float coeffs[4])
{
    // Keys cubic with parameter A
    const float A = OPENCV_CUBIC_A;
    float x = t;
    coeffs[0] = ((A*(x + 1) - 5*A)*(x + 1) + 8*A)*(x + 1) - 4*A;
    coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1.0f;
    float xm = 1.0f - x;
    coeffs[2] = ((A + 2)*xm - (A + 3))*xm*xm + 1.0f;
    coeffs[3] = 1.0f - coeffs[0] - coeffs[1] - coeffs[2];
}

template<typename T, int CN>
__device__ __forceinline__ void loadPixel(const T* rowPtr, int col, float out[CN])
{
#pragma unroll
    for (int c=0;c<CN;++c) out[c] = static_cast<float>(rowPtr[col*CN + c]);
}

template<typename T, int CN>
__device__ __forceinline__ void storePixel(T* rowPtr, int col, const float in[CN])
{
#pragma unroll
    for (int c=0;c<CN;++c) rowPtr[col*CN + c] = sat_cast_from_float<T>(in[c]);
}

template<typename T, int CN>
__device__ __forceinline__ void fetchNearest(
    const T* src, int srcStepElems, int w, int h,
    float xs, float ys, int borderType, const float cval[CN], float out[CN])
{
    int x = __float2int_rn(xs);
    int y = __float2int_rn(ys);

    x = borderInterpolateDevice(x, w, borderType);
    y = borderInterpolateDevice(y, h, borderType);

    if (x < 0 || y < 0) {
#pragma unroll
        for (int c=0;c<CN;++c) out[c] = cval[c];
        return;
    }
    const T* row = src + y * srcStepElems;
    loadPixel<T,CN>(row, x, out);
}

template<typename T, int CN>
__device__ __forceinline__ void fetchBilinear(
    const T* src, int srcStepElems, int w, int h,
    float xs, float ys, int borderType, const float cval[CN], float out[CN])
{
    int x0 = static_cast<int>(floorf(xs));
    int y0 = static_cast<int>(floorf(ys));
    float ax = xs - x0;
    float ay = ys - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    int bx0 = borderInterpolateDevice(x0, w, borderType);
    int bx1 = borderInterpolateDevice(x1, w, borderType);
    int by0 = borderInterpolateDevice(y0, h, borderType);
    int by1 = borderInterpolateDevice(y1, h, borderType);

    float v00[CN], v01[CN], v10[CN], v11[CN];
#pragma unroll
    for (int c=0;c<CN;++c) v00[c]=v01[c]=v10[c]=v11[c]=cval[c];

    if (by0 >= 0) {
        const T* row0 = src + by0 * srcStepElems;
        if (bx0 >= 0) loadPixel<T,CN>(row0, bx0, v00);
        if (bx1 >= 0) loadPixel<T,CN>(row0, bx1, v01);
    }
    if (by1 >= 0) {
        const T* row1 = src + by1 * srcStepElems;
        if (bx0 >= 0) loadPixel<T,CN>(row1, bx0, v10);
        if (bx1 >= 0) loadPixel<T,CN>(row1, bx1, v11);
    }

    float w00 = (1.0f - ax) * (1.0f - ay);
    float w01 =        ax  * (1.0f - ay);
    float w10 = (1.0f - ax) *        ay;
    float w11 =        ax  *        ay;

#pragma unroll
    for (int c=0;c<CN;++c){
        out[c] = v00[c]*w00 + v01[c]*w01 + v10[c]*w10 + v11[c]*w11;
    }
}

template<typename T, int CN>
__device__ __forceinline__ void fetchBicubic(
    const T* src, int srcStepElems, int w, int h,
    float xs, float ys, int borderType, const float cval[CN], float out[CN])
{
#pragma unroll
    for (int c=0;c<CN;++c) out[c] = 0.0f;

    int xi = static_cast<int>(floorf(xs));
    int yi = static_cast<int>(floorf(ys));
    float tx = xs - xi;
    float ty = ys - yi;

    float cx[4], cy[4];
    cubicCoeffs(tx, cx);
    cubicCoeffs(ty, cy);

    int xIdx[4], yIdx[4];
#pragma unroll
    for (int i=0;i<4;++i) {
        xIdx[i] = borderInterpolateDevice(xi - 1 + i, w, borderType);
        yIdx[i] = borderInterpolateDevice(yi - 1 + i, h, borderType);
    }

#pragma unroll
    for (int m=0; m<4; ++m) {
        int yy = yIdx[m];
        float rowAccum[CN];
#pragma unroll
        for (int c=0;c<CN;++c) rowAccum[c] = 0.0f;

        if (yy >= 0) {
            const T* row = src + yy * srcStepElems;
#pragma unroll
            for (int n=0; n<4; ++n) {
                int xx = xIdx[n];
                float px[CN];
#pragma unroll
                for (int c=0;c<CN;++c) px[c] = cval[c];
                if (xx >= 0) loadPixel<T,CN>(row, xx, px);
#pragma unroll
                for (int c=0;c<CN;++c) rowAccum[c] += px[c] * cx[n];
            }
        } else {
#pragma unroll
            for (int n=0; n<4; ++n)
#pragma unroll
                for (int c=0;c<CN;++c) rowAccum[c] += cval[c] * cx[n];
        }
#pragma unroll
        for (int c=0;c<CN;++c) out[c] += rowAccum[c] * cy[m];
    }
}

template<typename T, int CN, int INTERP>
__device__ __forceinline__ void sample(
    const T* src, int srcStepElems, int w, int h,
    float xs, float ys, int borderType, const float cval[CN], float out[CN])
{
    if constexpr (INTERP == I_NEAREST)  fetchNearest<T,CN>(src, srcStepElems, w, h, xs, ys, borderType, cval, out);
    if constexpr (INTERP == I_LINEAR)   fetchBilinear<T,CN>(src, srcStepElems, w, h, xs, ys, borderType, cval, out);
    if constexpr (INTERP == I_CUBIC)    fetchBicubic<T,CN>(src, srcStepElems, w, h, xs, ys, borderType, cval, out);
}

template<typename T, int CN, int INTERP>
__global__ void warpPerspectiveKernel(
    const T* __restrict__ src, size_t srcStepBytes, int srcW, int srcH,
    T* __restrict__ dst, size_t dstStepBytes, int dstW, int dstH,
    float M0,float M1,float M2, float M3,float M4,float M5, float M6,float M7,float M8,
    int borderMode, float c0, float c1, float c2, float c3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const int srcStep = static_cast<int>(srcStepBytes / sizeof(T));
    const int dstStep = static_cast<int>(dstStepBytes / sizeof(T));

    float X = M0 * x + M1 * y + M2;
    float Y = M3 * x + M4 * y + M5;
    float Z = M6 * x + M7 * y + M8;

    float outPix[CN];

    // NEW (safe for CN=1/3/4)
    const float cval4[4] = { c0, c1, c2, c3 };
    float cval[CN];
    #pragma unroll
    for (int i = 0; i < CN; ++i) cval[i] = cval4[i];

    // If Z is too close to 0, treat as outside
    if (fabsf(Z) < 1e-12f) {
#pragma unroll
        for (int c=0;c<CN;++c) outPix[c] = cval[c];
        T* drow = dst + y * dstStep;
        storePixel<T,CN>(drow, x, outPix);
        return;
    }

    float xs = X / Z;
    float ys = Y / Z;

    sample<T,CN,INTERP>(src, srcStep, srcW, srcH, xs, ys, borderMode, cval, outPix);

    T* drow = dst + y * dstStep;
    storePixel<T,CN>(drow, x, outPix);
}

template<typename T, int CN>
static cudaError_t launchTyped(
    const void* src, size_t srcStepBytes, int srcW, int srcH,
    void* dst, size_t dstStepBytes, int dstW, int dstH,
    const float M[9], int interp, int borderMode,
    const float borderValue[4], cudaStream_t stream)
{
    dim3 block(16,16);
    dim3 grid((dstW + block.x - 1)/block.x, (dstH + block.y - 1)/block.y);

    float c0 = borderValue[0];
    float c1 = (CN>1) ? borderValue[1] : 0.f;
    float c2 = (CN>2) ? borderValue[2] : 0.f;
    float c3 = (CN>3) ? borderValue[3] : 0.f;

#define CALL(INTERP_ID) \
    warpPerspectiveKernel<T,CN,INTERP_ID><<<grid, block, 0, stream>>>( \
        static_cast<const T*>(src), srcStepBytes, srcW, srcH, \
        static_cast<T*>(dst),       dstStepBytes, dstW, dstH, \
        M[0],M[1],M[2], M[3],M[4],M[5], M[6],M[7],M[8], \
        borderMode, c0,c1,c2,c3)

    if (interp == I_NEAREST)      { CALL(I_NEAREST); }
    else if (interp == I_CUBIC)   { CALL(I_CUBIC); }
    else                          { CALL(I_LINEAR); } // default

#undef CALL

    return cudaGetLastError();
}

} // namespace

extern "C"
cudaError_t warpPerspectiveCudaRaw(
    const void* src, size_t srcStepBytes, int srcW, int srcH,
    void* dst, size_t dstStepBytes, int dstW, int dstH,
    const float M[9],
    int depth, int channels,
    int interp, int borderMode,
    const float borderValue[4],
    cudaStream_t stream)
{
    // Dispatch by depth & channels
    switch (depth) {
        case D_8U:
            if (channels == 1) return launchTyped<unsigned char,1>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 3) return launchTyped<unsigned char,3>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 4) return launchTyped<unsigned char,4>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            break;
        case D_16U:
            if (channels == 1) return launchTyped<unsigned short,1>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 3) return launchTyped<unsigned short,3>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 4) return launchTyped<unsigned short,4>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            break;
        case D_32S:
            if (channels == 1) return launchTyped<int,1>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 3) return launchTyped<int,3>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 4) return launchTyped<int,4>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            break;
        case D_32F:
            if (channels == 1) return launchTyped<float,1>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 3) return launchTyped<float,3>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            if (channels == 4) return launchTyped<float,4>(src,srcStepBytes,srcW,srcH,dst,dstStepBytes,dstW,dstH,M,interp,borderMode,borderValue,stream);
            break;
        default: break;
    }
    return cudaErrorInvalidValue;
}





