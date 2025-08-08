// Minimal CUDA Bayer (CFA) -> RGB (packed, 8-bit) with bilinear demosaicing.
// Mirrors the NPP signature semantics (but uses plain C++/CUDA types).
// Build with: nvcc -O2 -arch=sm_70 bayer_to_rgb.cu -o bayer_to_rgb

#include "cudaBayer.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

using namespace bayer;

namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ uint8_t ld1(const uint8_t *base, int stepBytes,
                                       int x, int y) {
  return *(base + y * stepBytes + x);
}

__device__ __forceinline__ void st3(uint8_t *base, int stepBytes, int x, int y,
                                    uint8_t r, uint8_t g, uint8_t b) {
  uint8_t *p = base + y * stepBytes + 3 * x;
  p[0] = r;
  p[1] = g;
  p[2] = b; // packed RGB
}

struct RBParity {
  int rx, ry;
  int bx, by;
};

__host__ __device__ __forceinline__ RBParity
rb_parity_from_grid(BayerGrid grid) {
  // Coordinates are relative to the ROI origin.
  // These indicate where RED and BLUE land at (parity x,y) == (0,0).
  switch (grid) {
  case BAYER_RGGB:
    return {0, 0, 1, 1};
  case BAYER_GRBG:
    return {1, 0, 0, 1};
  case BAYER_GBRG:
    return {0, 1, 1, 0};
  case BAYER_BGGR:
    return {1, 1, 0, 0};
  default:
    return {0, 0, 1, 1};
  }
}

__global__ void kCfaToRgb8uC1C3(const uint8_t *__restrict__ src, int srcStep,
                                Size srcSize, Rect roi,
                                uint8_t *__restrict__ dst, int dstStep,
                                BayerGrid grid) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x; // within ROI
  int iy = blockIdx.y * blockDim.y + threadIdx.y; // within ROI
  if (ix >= roi.width || iy >= roi.height)
    return;

  // Absolute coordinates in the full source image.
  int sx = roi.x + ix;
  int sy = roi.y + iy;

  // Neighbor coords with clamping to full image bounds.
  int x0 = clampi(sx, 0, srcSize.width - 1);
  int y0 = clampi(sy, 0, srcSize.height - 1);
  int xm = clampi(sx - 1, 0, srcSize.width - 1);
  int xp = clampi(sx + 1, 0, srcSize.width - 1);
  int ym = clampi(sy - 1, 0, srcSize.height - 1);
  int yp = clampi(sy + 1, 0, srcSize.height - 1);

  // Parity within ROI (important: grid is defined at (roi.x, roi.y))
  int px = ix & 1;
  int py = iy & 1;

  // Determine which parity is RED and which is BLUE.
  RBParity rb = rb_parity_from_grid(grid);

  auto isRed = [&](int ppx, int ppy) {
    return (ppx == rb.rx) && (ppy == rb.ry);
  };
  auto isBlue = [&](int ppx, int ppy) {
    return (ppx == rb.bx) && (ppy == rb.by);
  };
  auto isGreen = [&](int ppx, int ppy) {
    return !isRed(ppx, ppy) && !isBlue(ppx, ppy);
  };

  // Load samples
  uint8_t C = ld1(src, srcStep, x0, y0);
  uint8_t L = ld1(src, srcStep, xm, y0);
  uint8_t Rr = ld1(src, srcStep, xp, y0);
  uint8_t U = ld1(src, srcStep, x0, ym);
  uint8_t D = ld1(src, srcStep, x0, yp);
  uint8_t UL = ld1(src, srcStep, xm, ym);
  uint8_t UR = ld1(src, srcStep, xp, ym);
  uint8_t DL = ld1(src, srcStep, xm, yp);
  uint8_t DR = ld1(src, srcStep, xp, yp);

  // Helpers for averaging with rounding
  auto avg2 = [](int a, int b) -> uint8_t {
    return (uint8_t)((a + b + 1) >> 1);
  };
  auto avg4 = [](int a, int b, int c, int d) -> uint8_t {
    return (uint8_t)((a + b + c + d + 2) >> 2);
  };

  uint8_t outR = 0, outG = 0, outB = 0;

  if (isRed(px, py)) {
    // R site
    outR = C;
    // G from cross
    outG = avg4(L, Rr, U, D);
    // B from diagonals
    outB = avg4(UL, UR, DL, DR);
  } else if (isBlue(px, py)) {
    // B site
    outB = C;
    // G from cross
    outG = avg4(L, Rr, U, D);
    // R from diagonals
    outR = avg4(UL, UR, DL, DR);
  } else {
    // Green site
    outG = C;

    // Decide orientation: is this green on a "red row" (same row parity as
    // red)?
    bool sameRowAsRed = (py == rb.ry);
    // If same row as red: red is horizontal, blue is vertical.
    // If different row:   red is vertical,   blue is horizontal.

    if (sameRowAsRed) {
      outR = avg2(L, Rr);
      outB = avg2(U, D);
    } else {
      outR = avg2(U, D);
      outB = avg2(L, Rr);
    }
  }

  // Store to destination (ROI-relative coords)
  st3(dst, dstStep, ix, iy, outR, outG, outB);
}

} // namespace

namespace bayer {
// Host-side convenience wrapper (mirrors NPP args, minus context/interp).
// pSrc: C1 8u, nSrcStep: bytes/row
// pDst: C3 8u (packed RGB), nDstStep: bytes/row
cudaError_t CFAToRGB_8u_C1C3R(const uint8_t *pSrc, int nSrcStep, Size oSrcSize,
                              Rect oSrcROI, uint8_t *pDst, int nDstStep,
                              BayerGrid eGrid, cudaStream_t stream) {
  if (!pSrc || !pDst)
    return cudaErrorInvalidDevicePointer;
  if (oSrcROI.x < 0 || oSrcROI.y < 0 ||
      oSrcROI.x + oSrcROI.width > oSrcSize.width ||
      oSrcROI.y + oSrcROI.height > oSrcSize.height)
    return cudaErrorInvalidValue;

  dim3 block(32, 8);
  dim3 grid((oSrcROI.width + block.x - 1) / block.x,
            (oSrcROI.height + block.y - 1) / block.y);

  kCfaToRgb8uC1C3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, oSrcSize, oSrcROI,
                                              pDst, nDstStep, eGrid);

  return cudaGetLastError();
}
} // namespace bayer
