#include "cudaOverlay.h"
#include "cudaDraw.h"
#include "cudaResize.h"
#include "cudaFilterMode.h"

extern "C" void _force_link_cuda_symbols() {
    // Create dummy references to force link of these symbols from static CUDA archives
    void* p = nullptr;
    (void)cudaOverlay(p, 0, 0, p, 0, 0, IMAGE_UNKNOWN, 0, 0, 0);
    (void)cudaOverlayPitch(p, 0, 0, 0, p, 0, 0, 0, IMAGE_UNKNOWN, 0, 0, 0);
    const float4 color = {0,0,0,0};
    (void)cudaDrawLine(p, p, 0, 0, IMAGE_UNKNOWN, 0, 0, 0, 0, color, 1.0f, 0);
    (void)cudaResize(p, 0, 0, p, 0, 0, IMAGE_UNKNOWN, FILTER_POINT, 0);
}
