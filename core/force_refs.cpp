#include "video/videoSource.h"
#include "video/videoOutput.h"
#include "video/videoOptions.h"
#include "image/loadImage.h"
#include "cuda/cudaOverlay.h"
#include "cuda/cudaResize.h"
#include "cuda/cudaDraw.h"

// Reference key symbols to ensure they are pulled into the shared library
extern "C" void _force_link_all_symbols() {
    videoOptions opts;
    const char* uri = "";
    videoSource::Create(uri, opts);
    videoOutput::Create(uri, opts);

    void* p = nullptr;
    const float4 color = {0,0,0,0};
    (void)cudaOverlay(p, 0, 0, p, 0, 0, IMAGE_UNKNOWN, 0, 0, 0);
    (void)cudaResize(p, 0, 0, p, 0, 0, IMAGE_UNKNOWN, FILTER_POINT, 0);
    (void)cudaDrawLine(p, p, 0, 0, IMAGE_UNKNOWN, 0, 0, 0, 0, color, 1.0f, 0);
}
