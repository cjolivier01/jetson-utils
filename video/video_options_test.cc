#include "videoOptions.h"
#include "URI.h"
#include <cstring>

int main() {
    // Verify defaults
    videoOptions opt;
    if (opt.width != 0 || opt.height != 0) return 1;
    if (opt.ioType != videoOptions::INPUT) return 1;

    // Verify URI parsing and codec defaults
    URI uri;
    if (!uri.Parse("file:///tmp/test.mp4")) return 1;
    opt.resource = uri;
    opt.codec = videoOptions::CODEC_H264;

    const char* t = videoOptions::IoTypeToStr(opt.ioType);
    if (!t || std::strcmp(t, "input") != 0) return 1;

    return 0;
}
