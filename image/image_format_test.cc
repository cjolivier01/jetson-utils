#include "imageFormat.h"

int main() {
    // Basic sanity on imageFormat helpers
    if (imageFormatFromStr("rgba8") != IMAGE_RGBA8) return 1;
    if (imageFormatChannels(IMAGE_RGBA8) != 4) return 1;
    if (imageFormatIsRGB(IMAGE_BGR8)) return 1; // BGR is not RGB layout
    return 0;
}

