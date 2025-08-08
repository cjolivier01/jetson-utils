#include "cudaOverlay.h"

// Provide a definition for the base cudaOverlay() by delegating to the
// pitched version, to ensure the symbol is available from the extension.
cudaError_t cudaOverlay(
    void* input,
    size_t inputWidth,
    size_t inputHeight,
    void* output,
    size_t outputWidth,
    size_t outputHeight,
    imageFormat format,
    int x,
    int y,
    cudaStream_t stream)
{
    const size_t inputPitch = 0;
    const size_t outputPitch = 0;
    return cudaOverlayPitch(
        input,
        inputWidth,
        inputHeight,
        inputPitch,
        output,
        outputWidth,
        outputHeight,
        outputPitch,
        format,
        x,
        y,
        stream);
}
