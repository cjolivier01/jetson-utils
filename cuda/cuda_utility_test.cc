#include "cudaUtility.h"

int main() {
    // Use __host__ iDivUp helper (no CUDA runtime required)
    int v = iDivUp(10, 8);
    if (v != 2) return 1;

    const char* tag = LOG_CUDA;
    (void)tag;
    return 0;
}

