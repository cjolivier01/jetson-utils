// Basic compile/link test for cudaMath.h host helpers
#include "cudaMath.h"

int main() {
    float3 a = make_float3(1.0f, 2.0f, 3.0f);
    float3 b = make_float3(2.0f);
    float3 c = make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    // just a trivial check to ensure host functions work
    if (c.x != 3.0f) return 1;
    return 0;
}

