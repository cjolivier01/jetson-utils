#pragma once

#include <cuda_runtime.h>

//--------------------------------------------------------
// Structure to hold the four outputs.
// (Remember: any device memory allocated here must later be freed by the caller.)
struct SimpleFullResult {
  float* full_img_1;
  unsigned char* full_mask_1; // can be nullptr if no mask was provided
  float* full_img_2;
  unsigned char* full_mask_2; // can be nullptr if no mask was provided
};


SimpleFullResult simple_make_full(
    // Image 1 (float image)
    const float* d_img_1,
    int img1_width,
    int img1_height,
    int img1_channels,
    // Optional mask 1 (1-channel unsigned char; pass nullptr if not provided)
    const unsigned char* d_mask_1,
    int mask1_width,
    int mask1_height,
    int mask1_channels,
    // Offsets for image 1
    int& x1,
    int& y1,
    // Image 2 (float image)
    const float* d_img_2,
    int img2_width,
    int img2_height,
    int img2_channels,
    // Optional mask 2
    const unsigned char* d_mask_2,
    int mask2_width,
    int mask2_height,
    int mask2_channels,
    // Offsets for image 2
    int& x2,
    int& y2,
    // Canvas dimensions
    int canvas_w,
    int canvas_h,
    float* d_full_img_1,
    unsigned char* d_full_mask_1,
    float* d_full_img_2,
    unsigned char* d_full_mask_2,
    // If true, adjust the origins so that one image is anchored at (0,0)
    bool adjust_origin,
    // Optional CUDA stream (default stream if not provided)
    cudaStream_t stream = 0);
