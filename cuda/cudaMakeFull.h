#pragma once

//--------------------------------------------------------
// Structure to hold the four outputs.
// (Remember: any device memory allocated here must later be freed by the caller.)
struct SimpleFullResult {
  float* full_img_1;
  unsigned char* full_mask_1; // can be nullptr if no mask was provided
  float* full_img_2;
  unsigned char* full_mask_2; // can be nullptr if no mask was provided
};


