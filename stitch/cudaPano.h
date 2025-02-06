#pragma once

#include "cudaMat.h"
#include "cudaBlend.h"

#include <memory>
namespace hm {
namespace cuda {

template <typename T, typename T_compute>
struct StitchingContext {
  StitchingContext(int batch_size, bool is_hard_seam) : batch_size_(batch_size), is_hard_seam_(is_hard_seam) {}
  // Static buffers
  std::unique_ptr<CudaMat<uint16_t>> remap_1_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_1_y;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_y;

  std::unique_ptr<CudaMat<T_compute>> cudaBlendSoftSeam;
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam;

  // Scratch buffers
  std::unique_ptr<CudaMat<T_compute>> cudaFull1;
  std::unique_ptr<CudaMat<T_compute>> cudaFull2;

  // Laplacian Blend Scratch context
  std::unique_ptr<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>> laplacian_blend_context;

  constexpr int batch_size() const {
    return batch_size_;
  }
  constexpr bool is_hard_seam() const {
    return is_hard_seam_;
  }

 private:
  int batch_size_;
  bool is_hard_seam_;
};

} // namespace cuda
} // namespace hm
