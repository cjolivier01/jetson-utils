#pragma once

#include "cuda_runtime_compat.h"

#include <cstdint>

namespace cuda {
namespace misc {

// Row start pointer using pitch
template <typename T>
inline T* __device__ row_start(T* image, size_t y, size_t pitch) {
  return reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(image) + y * pitch);
}

// Row start pointer using pitch
template <typename T>
inline const T* __device__ row_start(const T* image, size_t y, size_t pitch) {
  return reinterpret_cast<const T*>(reinterpret_cast<const unsigned char*>(image) + y * pitch);
}

} // namespace misc
} // namespace cuda
