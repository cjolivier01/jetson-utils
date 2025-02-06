#pragma once

#include "cudaMat.h"

#include <cuda_runtime.h>
#include <cassert>

/**
 * @brief Constructs a CudaMat from a single cv::Mat.
 *
 * Allocates device memory and (optionally) copies the cv::Mat data to the device.
 *
 * @tparam T The CUDA pixel type.
 * @param mat The input cv::Mat.
 * @param copy If true, copies the data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const cv::Mat& mat, bool copy)
    : rows_(mat.rows), cols_(mat.cols), type_(mat.type()), batch_size_(1) {
  size = mat.total() * mat.elemSize();
  cudaMalloc(&d_data, size);
  assert(mat.isContinuous());
  if (copy) {
    cudaMemcpy(d_data, mat.data, size, cudaMemcpyHostToDevice);
  }
}

/**
 * @brief Constructs a CudaMat from a batch of cv::Mat images.
 *
 * Allocates device memory to hold all images and (optionally) copies each image to the device.
 *
 * @tparam T The CUDA pixel type.
 * @param mat_batch Vector of input cv::Mat images.
 * @param copy If true, copies data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy)
    : batch_size_(static_cast<int>(mat_batch.size())) {
  assert(batch_size_ > 0);
  const cv::Mat& first = mat_batch.at(0);
  rows_ = first.rows;
  cols_ = first.cols;
  type_ = first.type();
  size_t size_each = first.total() * first.elemSize();
  size = size_each * batch_size_;
  cudaMalloc(&d_data, size);
  if (copy) {
    uint8_t* p = reinterpret_cast<uint8_t*>(d_data);
    for (const cv::Mat& mat : mat_batch) {
      assert(mat.isContinuous());
      cudaMemcpy(p, mat.data, size_each, cudaMemcpyHostToDevice);
      p += size_each;
    }
  }
}

/**
 * @brief Destructor for CudaMat.
 *
 * Frees the allocated device memory.
 *
 * @tparam T The CUDA pixel type.
 */
template <typename T>
CudaMat<T>::~CudaMat() {
  if (d_data) {
    cudaFree(d_data);
  }
}

/**
 * @brief Downloads an image from device memory to a cv::Mat.
 *
 * Downloads the image at the specified batch index.
 *
 * @tparam T The CUDA pixel type.
 * @param batch_item The index of the image in the batch (default is 0).
 * @return A cv::Mat containing the downloaded image.
 */
template <typename T>
cv::Mat CudaMat<T>::download(int batch_item) const {
  assert(batch_item >= 0 && batch_item < batch_size_);
  cv::Mat mat(rows_, cols_, type_);
  size_t size_each = mat.total() * mat.elemSize();
  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_data) + batch_item * size_each;
  cudaMemcpy(mat.data, src_ptr, size_each, cudaMemcpyDeviceToHost);
  return mat;
}

/**
 * @brief Returns a pointer to the device memory.
 *
 * @tparam T The CUDA pixel type.
 * @return Pointer to device memory.
 */
template <typename T>
T* CudaMat<T>::data() {
  return d_data;
}

/**
 * @brief Returns a const pointer to the device memory.
 *
 * @tparam T The CUDA pixel type.
 * @return Const pointer to device memory.
 */
template <typename T>
const T* CudaMat<T>::data() const {
  return d_data;
}

/**
 * @brief Returns the image width (number of columns).
 *
 * @tparam T The CUDA pixel type.
 * @return Image width.
 */
template <typename T>
constexpr int CudaMat<T>::width() const {
  return cols_;
}

/**
 * @brief Returns the image height (number of rows).
 *
 * @tparam T The CUDA pixel type.
 * @return Image height.
 */
template <typename T>
constexpr int CudaMat<T>::height() const {
  return rows_;
}

/**
 * @brief Returns the OpenCV type of the image.
 *
 * @tparam T The CUDA pixel type.
 * @return OpenCV type constant.
 */
template <typename T>
constexpr int CudaMat<T>::type() const {
  return type_;
}

/**
 * @brief Returns the number of images in the batch.
 *
 * @tparam T The CUDA pixel type.
 * @return Batch size.
 */
template <typename T>
constexpr int CudaMat<T>::batch_size() const {
  return batch_size_;
}
