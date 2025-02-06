#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "imageFormat.h" // Assumed to define the jetson‑utils imageFormat enum (e.g. IMAGE_BGR8, etc.)

/**
 * @file cvCudaConversions.h
 * @brief Utilities for converting between OpenCV cv::Mat, jetson‑utils image formats,
 *        and CUDA pixel types.
 *
 * This header provides:
 *  - Functions to convert between cv::Mat and jetson‑utils imageFormat.
 *  - Functions to convert between cv::Mat and custom CUDA pixel types.
 *  - Template specializations to map CUDA pixel type enum values to actual CUDA vector types.
 *  - A template class CudaMat to allocate and manage device memory for one or more images.
 */

/**
 * @brief Converts an OpenCV cv::Mat to the corresponding jetson‑utils imageFormat.
 *
 * This function inspects the cv::Mat’s depth and number of channels and returns the matching
 * image format. For example, an 8-bit 3-channel image (CV_8UC3) is mapped to IMAGE_BGR8.
 *
 * @param mat The input OpenCV image.
 * @return The corresponding jetson‑utils imageFormat.
 */
imageFormat cvMatToImageFormat(const cv::Mat& mat);

/**
 * @brief Converts a jetson‑utils imageFormat to an OpenCV type constant.
 *
 * Given an image format (e.g. IMAGE_BGR8), this function returns the matching OpenCV type
 * (e.g. CV_8UC3). Returns -1 if the format is unknown.
 *
 * @param fmt The jetson‑utils image format.
 * @return The corresponding OpenCV type constant.
 */
int imageFormatToCvType(imageFormat fmt);

/**
 * @brief Enumeration of CUDA pixel types.
 *
 * This enum defines a set of common CUDA pixel types that correspond to various image
 * data formats.
 */
enum CudaPixelType {
  CUDA_PIXEL_UNKNOWN = -1, ///< Unknown or unsupported pixel type.
  CUDA_PIXEL_UCHAR1, ///< 8-bit, 1 channel (unsigned char).
  CUDA_PIXEL_UCHAR3, ///< 8-bit, 3 channels (uchar3).
  CUDA_PIXEL_UCHAR4, ///< 8-bit, 4 channels (uchar4).
  CUDA_PIXEL_INT1, ///< 32-bit int, 1 channel (int).
  CUDA_PIXEL_INT3, ///< 32-bit int, 3 channels (int3).
  CUDA_PIXEL_INT4, ///< 32-bit int, 4 channels (int4).
  CUDA_PIXEL_FLOAT1, ///< 32-bit float, 1 channel (float).
  CUDA_PIXEL_FLOAT3, ///< 32-bit float, 3 channels (float3).
  CUDA_PIXEL_FLOAT4 ///< 32-bit float, 4 channels (float4).
};

/**
 * @brief Converts an OpenCV cv::Mat to a CudaPixelType.
 *
 * This function inspects the depth and number of channels of the input cv::Mat and returns the
 * corresponding CUDA pixel type. For example, CV_8UC3 is mapped to CUDA_PIXEL_UCHAR3.
 *
 * @param mat The input OpenCV image.
 * @return The corresponding CudaPixelType.
 */
CudaPixelType cvMatToCudaPixelType(const cv::Mat& mat);

/**
 * @brief Converts a CudaPixelType to an OpenCV type constant.
 *
 * This function returns the OpenCV type (e.g. CV_8UC3) that corresponds to the given CUDA pixel type.
 *
 * @param fmt The CUDA pixel type.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int cudaPixelTypeToCvType(CudaPixelType fmt);

/**
 * @brief Template mapping from a CudaPixelType enumeration to the corresponding CUDA type.
 *
 * Specializations of this template define a nested type alias `type` corresponding to the actual
 * CUDA vector or scalar type.
 *
 * @tparam T The enumerator from the CudaPixelType enum.
 */
template <CudaPixelType T>
struct CudaPixelTypeToCudaType; // Primary template declaration (no definition).

// --- Template Specializations --- //

/** Specialization: 8-bit, 1-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR1> {
  using type = unsigned char;
};

/** Specialization: 8-bit, 3-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR3> {
  using type = uchar3;
};

/** Specialization: 8-bit, 4-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR4> {
  using type = uchar4;
};

/** Specialization: 32-bit int, 1-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT1> {
  using type = int;
};

/** Specialization: 32-bit int, 3-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT3> {
  using type = int3;
};

/** Specialization: 32-bit int, 4-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT4> {
  using type = int4;
};

/** Specialization: 32-bit float, 1-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT1> {
  using type = float;
};

/** Specialization: 32-bit float, 3-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT3> {
  using type = float3;
};

/** Specialization: 32-bit float, 4-channel. */
template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT4> {
  using type = float4;
};

/**
 * @brief Templated class to manage CUDA device memory for one or more images.
 *
 * The CudaMat class allocates device memory for an image (or a batch of images) and provides
 * functionality to download the device memory back to a cv::Mat.
 *
 * @tparam T The type stored in device memory (default is float3).
 */
template <typename T = float3>
class CudaMat {
 public:
  // Delete copy and move constructors.
  CudaMat(const CudaMat&) = delete;
  CudaMat(CudaMat&&) = delete;

  /**
   * @brief Constructs a CudaMat from a single cv::Mat.
   *
   * Allocates device memory for the image and (optionally) copies the cv::Mat data to the device.
   *
   * @param mat The input cv::Mat.
   * @param copy If true, the data is copied to device memory.
   */
  CudaMat(const cv::Mat& mat, bool copy = true);

  /**
   * @brief Constructs a CudaMat from a batch of cv::Mat images.
   *
   * Allocates device memory to hold all images in the batch and (optionally) copies the data.
   *
   * @param mat_batch A vector of cv::Mat images.
   * @param copy If true, the data is copied to device memory.
   */
  CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy = true);

  /**
   * @brief Destructor.
   *
   * Frees the allocated device memory.
   */
  ~CudaMat();

  /**
   * @brief Downloads an image from device memory to a cv::Mat.
   *
   * For batched images, the parameter `batch_item` specifies which image to download.
   *
   * @param batch_item The index of the image to download (default is 0).
   * @return A cv::Mat containing the downloaded image.
   */
  cv::Mat download(int batch_item = 0) const;

  /// @brief Returns a pointer to the device memory.
  T* data();
  /// @brief Returns a const pointer to the device memory.
  const T* data() const;
  /// @brief Returns the image width (number of columns).
  constexpr int width() const;
  /// @brief Returns the image height (number of rows).
  constexpr int height() const;
  /// @brief Returns the OpenCV type of the image.
  constexpr int type() const;
  /// @brief Returns the number of images in the batch.
  constexpr int batch_size() const;

 private:
  T* d_data; ///< Pointer to the device memory.
  size_t size; ///< Total size (in bytes) allocated on the device.
  int rows_, cols_, type_; ///< Dimensions and OpenCV type of the image.
  int batch_size_; ///< Number of images in the batch.
};

// Include the inline implementations of the template methods.
#include "cudaMat.inl"
