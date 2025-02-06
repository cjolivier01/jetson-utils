#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include "imageFormat.h"  // Assumed to define the jetson‑utils imageFormat enum (e.g. IMAGE_BGR8, etc.)

//---------------------------------------------------------------------------
// Non-template function declarations (implemented in cvCudaConversions.cpp)
//---------------------------------------------------------------------------

// Given a cv::Mat, return the corresponding jetson‑utils imageFormat.
imageFormat cvMatToImageFormat(const cv::Mat& mat);

// Given a jetson‑utils imageFormat, return the corresponding OpenCV type (e.g. CV_8UC3).
int imageFormatToCvType(imageFormat fmt);

//---------------------------------------------------------------------------
// CUDA Pixel Type enum and conversion function declarations
//---------------------------------------------------------------------------

enum CudaPixelType {
    CUDA_PIXEL_UNKNOWN = -1,
    CUDA_PIXEL_UCHAR1,   // 8-bit, 1 channel (unsigned char)
    CUDA_PIXEL_UCHAR3,   // 8-bit, 3 channels (uchar3)
    CUDA_PIXEL_UCHAR4,   // 8-bit, 4 channels (uchar4)
    CUDA_PIXEL_INT1,     // 32-bit int, 1 channel (int)
    CUDA_PIXEL_INT3,     // 32-bit int, 3 channels (int3)
    CUDA_PIXEL_INT4,     // 32-bit int, 4 channels (int4)
    CUDA_PIXEL_FLOAT1,   // 32-bit float, 1 channel (float)
    CUDA_PIXEL_FLOAT3,   // 32-bit float, 3 channels (float3)
    CUDA_PIXEL_FLOAT4    // 32-bit float, 4 channels (float4)
};

// Convert a cv::Mat to a CudaPixelType.
CudaPixelType cvMatToCudaPixelType(const cv::Mat& mat);

// Given a CudaPixelType, return the corresponding OpenCV type (e.g. CV_8UC3).
int cudaPixelTypeToCvType(CudaPixelType fmt);

//---------------------------------------------------------------------------
// Template Specializations: CudaPixelTypeToCudaType
//---------------------------------------------------------------------------

template <CudaPixelType T>
struct CudaPixelTypeToCudaType;  // Primary template (no definition)

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR1> {
    using type = unsigned char;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR3> {
    using type = uchar3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR4> {
    using type = uchar4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT1> {
    using type = int;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT3> {
    using type = int3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT4> {
    using type = int4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT1> {
    using type = float;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT3> {
    using type = float3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT4> {
    using type = float4;
};

//---------------------------------------------------------------------------
// Template Class: CudaMat
//---------------------------------------------------------------------------

template <typename T = float3>
class CudaMat {
 public:
  // Delete copy and move constructors.
  CudaMat(const CudaMat&) = delete;
  CudaMat(CudaMat&&) = delete;

  // Constructors.
  CudaMat(const cv::Mat& mat, bool copy = true);
  CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy = true);

  // Destructor.
  ~CudaMat();

  // Download the image corresponding to the given batch index.
  cv::Mat download(int batch_item = 0) const;

  // Accessor functions.
  T* data();
  const T* data() const;
  constexpr int width() const;
  constexpr int height() const;
  constexpr int type() const;
  constexpr int batch_size() const;

 private:
  T* d_data;
  size_t size; // Total size (in bytes) allocated on the device.
  int rows_, cols_, type_;
  int batch_size_;
};


