#include "cudaMat.h"
#include <cuda_runtime.h>
#include <cassert>

/**
 * @brief Converts an OpenCV cv::Mat to the corresponding jetson‑utils imageFormat.
 *
 * Inspects the cv::Mat’s depth and channel count to determine the appropriate image format.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding jetson‑utils imageFormat.
 */
imageFormat cvMatToImageFormat(const cv::Mat& mat) {
  int depth = mat.depth(); // e.g. CV_8U, CV_32F, etc.
  int channels = mat.channels(); // e.g. 1, 3, or 4

  if (depth == CV_8U) {
    if (channels == 1)
      return IMAGE_GRAY8;
    else if (channels == 3)
      return IMAGE_BGR8; // (or IMAGE_RGB8 if channels are swapped)
    else if (channels == 4)
      return IMAGE_BGRA8;
  } else if (depth == CV_32F) {
    if (channels == 1)
      return IMAGE_GRAY32F;
    else if (channels == 3)
      return IMAGE_BGR32F;
    else if (channels == 4)
      return IMAGE_BGRA32F;
  }
  return IMAGE_UNKNOWN;
}

/**
 * @brief Converts a jetson‑utils imageFormat to an OpenCV type constant.
 *
 * Maps the provided imageFormat to an OpenCV type such as CV_8UC3.
 *
 * @param fmt The jetson‑utils image format.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int imageFormatToCvType(imageFormat fmt) {
  switch (fmt) {
    case IMAGE_GRAY8:
      return CV_8UC1;
    case IMAGE_BGR8:
      return CV_8UC3;
    case IMAGE_BGRA8:
      return CV_8UC4;
    case IMAGE_GRAY32F:
      return CV_32FC1;
    case IMAGE_BGR32F:
      return CV_32FC3;
    case IMAGE_BGRA32F:
      return CV_32FC4;
    default:
      return -1; // Unknown format
  }
}

/**
 * @brief Converts an OpenCV cv::Mat to a CudaPixelType.
 *
 * Determines the CUDA pixel type based on the cv::Mat’s depth and channel count.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding CudaPixelType.
 */
CudaPixelType cvMatToCudaPixelType(const cv::Mat& mat) {
  int depth = mat.depth();
  int channels = mat.channels();

  if (depth == CV_8U) {
    if (channels == 1)
      return CUDA_PIXEL_UCHAR1;
    else if (channels == 3)
      return CUDA_PIXEL_UCHAR3; // Maps to CUDA’s uchar3.
    else if (channels == 4)
      return CUDA_PIXEL_UCHAR4;
  } else if (depth == CV_32S) {
    if (channels == 1)
      return CUDA_PIXEL_INT1;
    else if (channels == 3)
      return CUDA_PIXEL_INT3;
    else if (channels == 4)
      return CUDA_PIXEL_INT4;
  } else if (depth == CV_32F) {
    if (channels == 1)
      return CUDA_PIXEL_FLOAT1;
    else if (channels == 3)
      return CUDA_PIXEL_FLOAT3; // Maps to CUDA’s float3.
    else if (channels == 4)
      return CUDA_PIXEL_FLOAT4; // Maps to CUDA’s float4.
  }
  return CUDA_PIXEL_UNKNOWN;
}

/**
 * @brief Converts a CudaPixelType to an OpenCV type constant.
 *
 * Maps the given CUDA pixel type (e.g. CUDA_PIXEL_UCHAR3) to the corresponding OpenCV type (e.g. CV_8UC3).
 *
 * @param fmt The CUDA pixel type.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int cudaPixelTypeToCvType(CudaPixelType fmt) {
  switch (fmt) {
    case CUDA_PIXEL_UCHAR1:
      return CV_8UC1;
    case CUDA_PIXEL_UCHAR3:
      return CV_8UC3;
    case CUDA_PIXEL_UCHAR4:
      return CV_8UC4;
    case CUDA_PIXEL_INT1:
      return CV_32SC1;
    case CUDA_PIXEL_INT3:
      return CV_32SC3;
    case CUDA_PIXEL_INT4:
      return CV_32SC4;
    case CUDA_PIXEL_FLOAT1:
      return CV_32FC1;
    case CUDA_PIXEL_FLOAT3:
      return CV_32FC3;
    case CUDA_PIXEL_FLOAT4:
      return CV_32FC4;
    default:
      return -1;
  }
}
