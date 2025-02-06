#include "cudaMat.h"

#include <cassert>
#include <cuda_runtime.h>

template <typename T>
CudaMat<T>::CudaMat(const cv::Mat& mat, bool copy)
    : rows_(mat.rows), cols_(mat.cols), type_(mat.type()), batch_size_(1)
{
    size = mat.total() * mat.elemSize();
    cudaMalloc(&d_data, size);
    assert(mat.isContinuous());
    if (copy) {
        cudaMemcpy(d_data, mat.data, size, cudaMemcpyHostToDevice);
    }
}

template <typename T>
CudaMat<T>::CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy)
    : batch_size_(static_cast<int>(mat_batch.size()))
{
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

template <typename T>
CudaMat<T>::~CudaMat() {
    if (d_data) {
        cudaFree(d_data);
    }
}

template <typename T>
cv::Mat CudaMat<T>::download(int batch_item) const {
    assert(batch_item >= 0 && batch_item < batch_size_);
    cv::Mat mat(rows_, cols_, type_);
    size_t size_each = mat.total() * mat.elemSize();
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_data) + batch_item * size_each;
    cudaMemcpy(mat.data, src_ptr, size_each, cudaMemcpyDeviceToHost);
    return mat;
}

template <typename T>
T* CudaMat<T>::data() {
    return d_data;
}

template <typename T>
const T* CudaMat<T>::data() const {
    return d_data;
}

template <typename T>
constexpr int CudaMat<T>::width() const {
    return cols_;
}

template <typename T>
constexpr int CudaMat<T>::height() const {
    return rows_;
}

template <typename T>
constexpr int CudaMat<T>::type() const {
    return type_;
}

template <typename T>
constexpr int CudaMat<T>::batch_size() const {
    return batch_size_;
}
