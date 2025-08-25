#pragma once

#include <cuda_runtime.h>
#include "helper_cuda.h"

template <typename T>
class CudaVector {
public:

    CudaVector() : size_(0), capacity_(1) {
        checkCudaErrors(cudaMallocManaged(&data_, capacity_ * sizeof(T)));
    }

    ~CudaVector() {
        for (size_t i = 0; i < size_; i++) {
            data_[i].~T();
        }
        checkCudaErrors(cudaFree(data_));
    }

    __host__ __device__ T& operator[](size_t index) { return data_[index]; }
    __host__ __device__ const T& operator[](size_t index) const { return data_[index]; }

    __host__ __device__ size_t size() const { return size_; }
    __host__ __device__ T* data() const { return data_; }

    void push_back(const T& value) {
        static_assert(std::is_copy_constructible_v<T>,
                      "push_back(const T&) requires copy-constructible T");
        if (size_ == capacity_) reserve(capacity_ ? capacity_ * 2 : 1);
        new (&data_[size_]) T(value);   // copy-construct
        ++size_;
    }

    void push_back(T&& value) {
        if (size_ == capacity_) reserve(capacity_ ? capacity_ * 2 : 1);
        new (&data_[size_]) T(std::move(value));  // move-construct
        ++size_;
    }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) reserve(capacity_ ? capacity_ * 2 : 1);
        new (&data_[size_]) T(std::forward<Args>(args)...);
        return data_[size_++];
    }

private:
    void reserve(size_t new_cap) {
        if (new_cap <= capacity_) return;
        T* new_data = nullptr;
        checkCudaErrors(cudaMallocManaged(&new_data, new_cap * sizeof(T)));
        for (size_t i = 0; i < size_; ++i) {
            new (&new_data[i]) T(std::move_if_noexcept(data_[i])); // moves
            data_[i].~T();                                         // destroy old
        }
        if (data_) checkCudaErrors(cudaFree(data_));
        data_ = new_data;
        capacity_ = new_cap;
    }
    T* data_;
    size_t size_;
    size_t capacity_;
};