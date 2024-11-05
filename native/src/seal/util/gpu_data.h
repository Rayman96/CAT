#pragma once

#include "seal/util/common.h"
#include "seal/util/helper.cuh"
#include <algorithm>
#include <iostream>
#include <limits>
#include <type_traits>

namespace seal
{
    class GPUData
    {
        public:
            GPUData() : data_(nullptr) {}

            GPUData(size_t size) {
                d_capacity_ = size;
                allocate_gpu<uint64_t>(&rawPtr_, d_capacity_);
            };

            GPUData(uint64_t* data, size_t size) {
                d_capacity_ = size;
                allocate_gpu<uint64_t>(&rawPtr_, d_capacity_);
                cudaMemcpy(rawPtr_, data, d_capacity_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
            }

            ~GPUData() {
                if (data_ != nullptr) {
                    deallocate_gpu<uint64_t>(&rawPtr_, d_capacity_);
                }
            }


            uint64_t* data() {
                return data_.get();
            }

            uint64_t* data() const {
                return data_.get();
            }

            void setData(std::shared_ptr<uint64_t> newData) {
                data_ = newData;  // 使用复制构造函数，确保共享所有权
            }

            size_t capacity() {
                return d_capacity_;
            }

            void alloc(size_t size) {
                if (size > d_capacity_) {
                    release();
                    d_capacity_ = size;
                    allocate_gpu<uint64_t>(&rawPtr_, d_capacity_);
                }
            }

            void release() {
                deallocate_gpu<uint64_t>(&rawPtr_, d_capacity_);
                d_capacity_ = 0;
            }

            std::shared_ptr<uint64_t> data_ = nullptr;
            size_t d_capacity_ = 0;

        private:
            uint64_t* rawPtr_ = data_.get();
    };
}