#include <iostream>
#include <vector>
#include <map>
#include <cuda_runtime.h>


namespace seal
{
    class GPUMemoryPool {
    private:
        struct BlockInfo {
            void* ptr;
            size_t size;
        };

    public:
        GPUMemoryPool() : totalSize_(0), allocatedSize_(0), pool_(nullptr) {}

        ~GPUMemoryPool() {
            if (pool_) {
                checkCudaErrors(cudaFree(pool_));
            }
        }

        void initialize(size_t totalSize) {
            totalSize_ = totalSize;
            checkCudaErrors(cudaMalloc(&pool_, totalSize_));
            cudaDeviceSynchronize();
        }

        void* allocate(size_t size) {
            if (size <= 0) {
                return nullptr;
            }

            void* ptr = nullptr;
            if (!freeBlocks_[size].empty()) {
                // std::cout << "Reusing block of size " << size << std::endl;
                ptr = freeBlocks_[size].back().ptr;
                freeBlocks_[size].pop_back();
            } else {

                ptr = addNewBlock(size);
                // std::cout << "Allocating new block of size " << size << std::endl;
                if (!ptr) {
                    std::cout << "Memory allocation failed." << std::endl;
                    return nullptr;
                }
            }

            return ptr;
        }

        void deallocate(void* ptr, size_t size) {
            if (ptr) {
                freeBlocks_[size].push_back({ptr, size});
            }
        }

        void printPoolStatus() {
            const int barWidth = 50; // 设定字符图形的宽度

            std::cout << "\nGPU Memory Pool Status:\n";
            std::cout << "==============================\n";
            
            // 使用比例来计算已分配和未分配的显存
            double allocatedRatio = static_cast<double>(allocatedSize_) / totalSize_;
            int allocatedChars = static_cast<int>(barWidth * allocatedRatio);
            int freeChars = barWidth - allocatedChars;

            // 打印分配状态的字符图形
            std::cout << "[";
            for (int i = 0; i < allocatedChars; i++) {
                std::cout << "|"; // 已分配显存用“|”表示
            }
            for (int i = 0; i < freeChars; i++) {
                std::cout << " "; // 未分配显存用空格表示
            }
            std::cout << "]\n";

            std::cout << "Allocated: " << allocatedSize_ << " bytes" << std::endl;
            std::cout << "Total: " << totalSize_ << " bytes" << std::endl;

            std::cout << "\nDetails:\n";
            std::cout << "-----------------------------\n";
            for (const auto& entry : freeBlocks_) {
                std::cout << "Block size: " << entry.first << " bytes, Free blocks: " << entry.second.size() << std::endl;
            }

            std::cout << "\n";
        }


    private:
        void* addNewBlock(size_t size) {
            if (allocatedSize_ + size > totalSize_) {
                // There is not enough space in the current pool, allocate a new block

                // printf("Allocated size: %lu, Total size: %lu\n", allocatedSize_, totalSize_);
                // printf("Allocating another new block of size %lu\n", size);
                void* blockPtr = nullptr;
                checkCudaErrors(cudaMalloc(&blockPtr, size));
                allocatedSize_ += size;
                return blockPtr;
            }

            // printf("allocate from pool\n");
            void* blockPtr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(pool_) + allocatedSize_);
            allocatedSize_ += size;
            return blockPtr;
        }


    private:
        mutable size_t totalSize_ = 0;
        mutable size_t allocatedSize_ = 0;
        mutable void* pool_;
        mutable std::map<size_t, std::vector<BlockInfo>> freeBlocks_;
    };


    class PoolManager {
    public:
        static PoolManager& getInstance() {
            static PoolManager instance;
            return instance;
        }

        void initialize(size_t initialPoolSize) {
            if (!initialized_) {
                gpuMemoryPool_ = new GPUMemoryPool();
                gpuMemoryPool_->initialize(initialPoolSize);
                initialized_ = true;
            }
        }

        GPUMemoryPool* getMemoryPool() {
            return gpuMemoryPool_;
        }

        ~PoolManager() {
            delete gpuMemoryPool_;
        }

    private:
        PoolManager() : initialized_(false), gpuMemoryPool_(nullptr) {}

        bool initialized_;
        GPUMemoryPool* gpuMemoryPool_;
    };

}
