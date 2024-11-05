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

        void printPoolStatusSimple() {
            std::cout << "\nGPU Memory Pool Status:\n";
            std::cout << "==============================\n";

            std::cout << "Allocated: " << allocatedSize_ << " bytes" << std::endl;
            std::cout << "Total: " << totalSize_ << " bytes" << std::endl;
            std::cout << "\n";
        }


        void reset() {
            
            printPoolStatus();

            // 1. 释放超出显存池的额外块
            for (auto& entry : freeBlocks_) {
                for (BlockInfo& block : entry.second) {
                    uintptr_t blockAddress = reinterpret_cast<uintptr_t>(block.ptr);
                    uintptr_t poolStartAddress = reinterpret_cast<uintptr_t>(pool_);
                    uintptr_t poolEndAddress = poolStartAddress + totalSize_;

                    if (blockAddress < poolStartAddress || blockAddress >= poolEndAddress) {
                        // 这个块是在显存池之外分配的
                        cudaError_t err = cudaFree(block.ptr);
                        if (err != cudaSuccess) {
                            std::cerr << "Warning: Failed to free memory block . CUDA error: " << cudaGetErrorString(err) << std::endl;
                        }
                    }
                }
            }

            // 2. 释放所有已经分配的内存块。
            if (pool_) {
                checkCudaErrors(cudaFree(pool_));
                pool_ = nullptr;
            }


            // 3. 清除freeBlocks_映射中的所有条目。
            freeBlocks_.clear();

            // 4. 重置allocatedSize_为0。
            allocatedSize_ = 0;

            // 5. 使用先前的totalSize_值重新初始化池。
            initialize(totalSize_);

            cudaDeviceSynchronize();
        }

        void check() {
            if (allocatedSize_ > 0.9 * totalSize_){
                reset();
            }
        }

        bool isDevicePointer(void* ptr) {
            cudaPointerAttributes attributes;
            cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

            if (err != cudaSuccess) {
                cudaGetLastError();  // 清除错误
                return false;  // 在这种情况下，它可能不是一个有效的CUDA指针
            }

            return attributes.type == cudaMemoryTypeDevice;
        }

        void* allocate(size_t size) {


            if (size <= 0) {
                return nullptr;
            }

            void* ptr = nullptr;
            if (!freeBlocks_[size].empty()) {
                // std::cout << "Reusing block of size " << size << std::endl;
                
                ptr = freeBlocks_[size].back().ptr;
                // if (reinterpret_cast<uintptr_t>(ptr) < reinterpret_cast<uintptr_t>(pool_) || reinterpret_cast<uintptr_t>(ptr) >= (reinterpret_cast<uintptr_t>(pool_) + totalSize_)) {
                //     std::cout << "Warning: Reusing memory block that is out of pool." << std::endl;
                //     // check ptr size 
                //     printf(isDevicePointer(ptr) ? "Device Pointer\n" : "Invalid Pointer\n");
                // }
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
            // std::cout << "return memory of size " << size << std::endl;
            if (ptr) {
                freeBlocks_[size].push_back({ptr, size});
            }
        }


    private:
        void* addNewBlock(size_t size) {
            if (allocatedSize_ + size > totalSize_) {
                // There is not enough space in the current pool, allocate a new block
                // std::cout << "Allocating from out of pool " << size << std::endl;
                void* blockPtr = nullptr;
                checkCudaErrors(cudaMalloc(&blockPtr, size));
                allocatedSize_ += size;
                return blockPtr;
            }

            // std::cout << "Allocating from pool " << size << std::endl;
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
