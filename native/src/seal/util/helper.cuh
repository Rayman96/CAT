#pragma once
#include "seal/context.cuh"
#include "seal/gpu_memorypool.h"
#include "common.h"
#include "seal/util/rns.cuh"
#include "seal/util/uintarith.cuh"
#include "seal/util/uintarithsmallmod.cuh"
#include <stdlib.h>
#include "kuint128.cuh"
#include <cuda_occupancy.h>
#include <vector>
#include <algorithm>
// 计算 a^b mod mod
namespace seal{
    // 计算 a^b mod mod
    __device__ __host__ inline uint64_t modpow128(uint64_t a, uint64_t b, uint64_t mod)
    {
        uint64_t res = 1;

        if (1 & b)
            res = a;

        while (b != 0)
        {
            b = b >> 1;
            k_uint128_t t128 = host64x2(a, a);
            a = (t128 % mod).low;
            if (b & 1)
            {
                k_uint128_t r128 = host64x2(res, a);
                res = (r128 % mod).low;
            }
        }
        return res;
    }

    __device__ __host__ inline unsigned modpow64(unsigned a, unsigned b, unsigned mod)
    {
        unsigned res = 1;

        if (1 & b)
            res = a;

        while (b != 0)
        {
            b = b >> 1;
            uint64_t t64 = (uint64_t)a * a;
            a = t64 % mod;
            if (b & 1)
            {
                uint64_t r64 = (uint64_t)a * res;
                res = r64 % mod;
            }
        }
        return res;
    }

    __device__ inline uint64_t modinv128(uint64_t a, uint64_t q)
    {
        uint64_t ainv = modpow128(a, q - 2, q);
        return ainv;
    }

    __device__ inline uint64_t bitReverse(uint64_t a, int bit_length)
    {
        uint64_t res = 0;

        for (int i = 0; i < bit_length; i++)
        {
            res <<= 1;
            res = (a & 1) | res;
            a >>= 1;
        }

        return res;
    }


    template <typename T>
    inline void allocate_gpu(T **ptr, size_t size){
        PoolManager& poolManager = PoolManager::getInstance();
        GPUMemoryPool* memoryPool = poolManager.getMemoryPool();
        *ptr = static_cast<T*>(memoryPool->allocate(size * sizeof(T)));
        // memoryPool->printPoolStatus();
    }

    template <typename T>
    inline void deallocate_gpu(T **ptr, size_t size){
        PoolManager& poolManager = PoolManager::getInstance();
        GPUMemoryPool* memoryPool = poolManager.getMemoryPool();
        memoryPool->deallocate(*ptr, size*sizeof(T));
        // memoryPool->printPoolStatus();

    }


    template <typename T>
    void allocate_gpu(std::shared_ptr<T>& ptr, size_t size) {
        PoolManager& poolManager = PoolManager::getInstance();
        GPUMemoryPool* memoryPool = poolManager.getMemoryPool();
        uint64_t* new_d_data = nullptr;
        new_d_data = static_cast<T*>(memoryPool->allocate(size * sizeof(T)));  // 在 GPU 上分配内存

        // 使用 std::shared_ptr 包装 new_d_data，确保内存管理
        ptr = std::shared_ptr<uint64_t>(new_d_data, [memoryPool, size](uint64_t* p) {
            // 自定义删除器，用于在 shared_ptr 引用计数为 0 时释放内存
            memoryPool->deallocate(p, size * sizeof(T));
        });
    }


    template<typename KernelFunc>
    inline int getOptimalBlockSize(KernelFunc kernel) {
        int minGridSize, optimalBlockSize;

        // 使用cudaOccupancyMaxPotentialBlockSize计算达到最大占用率的最小网格大小和最佳线程块大小
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, 
                                        &optimalBlockSize, 
                                        kernel, 
                                        0, 
                                        0);

        return optimalBlockSize;
    }

    inline int getClosestBlockSize(int optimalBlockSize) {
        std::vector<int> blockSizes = {128, 256, 512, 1024};

        // 使用lambda和标准库中的min_element函数找到最近的块大小
        return *std::min_element(blockSizes.begin(), blockSizes.end(),
                                [optimalBlockSize](int a, int b) {
                                    return abs(a - optimalBlockSize) < abs(b - optimalBlockSize);
                                });
    }

    inline void print_value(uint64_t *value, int count);


    __global__ void fillTablePsi128(uint64_t psiinv, uint64_t q, uint64_t psiinvTable[], uint64_t nbit);
    __global__ void fillTablePsi128_root(uint64_t psi, uint64_t q, uint64_t *psiTable, int nx, int ny);
    __global__ void fillTablePsi128_root_n2(uint64_t psi, uint64_t q, uint64_t *psiTable, int nx, int ny);
    __global__ void fillTablePsi128_root_n12(uint64_t psi, uint64_t q, uint64_t *psiTable, int nx, int ny);

    __global__ void print_helper(uint64_t *input, size_t size);
    __global__ void set_poly_kernel(uint64_t *ori, uint64_t *dest, uint64_t coeff_count, uint64_t coeff_modulus_size);
    __global__ void set_uint_kernel(uint64_t *value, size_t uint64_count, uint64_t *result);
    __global__ void set_zero_poly_kernel(size_t coeff_count, size_t coeff_modulus_size, uint64_t *destination);

}