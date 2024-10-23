// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/uintarith.cuh"
#include "seal/util/uintcore.h"
#include <cstdint>
#include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cudnn.h>

#ifdef SEAL_USE_INTEL_HEXL
#include "hexl/hexl.hpp"
#endif

using namespace std;

namespace seal
{
    namespace util
    {

        __global__ void add_poly_coeffmod_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, uint64_t coeff_modulus_size, uint64_t *modulus_value, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size)
            {
                uint64_t sum = operand1[idx] + operand2[idx];
                result[idx] = sum >= modulus_value[idx / coeff_count] ? sum - modulus_value[idx / coeff_count] : sum;
                idx += blockDim.x * gridDim.x;
            }
        }


        __global__ void add_poly_coeffmod_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, uint64_t modulus_value, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                uint64_t sum = operand1[idx] + operand2[idx];
                result[idx] = sum >= modulus_value ? sum - modulus_value : sum;
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void add_poly_coeffmod_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t encrypted_size, size_t coeff_count,
            size_t coeff_modulus_size, uint64_t *modulus_value, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < encrypted_size * coeff_count * coeff_modulus_size)
            {
                size_t idx_1 = (idx / coeff_count) % coeff_modulus_size * coeff_count + (idx % coeff_count);
                uint64_t sum = operand1[idx_1] + operand2[idx];
                result[idx_1] = sum >= modulus_value[(idx / coeff_count) % coeff_modulus_size]
                                    ? sum - modulus_value[(idx / coeff_count) % coeff_modulus_size]
                                    : sum;
                idx += blockDim.x * gridDim.x;
            }
        }


        __global__ void add_poly_scalar_kernel(
            uint64_t *operand1, uint64_t *output, uint64_t scalar, size_t coeff_count,
            uint64_t modulus_value){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                uint64_t sum = operand1[idx] + scalar;
                output[idx] = sum >= modulus_value ? sum - modulus_value : sum;
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void negate_poly_coeffmod_cuda(
            uint64_t *poly, uint64_t *modulus, uint64_t *result, std::size_t coeff_count, std::size_t coeff_modulus_size, std::size_t size){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t modulus_index = (index / coeff_count) % coeff_modulus_size;

            while (index < size) {
                std::int64_t non_zero = (poly[index] != 0);

                result[index] = (modulus[modulus_index] - poly[index]) & static_cast<std::uint64_t>(-non_zero);
                index += blockDim.x * gridDim.x;
            }

        }

        template <typename T>
        __global__ void eltwiseMultModKernel(T *result, const T *operand1, const T *operand2, const T modulus, const std::size_t size) {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < size) {
                result[index] = modMultiply(operand1[index], operand2[index], modulus);
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void eltwiseAddModKernel(uint64_t *result, const uint64_t *operand1, const uint64_t *operand2, const uint64_t *modulus, const std::size_t coeff_count, const std::size_t coeff_modulus_size) {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < coeff_count * coeff_modulus_size) {
                result[index] = operand1[index] + operand2[index] >= modulus[index / coeff_count] ? operand1[index] + operand2[index] - modulus[index / coeff_count]  : operand1[index] + operand2[index];
                index += blockDim.x * gridDim.x;

            }
        }

        template <typename T>
        __global__ void eltwiseSubModKernel(T *result, const T *operand1, const T *operand2, const T *modulus, const std::size_t coeff_count, const std::size_t coeff_modulus_size) {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < coeff_count * coeff_modulus_size) {
                // result[index] = operand1[index] - operand2[index] >= modulus[index / coeff_count] ? operand1[index] + operand2[index] - modulus[index / coeff_count]  : operand1[index] + operand2[index];
                result[index] = operand1[index] > operand2[index] ? operand1[index] - operand2[index] : operand1[index] + modulus[index / coeff_count] - operand2[index];
                index += blockDim.x * gridDim.x;

            }
        }

        template <typename T>
        __global__ void eltwiseAddModScalarKernel(T *result, const T *operand1, const T operand2, const T modulus, const std::size_t size) {
            register T scalar = operand2;

            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < size) {
                result[index] = operand1[index] + scalar >= modulus ? operand1[index] + scalar - modulus : operand1[index] + scalar;
                index += blockDim.x * gridDim.x;
            }
        }

        template <typename T>
        __global__ void eltwiseSubModKernel(T *result, const T *operand1, const T *operand2, const T modulus, const std::size_t size) {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < size) {
                result[index] = operand1[index] > operand2[index] ? operand1[index] - operand2[index] : operand1[index] + modulus - operand2[index];
                index += blockDim.x * gridDim.x;
            }
        }

        template <typename T>
        __global__ void eltwiseSubModScalarKernel(T *result, const T *operand1, const T operand2, const T modulus, const std::size_t size) {
            register T scalar = operand2;

            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < size) {
                result[index] = operand1[index] > scalar ? operand1[index] - scalar : operand1[index] + modulus - scalar;
                index += blockDim.x * gridDim.x;
            }
        }

        __global__  void dyadic_product_coeffmod_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, uint64_t modulus, uint64_t modulus_ratio_0,
            uint64_t modulus_ratio_1, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                const uint64_t modulus_value = modulus;
                const uint64_t const_ratio_0 = modulus_ratio_0;
                const uint64_t const_ratio_1 = modulus_ratio_1;

                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64_kernel2(operand1[idx], operand2[idx], z);
                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64_kernel(z[0], const_ratio_0, &carry);
                multiply_uint64_kernel2(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64_kernel(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64_kernel2(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64_kernel(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;
                result[idx] = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void dyadic_product_coeffmod_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, size_t coeff_modulus_size,
            size_t encrypted_ntt_size, uint64_t *modulus, uint64_t *modulus_ratio_0, uint64_t *modulus_ratio_1,
            uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size * encrypted_ntt_size)
            {
                const int modulu_index = (idx / coeff_count) % coeff_modulus_size;
                const size_t operand2_idx = idx % (coeff_count * coeff_modulus_size);
                const uint64_t modulus_value = modulus[modulu_index];
                const uint64_t const_ratio_0 = modulus_ratio_0[modulu_index];
                const uint64_t const_ratio_1 = modulus_ratio_1[modulu_index];

                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64_kernel2(operand1[idx], operand2[operand2_idx], z);
                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64_kernel(z[0], const_ratio_0, &carry);
                multiply_uint64_kernel2(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64_kernel(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64_kernel2(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64_kernel(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;
                result[idx] = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void dyadic_product_coeffmod_kernel_two_modulu(
            uint64_t *operand1, uint64_t *operand2, size_t encrypted_size, size_t coeff_count,
            size_t coeff_modulus_size, size_t coeff_modulus_size2, uint64_t *modulus, uint64_t *modulus_ratio_0,
            uint64_t *modulus_ratio_1, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size * encrypted_size)
            {
                const size_t idx_modulus = (idx / coeff_count) % coeff_modulus_size;
                const size_t idx_operand2 =
                    (idx / (coeff_count * coeff_modulus_size)) * (coeff_count * coeff_modulus_size2) +
                    idx_modulus * coeff_count + (idx % coeff_count);
                const uint64_t modulus_value = modulus[idx_modulus];
                const uint64_t const_ratio_0 = modulus_ratio_0[idx_modulus];
                const uint64_t const_ratio_1 = modulus_ratio_1[idx_modulus];

                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64_kernel2(operand1[idx], operand2[idx_operand2], z);
                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64_kernel(z[0], const_ratio_0, &carry);
                multiply_uint64_kernel2(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64_kernel(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64_kernel2(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64_kernel(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;
                result[idx] = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

                idx += blockDim.x * gridDim.x;
            }
        }

        template <typename T>
        void EltwiseAddScalarMod(T *result, const T *operand1, const T operand2, const T modulus, const std::size_t size) {
            // 将输入数据 operand1 和 operand2 复制到 GPU 设备内存
            T *d_operand1; 
            checkCudaErrors(cudaMalloc((void**) &d_operand1, sizeof(T) * size));
            checkCudaErrors(cudaMemcpy(d_operand1, operand1, sizeof(T) * size, cudaMemcpyHostToDevice));

            // 为结果数组 result 在 GPU 设备内存上分配空间并初始化为0
            T *d_result;
            checkCudaErrors(cudaMalloc((void**) &d_result, sizeof(T) * size));
            checkCudaErrors(cudaMemset(d_result, 0, sizeof(T) * size));

            // 在 GPU 上执行逐元素加法，并将结果写入 d_result 数组
            const std::size_t threadsPerBlock = 256;
            const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            seal::util::eltwiseAddModScalarKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_result, d_operand1, operand2, modulus, size);
            cudaDeviceSynchronize();
            // 将计算结果从 GPU 设备内存复制回主机内存
            checkCudaErrors(cudaMemcpy(result, d_result, sizeof(T) * size, cudaMemcpyDeviceToHost));

            // 释放 GPU 设备内存
            checkCudaErrors(cudaFree(d_operand1));
            checkCudaErrors(cudaFree(d_result));
        }

        template <typename T>
        void EltwiseSubScalarMod(T *result, const T *operand1, const T operand2, const T modulus, const std::size_t size) {
            // 将输入数据 operand1 和 operand2 复制到 GPU 设备内存
            T *d_operand1; 
            checkCudaErrors(cudaMalloc((void**) &d_operand1, sizeof(T) * size));
            checkCudaErrors(cudaMemcpy(d_operand1, operand1, sizeof(T) * size, cudaMemcpyHostToDevice));

            // 为结果数组 result 在 GPU 设备内存上分配空间并初始化为0
            T *d_result;
            checkCudaErrors(cudaMalloc((void**) &d_result, sizeof(T) * size));
            checkCudaErrors(cudaMemset(d_result, 0, sizeof(T) * size));

            // 在 GPU 上执行逐元素加法，并将结果写入 d_result 数组
            const std::size_t threadsPerBlock = 256;
            const std::size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            eltwiseSubModScalarKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_result, d_operand1, operand2, modulus, size);
            // cudaDeviceSynchronize();
            // 将计算结果从 GPU 设备内存复制回主机内存
            checkCudaErrors(cudaMemcpy(result, d_result, sizeof(T) * size, cudaMemcpyDeviceToHost));

            // 释放 GPU 设备内存
            checkCudaErrors(cudaFree(d_operand1));
            checkCudaErrors(cudaFree(d_result));
        }

        void modulo_poly_coeffs(ConstCoeffIter poly, std::size_t coeff_count, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseReduceMod(result, poly, coeff_count, modulus.value(), modulus.value(), 1);
#else
            SEAL_ITERATE(
                iter(poly, result), coeff_count, [&](auto I) { get<1>(I) = barrett_reduce_64(get<0>(I), modulus); });
#endif
        }

        void add_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
           const uint64_t modulus_value = modulus.value();

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseAddMod(&result[0], &operand1[0], &operand2[0], coeff_count, modulus_value);
#else
            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                std::uint64_t sum = get<0>(I) + get<1>(I);
                get<2>(I) = SEAL_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
            });
#endif
        }

        void add_poly_coeffmod_cuda(uint64_t *operand1, uint64_t *operand2, std::size_t coeff_modulus_size, std::size_t coeff_count, uint64_t* modulus,
            uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            // EltwiseAddMod(&result[0], &operand1[0], &operand2[0],  modulus.value(), coeff_count);
            const std::size_t threads_per_block = 256;
            const std::size_t blocksPerGrid = (coeff_count*coeff_modulus_size + threads_per_block - 1) / threads_per_block;
            eltwiseAddModKernel<<<blocksPerGrid, threads_per_block>>>(result, operand1, operand2, modulus, coeff_count, coeff_modulus_size);
            

        }

        void sub_poly_coeffmod_cuda(uint64_t *operand1, uint64_t *operand2, std::size_t coeff_modulus_size, std::size_t coeff_count, uint64_t* modulus,
            uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            // EltwiseAddMod(&result[0], &operand1[0], &operand2[0],  modulus.value(), coeff_count);
            const std::size_t threads_per_block = 256;
            const std::size_t blocksPerGrid = (coeff_count*coeff_modulus_size + threads_per_block - 1) / threads_per_block;
            eltwiseSubModKernel<uint64_t><<<blocksPerGrid, threads_per_block>>>(result, operand1, operand2, modulus, coeff_count, coeff_modulus_size);
            

        }

        void sub_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif

            const uint64_t modulus_value = modulus.value();
#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseSubMod(result, operand1, operand2, coeff_count, modulus_value);
#else
            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                unsigned long long temp_result;
                std::int64_t borrow = sub_uint64(get<0>(I), get<1>(I), &temp_result);
                get<2>(I) = temp_result + (modulus_value & static_cast<std::uint64_t>(-borrow));
            });
#endif
        }

        void add_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif
           EltwiseAddScalarMod(&result[0], &poly[0], scalar, modulus.value(), coeff_count);
        }

        void sub_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif
            EltwiseSubScalarMod(&result[0], &poly[0], scalar, modulus.value(), coeff_count);
        }

        void multiply_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, MultiplyUIntModOperand scalar, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseFMAMod(&result[0], &poly[0], scalar.operand, nullptr, coeff_count, modulus.value(), 8);
#else
            // EltwiseFMAMod(&result[0], &poly[0], scalar.operand, nullptr, coeff_count, modulus.value(), 8);
            // printf("666666");
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = multiply_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void dyadic_product_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1)
            {
                throw invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw invalid_argument("operand2");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (coeff_count == 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
            const uint64_t modulus_value = modulus.value();
            const uint64_t const_ratio_0 = modulus.const_ratio()[0];
            const uint64_t const_ratio_1 = modulus.const_ratio()[1];

            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
                // Reduces z using base 2^64 Barrett reduction
                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64(get<0>(I), get<1>(I), z);

                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64(z[0], const_ratio_0, &carry);
                multiply_uint64(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;

                // Claim: One more subtraction is enough
                get<2>(I) = SEAL_COND_SELECT(tmp3 >= modulus_value, tmp3 - modulus_value, tmp3);
            });
        }

        uint64_t poly_infty_norm_coeffmod(ConstCoeffIter operand, size_t coeff_count, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (!operand && coeff_count > 0)
            {
                throw invalid_argument("operand");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
            // Construct negative threshold (first negative modulus value) to compute absolute values of coeffs.
            uint64_t modulus_neg_threshold = (modulus.value() + 1) >> 1;

            // Mod out the poly coefficients and choose a symmetric representative from
            // [-modulus,modulus). Keep track of the max.
            uint64_t result = 0;
            SEAL_ITERATE(operand, coeff_count, [&](auto I) {
                uint64_t poly_coeff = barrett_reduce_64(I, modulus);
                if (poly_coeff >= modulus_neg_threshold)
                {
                    poly_coeff = modulus.value() - poly_coeff;
                }
                if (poly_coeff > result)
                {
                    result = poly_coeff;
                }
            });

            return result;
        }

        void negacyclic_shift_poly_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, size_t shift, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly)
            {
                throw invalid_argument("poly");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (poly == result)
            {
                throw invalid_argument("result cannot point to the same value as poly");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (util::get_power_of_two(static_cast<uint64_t>(coeff_count)) < 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (shift >= coeff_count)
            {
                throw invalid_argument("shift");
            }
#endif
            // Nothing to do
            if (shift == 0)
            {
                set_uint(poly, coeff_count, result);
                return;
            }

            uint64_t index_raw = shift;
            uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
            for (size_t i = 0; i < coeff_count; i++, poly++, index_raw++)
            {
                uint64_t index = index_raw & coeff_count_mod_mask;
                if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly)
                {
                    result[index] = *poly;
                }
                else
                {
                    result[index] = modulus.value() - *poly;
                }
            }
        }
    } // namespace util
} // namespace seal
