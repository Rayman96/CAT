#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
// #include <cuda_runtime.h>

// #include <cuda_runtime_api.h>

// #include <device_launch_parameters.h>
#include "kuint128.cuh"
namespace seal
{
    // --------------------------------------------------------------------------------------------------------------------------------------------------------
    // declarations for templated ntt functions

    template <unsigned l, unsigned n> // single kernel NTT
    __global__ void CTBasedNTTInnerSingle(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);

    template <unsigned l, unsigned n> // single kernel INTT
    __global__ void GSBasedINTTInnerSingle(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    template <unsigned l, unsigned n> // multi kernel NTT
    __global__ void CTBasedNTTInner(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);

    template <unsigned l, unsigned n> // multi kernel INTT
    __global__ void GSBasedINTTInner(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    template <unsigned l, unsigned n> // single kernel NTT batch
    __global__ void CTBasedNTTInnerSingle_batch(uint64_t a[], uint64_t psi_powers[], unsigned division);

    template <unsigned l, unsigned n> // single kernel INTT batch
    __global__ void GSBasedINTTInnerSingle_batch(uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    template <unsigned l, unsigned n> // multi kernel omg are you still reading this
    __global__ void CTBasedNTTInner_batch(uint64_t a[], uint64_t psi_powers[], unsigned division);

    template <unsigned l, unsigned n> // i'm not gonna write this one, figure this out on your own
    __global__ void GSBasedINTTInner_batch(uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // --------------------------------------------------------------------------------------------------------------------------------------------------------

    __device__ __forceinline__ void singleBarrett(k_uint128_t &a, uint64_t &q, uint64_t &mu, int &qbit)
    {
        k_uint128_t rx;

        rx = a >> (qbit - 2);

        mul64(rx.low, mu, rx);

        k_uint128_t::shiftr(rx, qbit + 2);

        mul64(rx.low, q, rx);

        sub128(a, rx);

        if (a.low >= q)
            a.low -= q;
    }

//     template <unsigned l, unsigned n>
//     __global__ void CTBasedNTTInnerSingle(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[])
//     {
//         register int local_tid = threadIdx.x;

//         extern __shared__ uint64_t shared_array[]; // declaration of shared_array

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         { // copying to shared from global
//             register int global_tid = local_tid + iteration_num * 1024;
//             shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
//         }

// #pragma unroll
//         for (int length = l; length < n; length *= 2)
//         { // for loops are required since we are handling all the remaining iterations in this kernel
//             register int step = (n / length) / 2;

// #pragma unroll
//             for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//             {
//                 register int global_tid = local_tid + iteration_num * 1024;
//                 register int psi_step = global_tid / step;
//                 register int target_index = psi_step * step * 2 + global_tid % step;
//                 ;

//                 psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

//                 register uint64_t psi = psi_powers[length + psi_step];

//                 register uint64_t first_target_value = shared_array[target_index];
//                 register k_uint128_t temp_storage =
//                     shared_array[target_index + step]; // this is for eliminating the possibility of overflow

//                 mul64(temp_storage.low, psi, temp_storage);

//                 singleBarrett(temp_storage, q, mu, qbit);
//                 register uint64_t second_target_value = temp_storage.low;

//                 register uint64_t target_result = first_target_value + second_target_value;

//                 target_result -= q * (target_result >= q);

//                 shared_array[target_index] = target_result;

//                 first_target_value += q * (first_target_value < second_target_value);

//                 shared_array[target_index + step] = first_target_value - second_target_value;
//             }

//             __syncthreads();
//         }

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         { // copy back to global from shared
//             register int global_tid = local_tid + iteration_num * 1024;
//             a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
//         }
//     }

//     template <unsigned l, unsigned n>
//     __global__ void GSBasedINTTInnerSingle(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[])
//     {
//         register int local_tid = threadIdx.x;

//         __shared__ uint64_t shared_array[2048]; // declaration of shared_array

//         register uint64_t q2 = (q + 1) >> 1;

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         { // copying to shared from global
//             register int global_tid = local_tid + iteration_num * 1024;
//             shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l)];
//         }

//         __syncthreads();

// #pragma unroll
//         for (int length = (n / 2); length >= l; length /= 2)
//         { // for loops are required since we are handling all the remaining iterations in this kernel
//             register int step = (n / length) / 2;

// #pragma unroll
//             for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//             {
//                 register int global_tid = local_tid + iteration_num * 1024;
//                 register int psi_step = global_tid / step;
//                 register int target_index = psi_step * step * 2 + global_tid % step;

//                 psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

//                 register uint64_t psiinv = psiinv_powers[length + psi_step];

//                 register uint64_t first_target_value = shared_array[target_index];
//                 register uint64_t second_target_value = shared_array[target_index + step];

//                 register uint64_t target_result = first_target_value + second_target_value;

//                 target_result -= q * (target_result >= q);

//                 shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

//                 first_target_value += q * (first_target_value < second_target_value);

//                 register k_uint128_t temp_storage = first_target_value - second_target_value;

//                 mul64(temp_storage.low, psiinv, temp_storage);

//                 singleBarrett(temp_storage, q, mu, qbit);

//                 register uint64_t temp_storage_low = temp_storage.low;

//                 shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
//             }

//             __syncthreads();
//         }

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         { // copy back to global from shared
//             register int global_tid = local_tid + iteration_num * 1024;
//             a[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
//         }
//     }

//     template <unsigned l, unsigned n>
//     __global__ void CTBasedNTTInner(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[])
//     {
//         // no shared memory - handling only one iteration in here

//         int length = l;

//         register int global_tid = blockIdx.x * 1024 + threadIdx.x;
//         register int step = (n / length) / 2;
//         register int psi_step = global_tid / step;
//         register int target_index = psi_step * step * 2 + global_tid % step;

//         register uint64_t psi = psi_powers[length + psi_step];

//         register uint64_t first_target_value = a[target_index];
//         register k_uint128_t temp_storage = a[target_index + step];

//         mul64(temp_storage.low, psi, temp_storage);

//         singleBarrett(temp_storage, q, mu, qbit);
//         register uint64_t second_target_value = temp_storage.low;

//         register uint64_t target_result = first_target_value + second_target_value;

//         target_result -= q * (target_result >= q);

//         a[target_index] = target_result;

//         first_target_value += q * (first_target_value < second_target_value);

//         a[target_index + step] = first_target_value - second_target_value;
//     }

//     template <unsigned l, unsigned n>
//     __global__ void GSBasedINTTInner(uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[])
//     {
//         // no shared memory - handling only one iteration in here

//         int length = l;

//         register int global_tid = blockIdx.x * 1024 + threadIdx.x;
//         register int step = (n / length) / 2;
//         register int psi_step = global_tid / step;
//         register int target_index = psi_step * step * 2 + global_tid % step;

//         register uint64_t psiinv = psiinv_powers[length + psi_step];

//         register uint64_t first_target_value = a[target_index];
//         register uint64_t second_target_value = a[target_index + step];

//         register uint64_t target_result = first_target_value + second_target_value;

//         target_result -= q * (target_result >= q);

//         register uint64_t q2 = (q + 1) >> 1;

//         target_result = (target_result >> 1) + q2 * (target_result & 1);

//         a[target_index] = target_result;

//         first_target_value += q * (first_target_value < second_target_value);

//         register k_uint128_t temp_storage = first_target_value - second_target_value;

//         mul64(temp_storage.low, psiinv, temp_storage);

//         singleBarrett(temp_storage, q, mu, qbit);

//         register uint64_t temp_storage_low = temp_storage.low;

//         temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

//         a[target_index + step] = temp_storage_low;
//     }

    __host__ void forwardNTTdouble(
        uint64_t *device_a, uint64_t *device_b, unsigned n, cudaStream_t &stream1, cudaStream_t &stream2, uint64_t q,
        uint64_t mu, int bit_length, uint64_t *psi_powers);

    __global__ void transpose(uint64_t *vec, int row, int col);
    __host__ void forwardNTT(
        uint64_t *device_a, unsigned n, cudaStream_t &stream1, uint64_t q, uint64_t mu, int bit_length,
        uint64_t *psi_powers);

    __host__ void inverseNTT(
        uint64_t *device_a, unsigned n, cudaStream_t &stream1, uint64_t q, uint64_t mu, int bit_length,
        uint64_t *psiinv_powers);

//     template <unsigned l, unsigned n>
//     __global__ void CTBasedNTTInnerSingle_batch(uint64_t a[], uint64_t psi_powers[], unsigned division)
//     {
//         unsigned index = blockIdx.y % division;
//         uint64_t q = q_cons[index];
//         uint64_t mu = mu_cons[index];
//         int qbit = q_bit_cons[index];

//         register int local_tid = threadIdx.x;

//         extern __shared__ uint64_t shared_array[];

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
//         }

// #pragma unroll
//         for (int length = l; length < n; length *= 2)
//         {
//             register int step = (n / length) / 2;

// #pragma unroll
//             for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//             {
//                 register int global_tid = local_tid + iteration_num * 1024;
//                 register int psi_step = global_tid / step;
//                 register int target_index = psi_step * step * 2 + global_tid % step;

//                 psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

//                 register uint64_t psi = psi_powers[length + psi_step + index * n];

//                 register uint64_t first_target_value = shared_array[target_index];
//                 register k_uint128_t temp_storage =
//                     shared_array[target_index + step]; // this is for eliminating the possibility of overflow

//                 mul64(temp_storage.low, psi, temp_storage);

//                 singleBarrett(temp_storage, q, mu, qbit);
//                 register uint64_t second_target_value = temp_storage.low;

//                 register uint64_t target_result = first_target_value + second_target_value;

//                 target_result -= q * (target_result >= q);

//                 shared_array[target_index] = target_result;

//                 first_target_value += q * (first_target_value < second_target_value);

//                 shared_array[target_index + step] = first_target_value - second_target_value;
//             }

//             __syncthreads();
//         }

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
//         }
//     }

//     template <unsigned l, unsigned n>
//     __global__ void GSBasedINTTInnerSingle_batch(uint64_t a[], uint64_t psiinv_powers[], unsigned division)
//     {
//         unsigned index = blockIdx.y % division;
//         uint64_t q = q_cons[index];
//         uint64_t mu = mu_cons[index];
//         int qbit = q_bit_cons[index];

//         register int local_tid = threadIdx.x;

//         __shared__ uint64_t shared_array[2048];

//         register uint64_t q2 = (q + 1) >> 1;

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             shared_array[global_tid] = a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n];
//         }

//         __syncthreads();

// #pragma unroll
//         for (int length = (n / 2); length >= l; length /= 2)
//         {
//             register int step = (n / length) / 2;

// #pragma unroll
//             for (int iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++)
//             {
//                 register int global_tid = local_tid + iteration_num * 1024;
//                 register int psi_step = global_tid / step;
//                 register int target_index = psi_step * step * 2 + global_tid % step;

//                 psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

//                 register uint64_t psiinv = psiinv_powers[length + psi_step + index * n];

//                 register uint64_t first_target_value = shared_array[target_index];
//                 register uint64_t second_target_value = shared_array[target_index + step];

//                 register uint64_t target_result = first_target_value + second_target_value;

//                 target_result -= q * (target_result >= q);

//                 shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);

//                 first_target_value += q * (first_target_value < second_target_value);

//                 register k_uint128_t temp_storage = first_target_value - second_target_value;

//                 mul64(temp_storage.low, psiinv, temp_storage);

//                 singleBarrett(temp_storage, q, mu, qbit);

//                 register uint64_t temp_storage_low = temp_storage.low;

//                 shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
//             }

//             __syncthreads();
//         }

// #pragma unroll
//         for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++)
//         {
//             register int global_tid = local_tid + iteration_num * 1024;
//             a[global_tid + blockIdx.x * (n / l) + blockIdx.y * n] = shared_array[global_tid];
//         }
//     }

//     template <unsigned l, unsigned n>
//     __global__ void CTBasedNTTInner_batch(uint64_t a[], uint64_t psi_powers[], unsigned division)
//     {
//         unsigned index = blockIdx.y % division;
//         uint64_t q = q_cons[index];
//         uint64_t mu = mu_cons[index];
//         int qbit = q_bit_cons[index];

//         int length = l;

//         register int global_tid = blockIdx.x * 1024 + threadIdx.x;
//         register int step = (n / length) / 2;
//         register int psi_step = global_tid / step;
//         register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

//         register uint64_t psi = psi_powers[length + psi_step + index * n];

//         register uint64_t first_target_value = a[target_index];
//         register k_uint128_t temp_storage = a[target_index + step];

//         mul64(temp_storage.low, psi, temp_storage);

//         singleBarrett(temp_storage, q, mu, qbit);
//         register uint64_t second_target_value = temp_storage.low;

//         register uint64_t target_result = first_target_value + second_target_value;

//         target_result -= q * (target_result >= q);

//         a[target_index] = target_result;

//         first_target_value += q * (first_target_value < second_target_value);

//         a[target_index + step] = first_target_value - second_target_value;
//     }

//     template <unsigned l, unsigned n>
//     __global__ void GSBasedINTTInner_batch(uint64_t a[], uint64_t psiinv_powers[], unsigned division)
//     {
//         unsigned index = blockIdx.y % division;
//         uint64_t q = q_cons[index];
//         uint64_t mu = mu_cons[index];
//         int qbit = q_bit_cons[index];

//         int length = l;

//         register int global_tid = blockIdx.x * 1024 + threadIdx.x;
//         register int step = (n / length) / 2;
//         register int psi_step = global_tid / step;
//         register int target_index = psi_step * step * 2 + global_tid % step + blockIdx.y * n;

//         register uint64_t psiinv = psiinv_powers[length + psi_step + index * n];

//         register uint64_t first_target_value = a[target_index];
//         register uint64_t second_target_value = a[target_index + step];

//         register uint64_t target_result = first_target_value + second_target_value;

//         target_result -= q * (target_result >= q);

//         register uint64_t q2 = (q + 1) >> 1;

//         target_result = (target_result >> 1) + q2 * (target_result & 1);

//         a[target_index] = target_result;

//         first_target_value += q * (first_target_value < second_target_value);

//         register k_uint128_t temp_storage = first_target_value - second_target_value;

//         mul64(temp_storage.low, psiinv, temp_storage);

//         singleBarrett(temp_storage, q, mu, qbit);

//         register uint64_t temp_storage_low = temp_storage.low;

//         temp_storage_low = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);

//         a[target_index + step] = temp_storage_low;
//     }

    __host__ void forwardNTT_batch(
        uint64_t *device_a, unsigned n, uint64_t *psi_powers, unsigned num, unsigned division);

    __host__ void inverseNTT_batch(
        uint64_t *device_a, unsigned n, uint64_t *psiinv_powers, unsigned num, unsigned division);

    // // --------------------------------------------------------------------------------------------------------------------------------------------------------
    // // explicit template instantiations
    // // all these are required for the program to compile

    // // n = 2048
    // template __global__ void CTBasedNTTInnerSingle<1, 2048>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void GSBasedINTTInnerSingle<1, 2048>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    // // n = 4096
    // template __global__ void CTBasedNTTInnerSingle<1, 4096>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void GSBasedINTTInner<1, 4096>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInnerSingle<2, 4096>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    // // n = 8192
    // template __global__ void CTBasedNTTInner<1, 8192>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInnerSingle<2, 8192>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void GSBasedINTTInner<1, 8192>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<2, 8192>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInnerSingle<4, 8192>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    // // n = 16384
    // template __global__ void CTBasedNTTInner<1, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInner<2, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInnerSingle<4, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void GSBasedINTTInner<1, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<2, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<4, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInnerSingle<8, 16384>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    // // n = 32768
    // template __global__ void CTBasedNTTInner<1, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInner<2, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInner<4, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void CTBasedNTTInnerSingle<8, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psi_powers[]);
    // template __global__ void GSBasedINTTInner<1, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<2, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<4, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInner<8, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);
    // template __global__ void GSBasedINTTInnerSingle<16, 32768>(
    //     uint64_t a[], uint64_t q, uint64_t mu, int qbit, uint64_t psiinv_powers[]);

    // // --------------------------------------------------------------------------------------------------------------------------------------------------------
    // // explicit template instantiations for batch ntt
    // // all these are required for the program to compile

    // // n = 2048
    // template __global__ void CTBasedNTTInnerSingle_batch<1, 2048>(
    //     uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void GSBasedINTTInnerSingle_batch<1, 2048>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // // n = 4096
    // template __global__ void CTBasedNTTInnerSingle_batch<1, 4096>(
    //     uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<1, 4096>(uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInnerSingle_batch<2, 4096>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // // n = 8192
    // template __global__ void CTBasedNTTInner_batch<1, 8192>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInnerSingle_batch<2, 8192>(
    //     uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<1, 8192>(uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<2, 8192>(uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInnerSingle_batch<4, 8192>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // // n = 16384
    // template __global__ void CTBasedNTTInner_batch<1, 16384>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInner_batch<2, 16384>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInnerSingle_batch<4, 16384>(
    //     uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<1, 16384>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<2, 16384>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<4, 16384>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInnerSingle_batch<8, 16384>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // // n = 32768
    // template __global__ void CTBasedNTTInner_batch<1, 32768>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInner_batch<2, 32768>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInner_batch<4, 32768>(uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void CTBasedNTTInnerSingle_batch<8, 32768>(
    //     uint64_t a[], uint64_t psi_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<1, 32768>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<2, 32768>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<4, 32768>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInner_batch<8, 32768>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);
    // template __global__ void GSBasedINTTInnerSingle_batch<16, 32768>(
    //     uint64_t a[], uint64_t psiinv_powers[], unsigned division);

    // --------------------------------------------------------------------------------------------------------------------------------------------------------
} // namespace seal