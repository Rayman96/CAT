#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

namespace seal
{
    namespace util
    {                
        __global__ void matrixMultiplication(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, uint64_t modulu, uint64_t ratio0,
            uint64_t ratio1)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m && col < p)
            {
                uint64_t sum = 0;
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[row * n + i], B[i * p + col], sum, modulu, ratio0, ratio1);
                }
                C[row * p + col] = sum;
            }
        }
        
        // 核函数，每一个线程计算矩阵中的一个元素。
        __global__ void elementMulMatrix(
            uint64_t *MatA, uint64_t *MatB, uint64_t *MatC, int n1, int n2, uint64_t modulu)
        {
            int col = threadIdx.x + blockDim.x * blockIdx.x;
            int row = threadIdx.y + blockDim.y * blockIdx.y;
            int idx = row * n2 + col;

            if (col < n2 && row < n1)
            {

                uint64_t operand = MatB[idx];
                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu, wide_quotient);
                quotient = wide_quotient[0];

                MatC[idx] = multiply_uint_mod_kernel(MatA[idx], quotient, operand, modulu);
            }
        }

        // 核函数，每一个线程计算矩阵中的一个元素。
        __global__ void elementMulroot(
            uint64_t *MatA, uint64_t *root, uint64_t *MatC, int n1, int n2, uint64_t modulu)
        {
            int col = threadIdx.x + blockDim.x * blockIdx.x;
            int row = threadIdx.y + blockDim.y * blockIdx.y;
            int idx = row * n2 + col;

            if (col < n2 && row < n1)
            {
                uint64_t operand = modpow128(root[0], 2 * row * col + col, modulu);
                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu, wide_quotient);
                quotient = wide_quotient[0];

                if (idx <3){
                    printf("cal root: %llu\n", operand);
                }


                MatC[idx] = multiply_uint_mod_kernel(MatA[idx], quotient, operand, modulu);
            }
        }

        // 用于ntt前两步综合计算
        __global__ void matrix_multi_elemul_merge(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D, int m, int n, int p, uint64_t modulu, uint64_t ratio0,
            uint64_t ratio1){
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m && col < p)
            {
                uint64_t sum = 0;
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[row * n + i], B[i * p + col], sum, modulu, ratio0, ratio1);
                }
                uint64_t operand = D[row * p + col];

                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu, wide_quotient);
                quotient = wide_quotient[0];

                C[row * p + col] = multiply_uint_mod_kernel(sum, quotient, operand, modulu);


            }
        }

        __global__ void matrix_multi_elemul_merge_batch_test(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D, int m, int n, int p, size_t modulu_size, uint64_t *modulu, uint64_t *ratio0,
            uint64_t *ratio1, uint64_t *elemul_root){
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m * modulu_size && col < p)
            {
                size_t modulu_idx = row / m;
                row = row % m;
                int64_t coeff_count = m * p;
                uint64_t sum = 0;

                const int64_t output_index = modulu_idx * coeff_count + row * p + col;

                extern __shared__ uint64_t modulu_value;
                modulu_value = modulu[modulu_idx];
                extern __shared__ uint64_t ratio0_value;
                ratio0_value = ratio0[modulu_idx];
                extern __shared__ uint64_t ratio1_value;
                ratio1_value = ratio1[modulu_idx];

                extern __shared__ int64_t n1n1_shift;
                n1n1_shift = modulu_idx * m * n;
                extern __shared__ int64_t n1n2_shift;
                n1n2_shift = modulu_idx * coeff_count;


#pragma unroll
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[n1n1_shift + row * n + i], B[n1n2_shift + i * p + col], 
                        sum, 
                        modulu_value, ratio0_value, ratio1_value);
                }


                uint64_t operand = D[output_index];

                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu_value, wide_quotient);
                quotient = wide_quotient[0];

                C[output_index] = multiply_uint_mod_kernel(sum, quotient, operand, modulu_value);




            }
        }

        __global__ void matrix_multi_elemul_merge_batch_key_switch(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D, int m, int n, int p, 
            size_t modulu_size, 
            uint64_t *modulu, 
            uint64_t *ratio0,
            uint64_t *ratio1, uint64_t *elemul_root){
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m * modulu_size && col < p)
            {
                size_t modulu_idx = row / m;
                row = row % m;
                int64_t coeff_count = m * p;
                uint64_t sum = 0;

                const int64_t output_index =  row * p + col;

                uint64_t modulu_value = modulu[0];
                 uint64_t ratio0_value = ratio0[0];
                 uint64_t ratio1_value = ratio1[0];

                 int64_t n1n2_shift = modulu_idx * coeff_count;

#pragma unroll
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[row * n + i], B[n1n2_shift + i * p + col], 
                        sum, 
                        modulu_value, ratio0_value, ratio1_value);
                }

                uint64_t operand = D[output_index];

                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu_value, wide_quotient);
                quotient = wide_quotient[0];

                C[modulu_idx *coeff_count + output_index] = multiply_uint_mod_kernel(sum, quotient, operand, modulu_value);

            }
        }

        __global__ void matrix_multi_elemul_merge_batch(uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, size_t modulu_size, uint64_t *modulu, uint64_t *ratio0,
            uint64_t *ratio1, uint64_t *elemul_root){
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m * modulu_size && col < p)
            {

                size_t modulu_idx = row / m;
                row = row % m;
                size_t coeff_count = m * p;

                uint64_t sum = 0;
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[modulu_idx * m * n + row * n + i], B[modulu_idx * coeff_count + i * p + col], 
                        sum, 
                        modulu[modulu_idx], ratio0[modulu_idx], ratio1[modulu_idx]);
                }
                // C[row * p + col] = sum;

                uint64_t operand = modpow128(elemul_root[modulu_idx], 2 * row * col + col, modulu[modulu_idx]);

                uint64_t quotient = 0;
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };

                divide_uint128_inplace_kernel(wide_coeff, modulu[modulu_idx], wide_quotient);
                quotient = wide_quotient[0];

                C[modulu_idx * coeff_count + row * p + col] = multiply_uint_mod_kernel(sum, quotient, operand, modulu[modulu_idx]);


            }
        }

        // 专用于ntt算法最后一步的乘法11
        __global__ void matrix_multi_transpose(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, uint64_t modulu, int bit_count,
            uint64_t ratio0, uint64_t ratio1)
        {
            uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
            uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m && col < p)
            {
                uint64_t sum = 0;
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[row * n + i], B[i * p + col], sum, modulu, ratio0, ratio1);
                }
                uint64_t idx = reverse_bits_kernel(col * m + row, bit_count);
                C[idx] = sum;
            }
        }

        __global__ void matrix_multi_transpose_batch(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, size_t modulu_size, uint64_t *modulu, int *bit_count,
            uint64_t *ratio0, uint64_t *ratio1)
        {
            uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
            uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m * modulu_size && col < p)
            {
                size_t modulu_idx = row / m;
                row = row % m;
                size_t coeff_count = m * n;

                uint64_t sum = 0;

#pragma unroll
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[ modulu_idx * coeff_count + row * n + i], 
                        B[ modulu_idx * n * p + i * p + col], 
                        sum, 
                        modulu[modulu_idx], ratio0[modulu_idx], ratio1[modulu_idx]);
                }
                uint64_t idx = reverse_bits_kernel(col * m + row, bit_count[modulu_idx]);
                C[ modulu_idx * coeff_count + idx] = sum;
            }
        }

        __global__ void matrix_multi_transpose_batch_key_swtich(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, size_t modulu_size, uint64_t *modulu, int *bit_count,
            uint64_t *ratio0, uint64_t *ratio1)
        {
            uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
            uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < m * modulu_size && col < p)
            {
                size_t modulu_idx = row / m;
                row = row % m;
                size_t coeff_count = m * n;

                uint64_t sum = 0;

#pragma unroll
                for (int i = 0; i < n; ++i)
                {
                    sum = multiply_add_uint_mod_kernel(
                        A[ modulu_idx * coeff_count + row * n + i], 
                        B[i * p + col], 
                        sum, 
                        modulu[0], ratio0[0], ratio1[0]);
                }
                uint64_t idx = reverse_bits_kernel(col * m + row, bit_count[0]);
                C[ modulu_idx * coeff_count + idx] = sum;
            }
        }


        void ntt_w1w2_helper(uint64_t *matrix_n1, uint64_t *input, uint64_t *result, uint64_t *matrix_n12, int n1, int n2, int modulu_size, uint64_t *modulu_value,
        uint64_t *ratio_0, uint64_t *ratio_1, uint64_t *root){
            dim3 block(16, 16);
            dim3 grid1_batch((n2 - 1) / block.x + 1, (n1* modulu_size - 1) / block.y + 1);

            matrix_multi_elemul_merge_batch_test<<<grid1_batch, block>>>(
                matrix_n1, 
                input, 
                result,
                matrix_n12,
                n1, n1, n2,
                modulu_size,
                modulu_value,
                ratio_0,
                ratio_1,
                root
            );
        }

        void ntt_v3(const SEALContext &context, parms_id_type parms_id, uint64_t *input, size_t modulu_size, int modulu_shift, cudaStream_t ntt_stream){
            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            size_t coeff_count = parms.poly_modulus_degree();
            auto &coeff_modulus = parms.coeff_modulus();

            uint64_t *d_coeff_modulus = parms.d_coeff_modulus_value() + modulu_shift;
            uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0() + modulu_shift;
            uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1() + modulu_shift;
            int *d_bit_count = context_data.d_bit_count() + modulu_shift;

            size_t c_vec_size = coeff_count *modulu_size;
            context.ensure_ntt_size(c_vec_size);
            uint64_t *ntt_temp_result = context.ntt_temp();

            std::pair<int, int> split_result = context_data.split_degree();
            int n1 = split_result.first, n2 = split_result.second;

            uint64_t *d_roots = context_data.d_roots() + modulu_shift;
            uint64_t *d_root_matrix_n1 = context_data.d_root_matrix_n1() + modulu_shift * n1 * n1;
            uint64_t *d_root_matrix_n2 = context_data.d_root_matrix_n2() + modulu_shift * n2 * n2;
            uint64_t *d_root_matrix_n12 = context_data.d_root_matrix_n12() + modulu_shift * n1 * n2;

            dim3 block(16, 16);
            dim3 grid1_batch((n2 - 1) / block.x + 1, (n1* modulu_size - 1) / block.y + 1);
            dim3 grid1((n2 - 1) / block.x + 1, (n1 - 1) / block.y + 1);

            matrix_multi_elemul_merge_batch_test<<<grid1_batch, block, 0, ntt_stream>>>(
                d_root_matrix_n1, 
                input, 
                ntt_temp_result,
                d_root_matrix_n12,
                n1, n1, n2,
                modulu_size,
                d_coeff_modulus,
                d_coeff_modulus_ratio_0,
                d_coeff_modulus_ratio_1,
                d_roots
            );

            matrix_multi_transpose_batch<<<grid1_batch, block, 0, ntt_stream>>>(
                    ntt_temp_result, 
                    d_root_matrix_n2, 
                    input, 
                    n1, n2, n2, 
                    modulu_size,
                    d_coeff_modulus, 
                    d_bit_count, 
                    d_coeff_modulus_ratio_0,
                    d_coeff_modulus_ratio_1);

        }

        void ntt_v3_key_switch(const SEALContext &context, parms_id_type parms_id, uint64_t *input, size_t modulu_size, cudaStream_t ntt_stream, int modulu_shift){
            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            size_t coeff_count = parms.poly_modulus_degree();
            auto &coeff_modulus = parms.coeff_modulus();

            uint64_t *d_coeff_modulus = parms.d_coeff_modulus_value() + modulu_shift;
            uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0() + modulu_shift;
            uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1() + modulu_shift;
            int *d_bit_count = context_data.d_bit_count() + modulu_shift;

            size_t c_vec_size = coeff_count *modulu_size;
            context.ensure_ntt_size(c_vec_size);
            uint64_t *ntt_temp_result = context.ntt_temp();

            std::pair<int, int> split_result = context_data.split_degree();
            int n1 = split_result.first, n2 = split_result.second;

            uint64_t *d_roots = context_data.d_roots() + modulu_shift;
            uint64_t *d_root_matrix_n1 = context_data.d_root_matrix_n1() + modulu_shift * n1 * n1;
            uint64_t *d_root_matrix_n2 = context_data.d_root_matrix_n2() + modulu_shift * n2 * n2;
            uint64_t *d_root_matrix_n12 = context_data.d_root_matrix_n12() + modulu_shift * n1 * n2;

            dim3 block(16, 16);
            dim3 grid1_batch((n2 - 1) / block.x + 1, (n1* modulu_size - 1) / block.y + 1);
            dim3 grid((n2 - 1) / block.x + 1, (n1 - 1) / block.y + 1);


            matrix_multi_elemul_merge_batch_key_switch<<<grid1_batch, block, 0, ntt_stream>>>(
                d_root_matrix_n1, 
                input, 
                ntt_temp_result,
                d_root_matrix_n12,
                n1, n1, n2,
                modulu_size,
                d_coeff_modulus,
                d_coeff_modulus_ratio_0,
                d_coeff_modulus_ratio_1,
                d_roots
            );

            matrix_multi_transpose_batch_key_swtich<<<grid1_batch, block, 0, ntt_stream>>>(
                    ntt_temp_result, 
                    d_root_matrix_n2, 
                    input, 
                    n1, n2, n2, 
                    modulu_size,
                    d_coeff_modulus, 
                    d_bit_count, 
                    d_coeff_modulus_ratio_0,
                    d_coeff_modulus_ratio_1);

        }

        void ntt_v1(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  size_t modulu_size, int modulu_shift, bool fix_root){
            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            size_t coeff_count = parms.poly_modulus_degree();
            auto &coeff_modulus = parms.coeff_modulus();
            uint64_t *d_root_power = context_data.d_root_powers() + modulu_shift * coeff_count;
            
            const int stream_num = context.num_streams();
            cudaStream_t *ntt_steam = context.stream_context();

            int cal_shift = 0;

            k_uint128_t mu;
            for (int i = 0; i < modulu_size; ++i)
            {
                if (!fix_root){
                    cal_shift = i;
                }

                int bit_count = coeff_modulus[modulu_shift + cal_shift].bit_count();
                mu = k_uint128_t::exp2(bit_count * 2);

                forwardNTT(
                    input + i * coeff_count, 
                    coeff_count, 
                    ntt_steam[i % stream_num], 
                    coeff_modulus[modulu_shift + cal_shift].value(), 
                    (mu / coeff_modulus[modulu_shift + cal_shift].value()).low,
                    bit_count, 
                    d_root_power + cal_shift * coeff_count);
            }
        }

        void ntt_v1_single(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  int modulu_shift, cudaStream_t stream){
            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            size_t coeff_count = parms.poly_modulus_degree();
            auto &coeff_modulus = parms.coeff_modulus();
            uint64_t *d_root_power = context_data.d_root_powers() + modulu_shift * coeff_count;
            
            uint64_t modulu_value = coeff_modulus[modulu_shift ].value();
            uint64_t ratio_0 = coeff_modulus[modulu_shift ].const_ratio().data()[0];
            uint64_t ratio_1 = coeff_modulus[modulu_shift ].const_ratio().data()[1];
            int bit_count = coeff_modulus[modulu_shift ].bit_count();

            k_uint128_t mu = k_uint128_t::exp2(bit_count * 2);
            uint64_t    temp_mu = (mu / modulu_value).low;

            forwardNTT(
                input, coeff_count, stream, modulu_value, temp_mu,
                bit_count, d_root_power );
        }

        void intt_v1_single(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  int modulu_shift, cudaStream_t stream){
            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            size_t coeff_count = parms.poly_modulus_degree();
            auto &coeff_modulus = parms.coeff_modulus();
            uint64_t *d_root_power = context_data.d_inv_root_powers() + modulu_shift * coeff_count;
            
            uint64_t modulu_value = coeff_modulus[modulu_shift ].value();
            int bit_count = coeff_modulus[modulu_shift ].bit_count();

            k_uint128_t mu = k_uint128_t::exp2(bit_count * 2);
            uint64_t    temp_mu = (mu / modulu_value).low;

            inverseNTT(
                input, coeff_count, stream, modulu_value, temp_mu,
                bit_count, d_root_power );
        }

        void ntt_v3_single(uint64_t *input, uint64_t *matrix_n1, uint64_t *matrix_n2,uint64_t *matrix_n12, uint64_t modulu, uint64_t ratio0, uint64_t ratio1, 
                            uint64_t root, int bit, std::pair<int, int> split_result, uint64_t *ntt_temp){
            int n1 = split_result.first, n2 = split_result.second;
            dim3 block(16, 16);
            dim3 grid((n2 - 1) / block.x + 1, (n1 - 1) / block.y + 1);

            matrix_multi_elemul_merge<<<grid, block>>>(
                                                        matrix_n1, 
                                                        input, 
                                                        ntt_temp,
                                                        matrix_n12,
                                                        n1, n1, n2,
                                                        modulu,
                                                        ratio0,
                                                        ratio1
                                                        );

            matrix_multi_transpose<<<grid, block>>>(ntt_temp, matrix_n2, input, n1, n2, n2, modulu, bit, ratio0, ratio1);
        } 

    }
}