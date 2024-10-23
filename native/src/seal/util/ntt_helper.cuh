#pragma once

#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/scalingvariant.h"
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
            uint64_t ratio1);

        __global__ void matrix_multi_transpose(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, uint64_t modulu, int bit_count,
            uint64_t ratio0, uint64_t ratio1);

        __global__ void elementMulMatrix(
            uint64_t *MatA, uint64_t *MatB, uint64_t *MatC, int n1, int n2, uint64_t modulu, uint64_t *ratio1);

        __global__ void matrix_multi_elemul_merge(uint64_t *A, uint64_t *B, uint64_t *C,uint64_t *D, int m, int n, int p, uint64_t modulu, uint64_t ratio0,
            uint64_t ratio1);
        
        __global__ void matrix_multi_elemul_merge_batch(uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, size_t modulu_size, uint64_t *modulu, uint64_t *ratio0,
            uint64_t *ratio1, uint64_t *elemul_root);

        __global__ void matrix_multi_elemul_merge_batch_test(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t *D, int m, int n, int p, size_t modulu_size, uint64_t *modulu, uint64_t *ratio0,
            uint64_t *ratio1, uint64_t *elemul_root);

        __global__ void matrix_multi_transpose_batch(
            uint64_t *A, uint64_t *B, uint64_t *C, int m, int n, int p, size_t modulu_size, uint64_t *modulu, int *bit_count,
            uint64_t *ratio0, uint64_t *ratio1);

        void ntt_v3(const SEALContext &context, parms_id_type parms_id, uint64_t *input, size_t modulu_size, int modulu_shift=0, cudaStream_t ntt_stream=0);

        void ntt_v3_key_switch(const SEALContext &context, parms_id_type parms_id, uint64_t *input, size_t modulu_size, cudaStream_t ntt_stream, int modulu_shift);

        void ntt_v1(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  size_t modulu_size, int modulu_shift=0, bool fix_root=false);

        void ntt_v1_single(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  int modulu_shift, cudaStream_t stream=0);
        
        void intt_v1_single(const SEALContext &context, parms_id_type parms_id, uint64_t *input,  int modulu_shift, cudaStream_t stream=0);

        void ntt_v3_single(uint64_t *input, uint64_t *matrix_n1, uint64_t *matrix_n2, uint64_t *matrix_n12, uint64_t modulu, uint64_t ratio0, uint64_t ratio1, 
                                    uint64_t root, int bit, std::pair<int, int> split_result, uint64_t *ntt_temp); 
    }
}