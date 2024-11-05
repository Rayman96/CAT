#include "helper.cuh"
#include "common.h"
#include "seal/util/rns.cuh"

namespace seal
{


    __global__ void fillTablePsi128(uint64_t psiinv, uint64_t q, uint64_t psiinvTable[], uint64_t nbit)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        psiinvTable[i] = modpow128(psiinv, bitReverse(i, nbit), q);
    }


    __global__ void fillTablePsi128_root(uint64_t psi, uint64_t q, uint64_t *psiTable, int nx, int ny)
    {
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idx = ix * nx + iy;
        if (ix < nx && iy < ny)
        {

            psiTable[idx] = modpow128(psi, 2 * ix * iy + iy, q);
        }
    }

    __global__ void fillTablePsi128_root_n12(uint64_t psi, uint64_t q, uint64_t *psiTable, int n, int p)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idx = row * p + col;
        if (row < n && col < p)
        {
            psiTable[idx] = modpow128(psi, 2 * row * col + col, q);
        }
    }

    __global__ void fillTablePsi128_root_n2(uint64_t psi, uint64_t q, uint64_t *psiTable, int nx, int ny)
    {
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idx = ix * ny + iy;
        if (ix < nx && iy < ny)
        {
            psiTable[idx] = modpow128(psi, 2 * ix * iy, q);
        }
    }

    __global__ void print_helper(uint64_t *input, size_t size)
    {
        uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size)
        {
            printf("kernel result[%d]: %llu\n", index, input[index]);
        }
    }

    __global__ void set_poly_kernel(uint64_t *ori, uint64_t *dest, uint64_t coeff_count, uint64_t coeff_modulus_size) {
        uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < coeff_count * coeff_modulus_size) {
            dest[index] = ori[index];
            index += blockDim.x * gridDim.x;
        }
    }

    __global__ void set_uint_kernel(uint64_t *value, size_t uint64_count, uint64_t *result)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((value == result) || !uint64_count)
        {
            return;
        }

        while (idx < uint64_count)
        {
            result[idx] = value[idx];
            idx += blockDim.x * gridDim.x;
        }
    }

    __global__ void set_zero_poly_kernel(size_t coeff_count, size_t coeff_modulus_size, uint64_t *destination)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < coeff_count * coeff_modulus_size)
        {
            destination[idx] = std::uint64_t(0);
            idx += blockDim.x * gridDim.x;
        }
    }

    inline void print_value(uint64_t *value, int count)
    {
        print_helper<<<1, count>>>(value, count);
    }

} // namespace seal