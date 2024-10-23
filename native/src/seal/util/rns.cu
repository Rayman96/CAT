// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/rns.cuh"
#include "seal/util/uintarithmod.cuh"
#include "seal/util/uintarithsmallmod.cuh"
#include "seal/util/scalingvariant.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <device_launch_parameters.h>

using namespace std;
using namespace seal::util;

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);
namespace seal
{

    namespace util
    {
        RNSBase::RNSBase(const vector<Modulus> &rnsbase, MemoryPoolHandle pool)
            : pool_(move(pool)), size_(rnsbase.size())
        {
            if (!size_)
            {
                throw invalid_argument("rnsbase cannot be empty");
            }
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }

            for (size_t i = 0; i < rnsbase.size(); i++)
            {
                // The base elements cannot be zero
                if (rnsbase[i].is_zero())
                {
                    throw invalid_argument("rnsbase is invalid");
                }

                for (size_t j = 0; j < i; j++)
                {
                    // The base must be coprime
                    if (!are_coprime(rnsbase[i].value(), rnsbase[j].value()))
                    {
                        throw invalid_argument("rnsbase is invalid");
                    }
                }
            }

            // Base is good; now copy it over to rnsbase_
            base_ = allocate<Modulus>(size_, pool_);
            copy_n(rnsbase.cbegin(), size_, base_.get());

            // Initialize CRT data
            if (!initialize())
            {
                throw invalid_argument("rnsbase is invalid");
            }
        }

        RNSBase::RNSBase(const RNSBase &copy, MemoryPoolHandle pool) : pool_(move(pool)), size_(copy.size_)
        {
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }

            // Copy over the base
            base_ = allocate<Modulus>(size_, pool_);
            copy_n(copy.base_.get(), size_, base_.get());

            // Copy over CRT data
            base_prod_ = allocate_uint(size_, pool_);
            set_uint(copy.base_prod_.get(), size_, base_prod_.get());

            punctured_prod_array_ = allocate_uint(size_ * size_, pool_);
            set_uint(copy.punctured_prod_array_.get(), size_ * size_, punctured_prod_array_.get());

            inv_punctured_prod_mod_base_array_ = allocate<MultiplyUIntModOperand>(size_, pool_);
            copy_n(copy.inv_punctured_prod_mod_base_array_.get(), size_, inv_punctured_prod_mod_base_array_.get());

            set_GPU_params();
        }

        bool RNSBase::contains(const Modulus &value) const noexcept
        {
            bool result = false;
            SEAL_ITERATE(iter(base_), size_, [&](auto &I) { result = result || (I == value); });
            return result;
        }

        bool RNSBase::is_subbase_of(const RNSBase &superbase) const noexcept
        {
            bool result = true;
            SEAL_ITERATE(iter(base_), size_, [&](auto &I) { result = result && superbase.contains(I); });
            return result;
        }

        RNSBase RNSBase::extend(const Modulus &value) const
        {
            if (value.is_zero())
            {
                throw invalid_argument("value cannot be zero");
            }

            SEAL_ITERATE(iter(base_), size_, [&](auto I) {
                // The base must be coprime
                if (!are_coprime(I.value(), value.value()))
                {
                    throw logic_error("cannot extend by given value");
                }
            });

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = add_safe(size_, size_t(1));
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_, newbase.base_.get());

            // Extend with value
            newbase.base_[newbase.size_ - 1] = value;

            // Initialize CRT data
            if (!newbase.initialize())
            {
                throw logic_error("cannot extend by given value");
            }

            return newbase;
        }

        RNSBase RNSBase::extend(const RNSBase &other) const
        {
            // The bases must be coprime
            for (size_t i = 0; i < other.size_; i++)
            {
                for (size_t j = 0; j < size_; j++)
                {
                    if (!are_coprime(other[i].value(), base_[j].value()))
                    {
                        throw invalid_argument("rnsbase is invalid");
                    }
                }
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = add_safe(size_, other.size_);
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_, newbase.base_.get());

            // Extend with other base
            copy_n(other.base_.get(), other.size_, newbase.base_.get() + size_);

            // Initialize CRT data
            if (!newbase.initialize())
            {
                throw logic_error("cannot extend by given base");
            }

            return newbase;
        }

        RNSBase RNSBase::drop() const
        {
            if (size_ == 1)
            {
                throw logic_error("cannot drop from base of size 1");
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = size_ - 1;
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            copy_n(base_.get(), size_ - 1, newbase.base_.get());

            // Initialize CRT data
            newbase.initialize();

            return newbase;
        }

        RNSBase RNSBase::drop(const Modulus &value) const
        {
            if (size_ == 1)
            {
                throw logic_error("cannot drop from base of size 1");
            }
            if (!contains(value))
            {
                throw logic_error("base does not contain value");
            }

            // Copy over this base
            RNSBase newbase(pool_);
            newbase.size_ = size_ - 1;
            newbase.base_ = allocate<Modulus>(newbase.size_, newbase.pool_);
            size_t source_index = 0;
            size_t dest_index = 0;
            while (dest_index < size_ - 1)
            {
                if (base_[source_index] != value)
                {
                    newbase.base_[dest_index] = base_[source_index];
                    dest_index++;
                }
                source_index++;
            }

            // Initialize CRT data
            newbase.initialize();

            return newbase;
        }

        void RNSBase::set_GPU_params()
        {
            uint64_t inv_punctured_prod_mod_base_array_quotient[size_];
            uint64_t inv_punctured_prod_mod_base_array_operand[size_];
            uint64_t base_value[size_];
            uint64_t base_ratio0[size_];
            uint64_t base_ratio1[size_];
            for (size_t i = 0; i < size_; i++)
            {
                inv_punctured_prod_mod_base_array_quotient[i] = inv_punctured_prod_mod_base_array_[i].quotient;
                inv_punctured_prod_mod_base_array_operand[i] = inv_punctured_prod_mod_base_array_[i].operand;
                base_value[i] = base_[i].value();
                base_ratio0[i] = base_[i].const_ratio()[0];
                base_ratio1[i] = base_[i].const_ratio()[1];
            }

            // check if memory is malloced
            cudaPointerAttributes attributes;
            cudaError_t error = cudaPointerGetAttributes(&attributes, d_base_);
            if (error == cudaSuccess)
            {
                if (attributes.devicePointer == NULL)
                {
                    // printf("Pointer is not allocated on the device\n");
                    checkCudaErrors(cudaMalloc((void **)&d_punctured_prod_, size_ * size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_base_prod_, size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_inv_punctured_prod_mod_base_array_quotient_, size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_inv_punctured_prod_mod_base_array_operand_, size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_base_, size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_base_ratio0_, size_ * sizeof(uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_base_ratio1_, size_ * sizeof(uint64_t)));
                }
            }
            else
            {
                printf("Failed to get pointer attributes: %s\n", cudaGetErrorString(error));
            }

            checkCudaErrors(cudaMemcpy(
                d_punctured_prod_, punctured_prod_array_.get(), size_ * size_ * sizeof(uint64_t),
                cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_base_prod_, base_prod_.get(), size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_inv_punctured_prod_mod_base_array_quotient_, inv_punctured_prod_mod_base_array_quotient,
                size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_inv_punctured_prod_mod_base_array_operand_, inv_punctured_prod_mod_base_array_operand,
                size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_base_, base_value, size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_base_ratio0_, base_ratio0, size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_base_ratio1_, base_ratio1, size_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

        bool RNSBase::initialize()
        {
            // Verify that the size is not too large
            if (!product_fits_in(size_, size_))
            {
                return false;
            }

            base_prod_ = allocate_uint(size_, pool_);
            punctured_prod_array_ = allocate_zero_uint(size_ * size_, pool_);
            inv_punctured_prod_mod_base_array_ = allocate<MultiplyUIntModOperand>(size_, pool_);

            if (size_ > 1)
            {
                auto rnsbase_values = allocate<uint64_t>(size_, pool_);
                SEAL_ITERATE(iter(base_, rnsbase_values), size_, [&](auto I) { get<1>(I) = get<0>(I).value(); });

                // Create punctured products
                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);
                SEAL_ITERATE(iter(punctured_prod, size_t(0)), size_, [&](auto I) {
                    multiply_many_uint64_except(rnsbase_values.get(), size_, get<1>(I), get<0>(I).ptr(), pool_);
                });

                // Compute the full product
                auto temp_mpi(allocate_uint(size_, pool_));
                multiply_uint(punctured_prod_array_.get(), size_, base_[0].value(), size_, temp_mpi.get());
                set_uint(temp_mpi.get(), size_, base_prod_.get());

                // Compute inverses of punctured products mod primes
                bool invertible = true;
                SEAL_ITERATE(iter(punctured_prod, base_, inv_punctured_prod_mod_base_array_), size_, [&](auto I) {
                    uint64_t temp = modulo_uint(get<0>(I), size_, get<1>(I));
                    invertible = invertible && try_invert_uint_mod(temp, get<1>(I), temp);
                    get<2>(I).set(temp, get<1>(I));
                });

                set_GPU_params();
                return invertible;
            }

            // Case of a single prime
            base_prod_[0] = base_[0].value();
            punctured_prod_array_[0] = 1;
            inv_punctured_prod_mod_base_array_[0].set(1, base_[0]);
            set_GPU_params();
            return true;
        }

        void RNSBase::decompose(uint64_t *value, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                // Copy the value
                auto value_copy(allocate_uint(size_, pool));
                set_uint(value, size_, value_copy.get());

                SEAL_ITERATE(iter(value, base_), size_, [&](auto I) {
                    get<0>(I) = modulo_uint(value_copy.get(), size_, get<1>(I));
                });
            }
        }

        void RNSBase::decompose_array(uint64_t *value, size_t count, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                // Decompose an array of multi-precision integers into an array of arrays, one per each base element

                // Copy the input array into a temporary location and set a StrideIter pointing to it
                // Note that the stride size is size_
                SEAL_ALLOCATE_GET_STRIDE_ITER(value_copy, uint64_t, count, size_, pool);
                set_uint(value, count * size_, value_copy);

                // Note how value_copy and value_out have size_ and count reversed
                RNSIter value_out(value, count);

                // For each output RNS array (one per base element) ...
                SEAL_ITERATE(iter(base_, value_out), size_, [&](auto I) {
                    // For each multi-precision integer in value_copy ...
                    SEAL_ITERATE(iter(get<1>(I), value_copy), count, [&](auto J) {
                        // Reduce the multi-precision integer modulo the base element and write to value_out
                        get<0>(J) = modulo_uint(get<1>(J), size_, get<0>(I));
                    });
                });
            }
        }


        __global__ void decompose_array_helper(uint64_t *value, size_t count, size_t size, uint64_t *value_copy, uint64_t *modulu_value,
                uint64_t *modulu_ratio0, uint64_t *modulu_ratio1)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < count * size)
            {
                value[index] = modulo_uint_kernel(value_copy + index, size, modulu_value, modulu_ratio0, modulu_ratio1);
                index += blockDim.x * gridDim.x;
            }
        }


        void RNSBase::decompose_array_cuda(uint64_t *value, size_t count) const{
            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                // Decompose an array of multi-precision integers into an array of arrays, one per each base element

                // Copy the input array into a temporary location and set a StrideIter pointing to it
                // Note that the stride size is size_

                // SEAL_ALLOCATE_GET_STRIDE_ITER(value_copy, uint64_t, count, size_, pool);
                // set_uint(value, count * size_, value_copy);

                uint64_t *d_value_copy = nullptr;
                allocate_gpu<uint64_t>(&d_value_copy, count * size_);
                checkCudaErrors(cudaMemcpy(d_value_copy, value, count * size_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

                decompose_array_helper<<<(size_ * count + 255) / 256, 256>>>(value, count, size_, d_value_copy, d_base_, d_base_ratio0_, d_base_ratio1_);

                // // Note how value_copy and value_out have size_ and count reversed
                // RNSIter value_out(value, count);

                // // For each output RNS array (one per base element) ...
                // SEAL_ITERATE(iter(base_, value_out), size_, [&](auto I) {
                //     // For each multi-precision integer in value_copy ...
                //     SEAL_ITERATE(iter(get<1>(I), value_copy), count, [&](auto J) {
                //         // Reduce the multi-precision integer modulo the base element and write to value_out
                //         get<0>(J) = modulo_uint(get<1>(J), size_, get<0>(I));
                //     });
                // });
            }

        }

        void RNSBase::compose(uint64_t *value, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                // Copy the value
                auto temp_value(allocate_uint(size_, pool));
                set_uint(value, size_, temp_value.get());

                // Clear the result
                set_zero_uint(size_, value);

                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);

                // Compose an array of integers (one per base element) into a single multi-precision integer
                auto temp_mpi(allocate_uint(size_, pool));
                SEAL_ITERATE(
                    iter(temp_value, inv_punctured_prod_mod_base_array_, punctured_prod, base_), size_, [&](auto I) {
                        uint64_t temp_prod = multiply_uint_mod(get<0>(I), get<1>(I), get<3>(I));
                        multiply_uint(get<2>(I), size_, temp_prod, size_, temp_mpi.get());
                        add_uint_uint_mod(temp_mpi.get(), value, base_prod_.get(), size_, value);
                    });
            }
        }

        void RNSBase::compose_array(uint64_t *value, size_t count, MemoryPoolHandle pool) const
        {
            if (!value)
            {
                throw invalid_argument("value cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }

            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                // Merge the coefficients first
                auto temp_array(allocate_uint(count * size_, pool));
                for (size_t i = 0; i < count; i++)
                {
                    for (size_t j = 0; j < size_; j++)
                    {
                        temp_array[j + (i * size_)] = value[(j * count) + i];
                    }
                }

                // Clear the result
                set_zero_uint(count * size_, value);

                StrideIter<uint64_t *> temp_array_iter(temp_array.get(), size_);
                StrideIter<uint64_t *> value_iter(value, size_);
                StrideIter<uint64_t *> punctured_prod(punctured_prod_array_.get(), size_);

                // Compose an array of RNS integers into a single array of multi-precision integers
                auto temp_mpi(allocate_uint(size_, pool));
                SEAL_ITERATE(iter(temp_array_iter, value_iter), count, [&](auto I) {
                    SEAL_ITERATE(
                        iter(get<0>(I), inv_punctured_prod_mod_base_array_, punctured_prod, base_), size_, [&](auto J) {
                            uint64_t temp_prod = multiply_uint_mod(get<0>(J), get<1>(J), get<3>(J));
                            multiply_uint(get<2>(J), size_, temp_prod, size_, temp_mpi.get());
                            add_uint_uint_mod(temp_mpi.get(), get<1>(I), base_prod_.get(), size_, get<1>(I));
                        });
                });
            }
        }

        uint64_t power_m(uint64_t x, uint64_t n, const Modulus &modulus)
        {
            uint64_t res = 1; // Initialize result

            x = x % modulus.value(); // Update x if it is more than or
            // equal to p

            while (n > 0)
            {
                // If y is odd, multiply x with result
                if (n & 1)
                    // res = (res * x) % p;
                    res = multiply_uint_mod(res, x, modulus);

                // y must be even now
                n = n >> 1; // y = y/2
                // x = (x * x) % p;
                x = multiply_uint_mod(x, x, modulus);
            }
            return res;
        }


        __global__ void fill_temp_array_kernel(uint64_t *value, uint64_t *destination, uint64_t count, uint64_t size)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < count * size)
            {
                destination[index] = value[(index % size) * count + (index / size)];
                index += blockDim.x * gridDim.x;
            }
        }

        __device__ void multiply_uint_kernel(
            const uint64_t *operand1, uint64_t size, const uint64_t operand2, uint64_t result_uint64_count,
            uint64_t *result)
        {
            unsigned long long carry = 0;

            for (size_t i = 0; i < size; i++)
            {
                unsigned long long temp_result[2];
                multiply_uint64_kernel(*operand1++, operand2, temp_result);
                unsigned long long temp;
                carry = temp_result[1] + add_uint64_carry_kernel(temp_result[0], carry, 0, &temp);
                *result++ = temp;
            }
            if (size < result_uint64_count)
            {
                *result = carry;
            }
        }

        __device__ void add_uint_uint_kernel(
            uint64_t *operand1, uint64_t *operand2, uint64_t *modulus, uint64_t uint64_count, uint64_t *result)
        {
            unsigned char carry = add_uint_kernel(*operand1++, *operand2++, result++);
            size_t uint64_count_copy = uint64_count;

            for (; --uint64_count; operand1++, operand2++, result++)
            {
                unsigned long long temp_result;
                carry = add_uint_kernel(*operand1, *operand2, carry, &temp_result);
                *result = temp_result;
            }
            result -= uint64_count_copy;

            if (carry || is_greater_than_or_equal_uint_kernel(result, modulus, uint64_count_copy))
            {
                sub_uint_kernel(result, modulus, uint64_count_copy, result);
            }
        }

        __global__ void compose_kernel(
            uint64_t *tmp_array, uint64_t *inv_punctured_prod_mod_base_array_quotient,
            uint64_t *inv_punctured_prod_mod_base_array_operand, uint64_t *base, uint64_t *d_punctured_prod,
            uint64_t *d_base_prod, uint64_t *value, uint64_t count, uint64_t size)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < count)
            {
                // const uint64_t tmp_mpi_size = size;
                for (int i = 0; i < size; i++)
                {
                    uint64_t d_temp_mpi[3];
                    uint64_t temp_prod = multiply_uint_mod_kernel(
                        tmp_array[index * size + i], inv_punctured_prod_mod_base_array_quotient[i],
                        inv_punctured_prod_mod_base_array_operand[i], base[i]);
                    multiply_uint_kernel(d_punctured_prod + i * size, size, temp_prod, size, d_temp_mpi);
                    add_uint_uint_kernel(d_temp_mpi, value + index * size, d_base_prod, size, value + index * size);
                }
                index += blockDim.x * gridDim.x;
            }
        }


        template <typename T>
        __global__ void eltwiseAddModScalarKernel(
            T *result, const T *operand1, const T operand2, const T modulus, const std::size_t size)
        {
            register T scalar = operand2;

            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < size)
            {
                result[index] =
                    operand1[index] + scalar >= modulus ? operand1[index] + scalar - modulus : operand1[index] + scalar;
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void modulo_poly_coeffs_kernel(
            uint64_t *last_input, uint64_t coeff_count, uint64_t base_q_value, uint64_t base_q_ratio,
            uint64_t *temp_result)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count)
            {
                temp_result[index] = barrett_reduce_64_kernel(last_input[index], base_q_value, base_q_ratio);
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void modulo_poly_coeffs_kernel(
            uint64_t *last_input, uint64_t coeff_count, size_t modulu_size, uint64_t *base_q_value, uint64_t *base_q_ratio,
            uint64_t *temp_result)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count)
            {
                size_t modulu_index = coeff_count / modulu_size;
                temp_result[index] = barrett_reduce_64_kernel(last_input[index%coeff_count], base_q_value[modulu_index], base_q_ratio[modulu_index]);
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void fill_kernel(uint64_t *source, size_t size, uint64_t *dest)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < size)
            {
                dest[index] = source[index];
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void add_neg_kernel(uint64_t *tmp, uint64_t size, uint64_t neg_half_mod)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < size)
            {
                tmp[index] = tmp[index] + neg_half_mod;
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void add_qi_lazy_kernel(uint64_t *dest, uint64_t *ori, uint64_t size, uint64_t qi_lazy)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < size)
            {
                dest[index] += qi_lazy - ori[index];
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void multiply_poly_scalar_kernel(
            uint64_t *input, uint64_t coeff_count, uint64_t coeff_modulus_size_, uint64_t quotient, uint64_t operand,
            uint64_t modulus_value, uint64_t *dest)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count)
            {
                unsigned long long tmp1, tmp2;
                const std::uint64_t p = modulus_value;
                multiply_uint64_hw64_kernel(input[index], quotient, &tmp1);
                tmp2 = operand * input[index] - tmp1 * p;
                dest[index] = tmp2 >= p ? tmp2 - p : tmp2;

                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void multiply_poly_scalar_coeffmod_kernel_rns(
            uint64_t *poly, size_t coeff_count, uint64_t operand, uint64_t quotient, const uint64_t modulus_value,
            uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            while (idx < coeff_count)
            {
                unsigned long long tmp1, tmp2;
                multiply_uint64_hw64_kernel(poly[idx], quotient, &tmp1);
                tmp2 = operand * poly[idx] - tmp1 * modulus_value;
                result[idx] = tmp2 >= modulus_value ? tmp2 - modulus_value : tmp2;

                idx += blockDim.x * gridDim.x;
            }

        }

        __global__ void sm_mrq_kernel( uint64_t *input, uint64_t *r_m_tilde, uint64_t *destination,
            size_t base_Bsk_size, size_t coeff_count, uint64_t m_tilde_value,
            uint64_t *prod_B_mod_Bsk, uint64_t *base_Bsk_value, uint64_t *base_Bsk_ratio,
            uint64_t *inv_m_tilde_mod_Bsk_operand, uint64_t *inv_m_tilde_mod_Bsk_quotient){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < base_Bsk_size * coeff_count) {
                size_t base_Bsk_idx = idx / coeff_count;
                size_t coeff_idx = idx % coeff_count;

                uint64_t prod_q_mod_Bsk_elt_operand = prod_B_mod_Bsk[base_Bsk_idx];
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, prod_q_mod_Bsk_elt_operand };
                divide_uint128_inplace_kernel(wide_coeff, base_Bsk_value[base_Bsk_idx], wide_quotient);
                uint64_t prod_q_mod_Bsk_elt_quotient = wide_quotient[0];

                uint64_t temp = r_m_tilde[coeff_idx];
                if (temp >= (m_tilde_value >> 1))
                {
                    temp += base_Bsk_value[base_Bsk_idx] - m_tilde_value;
                }

                uint64_t temp_oper = multiply_add_uint_mod_kernel(temp, prod_q_mod_Bsk_elt_operand, prod_q_mod_Bsk_elt_quotient, input[idx], base_Bsk_value[base_Bsk_idx], base_Bsk_ratio[base_Bsk_idx], 1);

                destination[idx] = multiply_uint_mod_kernel(
                    temp_oper,
                    inv_m_tilde_mod_Bsk_quotient[base_Bsk_idx],
                    inv_m_tilde_mod_Bsk_operand[base_Bsk_idx], 
                    base_Bsk_value[base_Bsk_idx]);
            
                idx += blockDim.x * gridDim.x;
            }

        }
        
        __global__ void multiply_poly_scalar_coeffmod_kernel_kernel(uint64_t *input, uint64_t *result,
            size_t coeff_count, size_t base_size,
            uint64_t *modulus_value, uint64_t *modulus_ratio,
            uint64_t scalar){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < coeff_count * base_size) {
                size_t base_idx = idx / coeff_count;
                uint64_t operand = barrett_reduce_64_kernel(scalar, modulus_value[base_idx], modulus_ratio[base_idx]);
                
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };
                divide_uint128_inplace_kernel(wide_coeff, modulus_value[base_idx], wide_quotient);
                uint64_t quotient = wide_quotient[0];

                uint64_t x = input[idx];
                result[idx] = multiply_uint_mod_kernel(x, quotient, operand, modulus_value[base_idx]);

                idx += blockDim.x * gridDim.x;
            }
        
        }

        __global__ void multiply_poly_scalar_coeffmod_kernel_one_modulu(uint64_t *input, uint64_t *result,
            size_t coeff_count,
            uint64_t modulus_value, uint64_t modulus_ratio,
            uint64_t scalar){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < coeff_count) {
                uint64_t operand = barrett_reduce_64_kernel(scalar, modulus_value, modulus_ratio);
                
                std::uint64_t wide_quotient[2]{ 0, 0 };
                std::uint64_t wide_coeff[2]{ 0, operand };
                divide_uint128_inplace_kernel(wide_coeff, modulus_value, wide_quotient);
                uint64_t quotient = wide_quotient[0];

                uint64_t x = input[idx];
                result[idx] = multiply_uint_mod_kernel(x, quotient, operand, modulus_value);

                idx += blockDim.x * gridDim.x;
            }
        
        }

        __global__ void add_uint_helper( uint64_t *input, uint64_t *operand, uint64_t *destination, 
                                        size_t coeff_count, size_t modulu_size,
                                        uint64_t *modulus_value, uint64_t *modulu_ratio0, uint64_t *modulu_ratio1)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while(index < coeff_count*modulu_size)
            {
                size_t modulu_index = index / coeff_count;

                uint64_t *temp_ratio = new uint64_t[2];
                temp_ratio[0] = modulu_ratio0[modulu_index];
                temp_ratio[1] = modulu_ratio1[modulu_index];
                uint64_t oper2 = barrett_reduce_128_kernel2(operand + index % coeff_count, modulus_value[modulu_index], temp_ratio);

                destination[index] = add_uint_mod_kernel(input[index], oper2, modulus_value[modulu_index]);

                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void sub_uint_helper( uint64_t *input, uint64_t *operand, uint64_t *destination, 
                                        size_t coeff_count, size_t modulu_size,
                                        uint64_t *modulus_value)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while(index < coeff_count*modulu_size)
            {
                size_t modulu_index = index / coeff_count;

                destination[index] = sub_uint_mod_kernel(input[index], operand[index], modulus_value[modulu_index]);

                index += blockDim.x * gridDim.x;
            }
        }

    namespace{
            template <typename T>
            __global__ void eltwiseSubModKernel(T *result, const T *operand1, const T *operand2, const T modulus, const std::size_t size) {
                std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

                while (index < size) {
                    result[index] = operand1[index] > operand2[index] ? operand1[index] - operand2[index] : operand1[index] + modulus - operand2[index];
                    index += blockDim.x * gridDim.x;
                }
            }
    }

        void RNSBase::compose_array_cuda(uint64_t *value, size_t count, MemoryPoolHandle pool) const
        {
            if (size_ > 1)
            {
                if (!product_fits_in(count, size_))
                {
                    throw logic_error("invalid parameters");
                }

                uint64_t *d_temp_array;

                checkCudaErrors(cudaMalloc((void **)&d_temp_array, count * size_ * sizeof(uint64_t)));
                fill_temp_array_kernel<<<(count * size_ + 255) / 256, 256>>>(value, d_temp_array, count, size_);
                // Clear the result
                checkCudaErrors(cudaMemset(value, 0, count * size_ * sizeof(uint64_t)));

                compose_kernel<<<(count + 255) / 256, 256>>>(
                    d_temp_array, d_inv_punctured_prod_mod_base_array_quotient_,
                    d_inv_punctured_prod_mod_base_array_operand_, d_base_, d_punctured_prod_, d_base_prod_, value,
                    count, size_);
            }
        }

        void BaseConverter::fast_convert(ConstCoeffIter in, CoeffIter out, MemoryPoolHandle pool) const
        {
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, ibase_size, pool);
            SEAL_ITERATE(
                iter(temp, in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base()), ibase_size,
                [&](auto I) { get<0>(I) = multiply_uint_mod(get<1>(I), get<2>(I), get<3>(I)); });

            // for (size_t j = 0; j < obase_size; j++)
            SEAL_ITERATE(iter(out, base_change_matrix_, obase_.base()), obase_size, [&](auto I) {
                get<0>(I) = dot_product_mod(temp, get<1>(I).get(), ibase_size, get<2>(I));
            });
        }


        __global__ void fast_convert_array_helper1(uint64_t *in, uint64_t *temp,
                                                    size_t count, size_t ibase_size,
                                                    uint64_t *d_inv_punctured_prod_mod_base_array_operand_,
                                                    uint64_t *d_inv_punctured_prod_mod_base_array_quotient_,
                                                    uint64_t *ibase_modulus_value,
                                                    uint64_t *ibase_modulus_ratio
                                                    ) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < count * ibase_size) {
                size_t base_idx = idx / count;
                size_t ibase_temp_idx = (idx % count) * ibase_size + base_idx;
                if (d_inv_punctured_prod_mod_base_array_operand_[base_idx] ==  1){
                    temp[ibase_temp_idx] = barrett_reduce_64_kernel(in[idx], 
                                                        ibase_modulus_value[base_idx], 
                                                        ibase_modulus_ratio[base_idx]);
                } else {
                    temp[ibase_temp_idx] = multiply_uint_mod_kernel(in[idx], 
                                                        d_inv_punctured_prod_mod_base_array_quotient_[base_idx], 
                                                        d_inv_punctured_prod_mod_base_array_operand_[base_idx], 
                                                        ibase_modulus_value[base_idx]);
                }

                idx += blockDim.x * gridDim.x;

            }
        }

        __global__ void fast_convert_array_helper2( uint64_t *temp, uint64_t *out, 
                                                    size_t count, size_t ibase_size, size_t obase_size,        
                                                    uint64_t *d_base_change_matrix_,
                                                    uint64_t *obase_modulus_value,
                                                    uint64_t *obase_modulus_ratio_0,
                                                    uint64_t *obase_modulus_ratio_1){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < count * obase_size) {
                size_t base_idx = idx / count;
                uint64_t temp_ratio[2] ;
                temp_ratio[0] = obase_modulus_ratio_0[base_idx];
                temp_ratio[1] = obase_modulus_ratio_1[base_idx];

                out[idx] = dot_product_mod_kernel(temp + idx % count * ibase_size, 
                                                d_base_change_matrix_ + base_idx * ibase_size, 
                                                ibase_size, 
                                                obase_modulus_value[base_idx], 
                                                temp_ratio);

                idx += blockDim.x * gridDim.x;
            }

        }

        __global__ void fast_floor_kernel(uint64_t *input, uint64_t *destination, size_t Bsk_size, size_t count, 
                                                    uint64_t *d_inv_punctured_prod_q_mod_base_array_operand_,
                                                    uint64_t *d_inv_punctured_prod_q_mod_base_array_quotient_,
                                                    uint64_t *Bsk_modulus_value) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < count * Bsk_size) {
                size_t base_idx = idx / count;
                destination[idx] = multiply_uint_mod_kernel(input[idx] + (Bsk_modulus_value[base_idx] - destination[idx]),
                                                            d_inv_punctured_prod_q_mod_base_array_quotient_[base_idx], 
                                                            d_inv_punctured_prod_q_mod_base_array_operand_[base_idx], 
                                                            Bsk_modulus_value[base_idx]);
                idx += blockDim.x * gridDim.x;
            }

        }

        __global__ void fastbconv_sk_kernel_helper1(uint64_t *input, uint64_t *temp, size_t count, uint64_t *output, uint64_t modulu_value,
                                            uint64_t operand, uint64_t quotient) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < count) {
                output[idx] = multiply_uint_mod_kernel(temp[idx] + (modulu_value - input[idx]), quotient, operand, modulu_value);
                idx += blockDim.x * gridDim.x;
            }
        }



        __global__ void fastbconv_sk_kernel_helper2(uint64_t *input, uint64_t *output, 
                size_t base_q_size, size_t count, uint64_t threshold, uint64_t msk_value,
                uint64_t *d_prod_B_mod_q, uint64_t *modulu_value, uint64_t *ratio1) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < base_q_size * count) {
                size_t modulu_index = idx / count;

                size_t input_index = idx % count;
                if (input[input_index] > threshold) {
                    std::int64_t non_zero = static_cast<std::int64_t>(input[input_index] != 0);
                    uint64_t neg_input = (msk_value - input[input_index]) & static_cast<std::uint64_t>(-non_zero);

                    uint64_t operand = d_prod_B_mod_q[modulu_index];
                    uint64_t quotient = 0;

                    std::uint64_t wide_quotient[2]{ 0, 0 };
                    std::uint64_t wide_coeff[2]{ 0, operand };
                    divide_uint128_inplace_kernel(wide_coeff, modulu_value[modulu_index], wide_quotient);
                    quotient = wide_quotient[0];

                    output[idx] = multiply_add_uint_mod_kernel(neg_input, operand, quotient, output[idx], modulu_value[modulu_index], ratio1[modulu_index], idx);
                } else {

                    uint64_t operand = modulu_value[modulu_index] - d_prod_B_mod_q[modulu_index];
                    uint64_t quotient = 0;

                    std::uint64_t wide_quotient[2]{ 0, 0 };
                    std::uint64_t wide_coeff[2]{ 0, operand };
                    divide_uint128_inplace_kernel(wide_coeff, modulu_value[modulu_index], wide_quotient);
                    quotient = wide_quotient[0];

                    output[idx] = multiply_add_uint_mod_kernel(input[input_index], operand, quotient, output[idx], modulu_value[modulu_index], ratio1[modulu_index], idx);
                }

                idx += blockDim.x * gridDim.x;
            }
        }

   
        void BaseConverter::fast_convert_array(ConstRNSIter in, RNSIter out, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (in.poly_modulus_degree() != out.poly_modulus_degree())
            {
                throw invalid_argument("in and out are incompatible");
            }
#endif
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();
            size_t count = in.poly_modulus_degree();

            // Note that the stride size is ibase_size
            SEAL_ALLOCATE_GET_STRIDE_ITER(temp, uint64_t, count, ibase_size, pool);

            SEAL_ITERATE(
                iter(in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base(), size_t(0)), ibase_size,
                [&](auto I) {
                    // The current ibase index
                    size_t ibase_index = get<3>(I);

                    if (get<1>(I).operand == 1)
                    {
                        // No multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                            // Reduce modulo ibase element
                            get<1>(J)[ibase_index] = barrett_reduce_64(get<0>(J), get<2>(I));
                        });
                    }
                    else
                    {
                        // Multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                            // Multiply coefficient of in with ibase_.inv_punctured_prod_mod_base_array_ element
                            get<1>(J)[ibase_index] = multiply_uint_mod(get<0>(J), get<1>(I), get<2>(I));
                        });
                    }
                });

            SEAL_ITERATE(iter(out, base_change_matrix_, obase_.base()), obase_size, [&](auto I) {
                SEAL_ITERATE(iter(get<0>(I), temp), count, [&](auto J) {
                    // Compute the base conversion sum modulo obase element
                    get<0>(J) = dot_product_mod(get<1>(J), get<1>(I).get(), ibase_size, get<2>(I));
                });
            });
        }
        
        void BaseConverter::ensure_size(uint64_t **input, size_t current_size, size_t &size) const
        {
            if (current_size > size)
            {
                // printf("before size %llu ,Reallocating %llu bytes\n", size,  current_size);
                checkCudaErrors(cudaMalloc((void **)input, current_size * sizeof(uint64_t)));
                size = current_size;
            }
        }

        void BaseConverter::fast_convert_array_cuda(uint64_t *d_in, uint64_t *d_out, size_t count) const
        {
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();

            // Note that the stride size is ibase_size

            uint64_t *d_ibase_modulu_value = ibase_.d_base();
            uint64_t *d_ibase_modulu_ratio_0 = ibase_.d_ratio0();
            uint64_t *d_ibase_modulu_ratio_1 = ibase_.d_ratio1();
            
            // ensure_size(&d_temp_convert_, count * ibase_size, d_temp_convert_size_);
            allocate_gpu<uint64_t>(&d_temp_convert_, count * ibase_size);
            uint64_t *d_obase_modulu_value = obase_.d_base();
            uint64_t *d_obase_modulu_ratio_0 = obase_.d_ratio0();
            uint64_t *d_obase_modulu_ratio_1 = obase_.d_ratio1();

// 两个可以合一起
            fast_convert_array_helper1<<<(ibase_size * count + 255)/256, 256>>> (d_in, d_temp_convert_,
                                                                                count, ibase_size,
                                                                                ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                                ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                                d_ibase_modulu_value,
                                                                                d_ibase_modulu_ratio_1
                                                                                );

            fast_convert_array_helper2<<<(obase_size * count + 255) / 256, 256>>>(d_temp_convert_, d_out, 
                                                                                count, 
                                                                                ibase_size, obase_size, 
                                                                                d_base_change_matrix_,
                                                                                d_obase_modulu_value, 
                                                                                d_obase_modulu_ratio_0, 
                                                                                d_obase_modulu_ratio_1);
            deallocate_gpu<uint64_t>(&d_temp_convert_, count * ibase_size);

        }

        template <size_t SIZE>
        __global__ void exact_convert_helper(uint64_t *input, size_t base_size, size_t count,
                                            uint64_t *operand, uint64_t *quotient, 
                                            uint64_t *ibase_value, uint64_t *ibase_ratio_1,
                                            uint64_t obase_value, uint64_t obase_ratio0, uint64_t obase_ratio1,
                                            uint64_t *base_change_matrix, uint64_t q_mod_p,
                                            uint64_t *output){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < count){
                uint64_t temp[SIZE];
                double_t aggregated_v = 0.0;

                for (int shift = 0; shift < base_size; shift++){
                    double_t divisor = static_cast<double_t>(ibase_value[shift]);

                    if (operand[shift] == 1){
                        temp[shift] = barrett_reduce_64_kernel(input[idx + count * shift], ibase_value[shift], ibase_ratio_1[shift]);
                    } else{
                        temp[shift] = multiply_uint_mod_kernel(input[idx + count * shift], quotient[shift], operand[shift], ibase_value[shift]);
                    }
                    double_t dividend = static_cast<double_t>(temp[shift]);
                    aggregated_v += dividend / divisor;
                }
                aggregated_v += 0.5; // 就是aggregated_rounded_v
                uint64_t aggregated_rounded_v = static_cast<uint64_t>(aggregated_v);

                uint64_t temp_ratio[2] ;
                temp_ratio[0] = obase_ratio0;
                temp_ratio[1] = obase_ratio1;

                uint64_t sum_mod_obase = dot_product_mod_kernel(temp, base_change_matrix, base_size, obase_value, temp_ratio);

                unsigned long long z[2];
                multiply_uint64_kernel2(aggregated_rounded_v, q_mod_p, z);

                uint64_t v_q_mod_p = barrett_reduce_128_kernel2(z, obase_value, temp_ratio);

                output[idx] = sub_uint_mod_kernel(sum_mod_obase, v_q_mod_p, obase_value);
            
                idx += blockDim.x * gridDim.x;
            }

        }

        void BaseConverter::exact_convert_array_cuda(uint64_t *in, uint64_t *out, size_t count) const{
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();

            if (obase_size != 1)
            {
                throw invalid_argument("out base in exact_convert_array must be one.");
            }

            auto p = obase_.base()[0];
            auto q_mod_p = modulo_uint(ibase_.base_prod(), ibase_size, p);
            switch (ibase_size)
            {
                case 1:
                    exact_convert_helper<1><<<(count + 255) / 256, 256>>>(in, ibase_size, count, 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                        ibase_.d_base(), ibase_.d_ratio1(),
                                                                        p.value(), p.const_ratio().data()[0], p.const_ratio().data()[1],
                                                                        d_base_change_matrix_, q_mod_p,
                                                                        out);
                    break;
                
                case 2:
                    exact_convert_helper<2><<<(count + 255) / 256, 256>>>(in, ibase_size, count, 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                        ibase_.d_base(), ibase_.d_ratio1(),
                                                                        p.value(), p.const_ratio().data()[0], p.const_ratio().data()[1],
                                                                        d_base_change_matrix_, q_mod_p,
                                                                        out);
                    break;

                case 3:
                    exact_convert_helper<3><<<(count + 255) / 256, 256>>>(in, ibase_size, count, 
                                                                    ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                    ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                    ibase_.d_base(), ibase_.d_ratio1(),
                                                                    p.value(), p.const_ratio().data()[0], p.const_ratio().data()[1],
                                                                    d_base_change_matrix_, q_mod_p,
                                                                    out);
                break;

                case 4:
                    exact_convert_helper<4><<<(count + 255) / 256, 256>>>(in, ibase_size, count, 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                        ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                        ibase_.d_base(), ibase_.d_ratio1(),
                                                                        p.value(), p.const_ratio().data()[0], p.const_ratio().data()[1],
                                                                        d_base_change_matrix_, q_mod_p,
                                                                        out);
                break;

                case 5:
                    exact_convert_helper<5><<<(count + 255) / 256, 256>>>(in, ibase_size, count, 
                                                                    ibase_.d_inv_punctured_prod_mod_base_array_operand(), 
                                                                    ibase_.d_inv_punctured_prod_mod_base_array_quotient(), 
                                                                    ibase_.d_base(), ibase_.d_ratio1(),
                                                                    p.value(), p.const_ratio().data()[0], p.const_ratio().data()[1],
                                                                    d_base_change_matrix_, q_mod_p,
                                                                    out);
                break;

                default:
                    throw invalid_argument("ibase_size out of bounds of 5.");
            }
        }

        // See "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme" (CT-RSA 2019) for details
        void BaseConverter::exact_convert_array(ConstRNSIter in, CoeffIter out, MemoryPoolHandle pool) const
        {
            size_t ibase_size = ibase_.size();
            size_t obase_size = obase_.size();
            size_t count = in.poly_modulus_degree();

            if (obase_size != 1)
            {
                throw invalid_argument("out base in exact_convert_array must be one.");
            }

            // Note that the stride size is ibase_size
            SEAL_ALLOCATE_GET_STRIDE_ITER(temp, uint64_t, count, ibase_size, pool);

            // The iterator storing v
            SEAL_ALLOCATE_GET_STRIDE_ITER(v, double_t, count, ibase_size, pool);

            // Aggregated rounded v
            SEAL_ALLOCATE_GET_PTR_ITER(aggregated_rounded_v, uint64_t, count, pool);

            // Calculate [x_{i} * \hat{q_{i}}]_{q_{i}}
            SEAL_ITERATE(
                iter(in, ibase_.inv_punctured_prod_mod_base_array(), ibase_.base(), size_t(0)), ibase_size,
                [&](auto I) {
                    // The current ibase index
                    size_t ibase_index = get<3>(I);
                    double_t divisor = static_cast<double_t>(get<2>(I).value());

                    if (get<1>(I).operand == 1)
                    {
                        // No multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp, v), count, [&](auto J) {
                            // Reduce modulo ibase element
                            get<1>(J)[ibase_index] = barrett_reduce_64(get<0>(J), get<2>(I));
                            double_t dividend = static_cast<double_t>(get<1>(J)[ibase_index]);
                            get<2>(J)[ibase_index] = dividend / divisor;
                        });
                    }
                    else
                    {
                        // Multiplication needed
                        SEAL_ITERATE(iter(get<0>(I), temp, v), count, [&](auto J) {
                            // Multiply coefficient of in with ibase_.inv_punctured_prod_mod_base_array_ element
                            get<1>(J)[ibase_index] = multiply_uint_mod(get<0>(J), get<1>(I), get<2>(I));
                            double_t dividend = static_cast<double_t>(get<1>(J)[ibase_index]);
                            get<2>(J)[ibase_index] = dividend / divisor;
                        });
                    }
                });

            // Aggrate v and rounding
            SEAL_ITERATE(iter(v, aggregated_rounded_v), count, [&](auto I) {
                // Otherwise a memory space of the last execution will be used.
                double_t aggregated_v = 0.0;
                for (size_t i = 0; i < ibase_size; ++i)
                {
                    aggregated_v += get<0>(I)[i];
                }
                aggregated_v += 0.5;
                get<1>(I) = static_cast<uint64_t>(aggregated_v);
            });

            auto p = obase_.base()[0];
            auto q_mod_p = modulo_uint(ibase_.base_prod(), ibase_size, p);
            auto base_change_matrix_first = base_change_matrix_[0].get();
            // Final multiplication
            SEAL_ITERATE(iter(out, temp, aggregated_rounded_v), count, [&](auto J) {
                // Compute the base conversion sum modulo obase element
                auto sum_mod_obase = dot_product_mod(get<1>(J), base_change_matrix_first, ibase_size, p);
                // Minus v*[q]_{p} mod p
                auto v_q_mod_p = multiply_uint_mod(get<2>(J), q_mod_p, p);
                get<0>(J) = sub_uint_mod(sum_mod_obase, v_q_mod_p, p);
            });
        }

        void BaseConverter::initialize()
        {
            // Verify that the size is not too large
            if (!product_fits_in(ibase_.size(), obase_.size()))
            {
                throw logic_error("invalid parameters");
            }

            // Create the base-change matrix rows
            base_change_matrix_ = allocate<Pointer<uint64_t>>(obase_.size(), pool_);

            SEAL_ITERATE(iter(base_change_matrix_, obase_.base()), obase_.size(), [&](auto I) {
                // Create the base-change matrix columns
                get<0>(I) = allocate_uint(ibase_.size(), pool_);

                StrideIter<const uint64_t *> ibase_punctured_prod_array(ibase_.punctured_prod_array(), ibase_.size());
                SEAL_ITERATE(iter(get<0>(I), ibase_punctured_prod_array), ibase_.size(), [&](auto J) {
                    // Base-change matrix contains the punctured products of ibase elements modulo the obase
                    get<0>(J) = modulo_uint(get<1>(J), ibase_.size(), get<1>(I));
                });
            });

            checkCudaErrors(cudaMalloc((void **)&d_base_change_matrix_,obase_.size() * ibase_.size() * sizeof(uint64_t)));
            for (int i = 0; i <obase_.size();i++) {
                checkCudaErrors(cudaMemcpy(d_base_change_matrix_ + i * ibase_.size(), base_change_matrix_[i].get(), ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }

        }

        RNSTool::RNSTool(
            size_t poly_modulus_degree, const RNSBase &coeff_modulus, const Modulus &plain_modulus,
            MemoryPoolHandle pool)
            : pool_(move(pool))
        {
#ifdef SEAL_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(poly_modulus_degree, coeff_modulus, plain_modulus);
        }

        void RNSTool::ensure_size(uint64_t **input, size_t current_size, size_t &size) const
        {
            if (current_size > size)
            {
                // printf("before size %llu ,Reallocating %llu bytes\n", size,  current_size);
                checkCudaErrors(cudaMalloc((void **)input, current_size * sizeof(uint64_t)));
                size = current_size;
            } 
        }

        void RNSTool::initialize(size_t poly_modulus_degree, const RNSBase &q, const Modulus &t)
        {
            // Return if q is out of bounds
            if (q.size() < SEAL_COEFF_MOD_COUNT_MIN || q.size() > SEAL_COEFF_MOD_COUNT_MAX)
            {
                throw invalid_argument("rnsbase is invalid");
            }

            // Return if coeff_count is not a power of two or out of bounds
            int coeff_count_power = get_power_of_two(poly_modulus_degree);
            if (coeff_count_power < 0 || poly_modulus_degree > SEAL_POLY_MOD_DEGREE_MAX ||
                poly_modulus_degree < SEAL_POLY_MOD_DEGREE_MIN)
            {
                throw invalid_argument("poly_modulus_degree is invalid");
            }

            t_ = t;
            coeff_count_ = poly_modulus_degree;

            // Allocate memory for the bases q, B, Bsk, Bsk U m_tilde, t_gamma
            size_t base_q_size = q.size();

            // In some cases we might need to increase the size of the base B by one, namely we require
            // K * n * t * q^2 < q * prod(B) * m_sk, where K takes into account cross terms when larger size ciphertexts
            // are used, and n is the "delta factor" for the ring. We reserve 32 bits for K * n. Here the coeff modulus
            // primes q_i are bounded to be SEAL_USER_MOD_BIT_COUNT_MAX (60) bits, and all primes in B and m_sk are
            // SEAL_INTERNAL_MOD_BIT_COUNT (61) bits.
            int total_coeff_bit_count = get_significant_bit_count_uint(q.base_prod(), q.size());

            size_t base_B_size = base_q_size;
            if (32 + t_.bit_count() + total_coeff_bit_count >=
                SEAL_INTERNAL_MOD_BIT_COUNT * safe_cast<int>(base_q_size) + SEAL_INTERNAL_MOD_BIT_COUNT)
            {
                base_B_size++;
            }

            size_t base_Bsk_size = add_safe(base_B_size, size_t(1));
            size_t base_Bsk_m_tilde_size = add_safe(base_Bsk_size, size_t(1));

            size_t base_t_gamma_size = 0;

            // Size check
            if (!product_fits_in(coeff_count_, base_Bsk_m_tilde_size))
            {
                throw logic_error("invalid parameters");
            }

            // Sample primes for B and two more primes: m_sk and gamma
            auto baseconv_primes =
                get_primes(mul_safe(size_t(2), coeff_count_), SEAL_INTERNAL_MOD_BIT_COUNT, base_Bsk_m_tilde_size);
            auto baseconv_primes_iter = baseconv_primes.cbegin();
            m_sk_ = *baseconv_primes_iter++;
            gamma_ = *baseconv_primes_iter++;
            vector<Modulus> base_B_primes;
            copy_n(baseconv_primes_iter, base_B_size, back_inserter(base_B_primes));

            // Set m_tilde_ to a non-prime value
            m_tilde_ = uint64_t(1) << 32;

            // Populate the base arrays
            base_q_ = allocate<RNSBase>(pool_, q, pool_);
            base_B_ = allocate<RNSBase>(pool_, base_B_primes, pool_);
            base_Bsk_ = allocate<RNSBase>(pool_, base_B_->extend(m_sk_));
            base_Bsk_m_tilde_ = allocate<RNSBase>(pool_, base_Bsk_->extend(m_tilde_));

            // Set up t-gamma base if t_ is non-zero (using BFV)
            if (!t_.is_zero())
            {
                base_t_gamma_size = 2;
                base_t_gamma_ = allocate<RNSBase>(pool_, vector<Modulus>{ t_, gamma_ }, pool_);
            }

            // Generate the Bsk NTTTables; these are used for NTT after base extension to Bsk
            try
            {
                CreateNTTTables(
                    coeff_count_power, vector<Modulus>(base_Bsk_->base(), base_Bsk_->base() + base_Bsk_size),
                    base_Bsk_ntt_tables_, pool_);

                size_t c_vec_size = mul_safe(coeff_count_, base_Bsk_size);
                uint64_t h_root_powers[c_vec_size];
                for (int i = 0; i < base_Bsk_size; i++)
                {
                    auto wrap_root_powers = base_Bsk_ntt_tables_.get()[i].get_from_root_powers();
                    for (int j = 0; j < coeff_count_; j++)
                    {
                        h_root_powers[i * coeff_count_ + j] = wrap_root_powers[j].operand;
                    }
                }
                cudaPointerAttributes attributes;
                cudaError_t error = cudaPointerGetAttributes(&attributes, d_base_Bsk_root_powers_);
                if (error == cudaSuccess)
                {
                    if (attributes.devicePointer == NULL)
                    {
                        // printf("Pointer is not allocated on the device\n");
                        checkCudaErrors(cudaMalloc((void **)&d_base_Bsk_root_powers_, c_vec_size * sizeof(std::uint64_t)));
                    }
                }

                checkCudaErrors(cudaMemcpy(
                    d_base_Bsk_root_powers_, h_root_powers, c_vec_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            }
            catch (const logic_error &)
            {
                throw logic_error("invalid rns bases");
            }

            if (!t_.is_zero())
            {
                // Set up BaseConvTool for q --> {t}
                base_q_to_t_conv_ = allocate<BaseConverter>(pool_, *base_q_, RNSBase({ t_ }, pool_), pool_);
            }

            // Set up BaseConverter for q --> Bsk
            base_q_to_Bsk_conv_ = allocate<BaseConverter>(pool_, *base_q_, *base_Bsk_, pool_);

            // Set up BaseConverter for q --> {m_tilde}
            base_q_to_m_tilde_conv_ = allocate<BaseConverter>(pool_, *base_q_, RNSBase({ m_tilde_ }, pool_), pool_);

            // Set up BaseConverter for B --> q
            base_B_to_q_conv_ = allocate<BaseConverter>(pool_, *base_B_, *base_q_, pool_);

            // Set up BaseConverter for B --> {m_sk}
            base_B_to_m_sk_conv_ = allocate<BaseConverter>(pool_, *base_B_, RNSBase({ m_sk_ }, pool_), pool_);

            if (base_t_gamma_)
            {
                // Set up BaseConverter for q --> {t, gamma}
                base_q_to_t_gamma_conv_ = allocate<BaseConverter>(pool_, *base_q_, *base_t_gamma_, pool_);
            }

            // Compute prod(B) mod q
            prod_B_mod_q_ = allocate_uint(base_q_size, pool_);
            SEAL_ITERATE(iter(prod_B_mod_q_, base_q_->base()), base_q_size, [&](auto I) {
                get<0>(I) = modulo_uint(base_B_->base_prod(), base_B_size, get<1>(I));
            });

            cudaPointerAttributes attributes;
            cudaError_t error = cudaPointerGetAttributes(&attributes, d_prod_B_mod_q_);
            if (error == cudaSuccess)
            {
                if (attributes.devicePointer == NULL)
                {
                    // printf("Pointer is not allocated on the device\n");
                    checkCudaErrors(cudaMalloc((void **)&d_prod_B_mod_q_, base_q_size * sizeof(uint64_t)));
                }
            }
            checkCudaErrors(cudaMemcpy(d_prod_B_mod_q_, prod_B_mod_q_.get(), base_q_size * sizeof(uint64_t), cudaMemcpyHostToDevice));



            uint64_t temp;

            // Compute prod(q)^(-1) mod Bsk
            inv_prod_q_mod_Bsk_ = allocate<MultiplyUIntModOperand>(base_Bsk_size, pool_);

            uint64_t temp_operand[base_Bsk_size];
            uint64_t temp_quotient[base_Bsk_size];
            for (size_t i = 0; i < base_Bsk_size; i++)
            {
                temp = modulo_uint(base_q_->base_prod(), base_q_size, (*base_Bsk_)[i]);
                if (!try_invert_uint_mod(temp, (*base_Bsk_)[i], temp))
                {
                    throw logic_error("invalid rns bases");
                }
                inv_prod_q_mod_Bsk_[i].set(temp, (*base_Bsk_)[i]);
                temp_operand[i] = inv_prod_q_mod_Bsk_[i].operand;
                temp_quotient[i] = inv_prod_q_mod_Bsk_[i].quotient;
            }


            error = cudaPointerGetAttributes(&attributes, d_inv_prod_q_mod_Bsk_operand_);
            if (error == cudaSuccess)
            {
                if (attributes.devicePointer == NULL)
                {
                    // printf("Pointer is not allocated on the device\n");
                    checkCudaErrors(cudaMalloc((void **)&d_inv_prod_q_mod_Bsk_operand_, base_Bsk_size * sizeof(std::uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_inv_prod_q_mod_Bsk_quotient_, base_Bsk_size * sizeof(std::uint64_t)));
                }
            }
            checkCudaErrors(cudaMemcpy(
                d_inv_prod_q_mod_Bsk_operand_, temp_operand, base_Bsk_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_inv_prod_q_mod_Bsk_quotient_, temp_quotient, base_Bsk_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            // Compute prod(B)^(-1) mod m_sk
            temp = modulo_uint(base_B_->base_prod(), base_B_size, m_sk_);
            if (!try_invert_uint_mod(temp, m_sk_, temp))
            {
                throw logic_error("invalid rns bases");
            }
            inv_prod_B_mod_m_sk_.set(temp, m_sk_);

            // Compute m_tilde^(-1) mod Bsk
            inv_m_tilde_mod_Bsk_ = allocate<MultiplyUIntModOperand>(base_Bsk_size, pool_);
            SEAL_ITERATE(iter(inv_m_tilde_mod_Bsk_, base_Bsk_->base()), base_Bsk_size, [&](auto I) {
                if (!try_invert_uint_mod(barrett_reduce_64(m_tilde_.value(), get<1>(I)), get<1>(I), temp))
                {
                    throw logic_error("invalid rns bases");
                }
                get<0>(I).set(temp, get<1>(I));
            });


            // uint64_t temp_operand[base_Bsk_size];
            // uint64_t temp_quotient[base_Bsk_size];
            for (size_t i = 0; i < base_Bsk_size; i++)
            {
                temp_operand[i] = inv_m_tilde_mod_Bsk_[i].operand;
                temp_quotient[i] = inv_m_tilde_mod_Bsk_[i].quotient;
            }
           
            checkCudaErrors(cudaMalloc((void **)&d_inv_m_tilde_mod_Bsk_operand_, base_Bsk_size * sizeof(std::uint64_t)));
            checkCudaErrors(cudaMalloc((void **)&d_inv_m_tilde_mod_Bsk_quotient_, base_Bsk_size * sizeof(std::uint64_t)));


            checkCudaErrors(cudaMemcpy(
                d_inv_m_tilde_mod_Bsk_operand_, temp_operand, base_Bsk_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_inv_m_tilde_mod_Bsk_quotient_, temp_quotient, base_Bsk_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            // Compute prod(q)^(-1) mod m_tilde
            temp = modulo_uint(base_q_->base_prod(), base_q_size, m_tilde_);
            if (!try_invert_uint_mod(temp, m_tilde_, temp))
            {
                throw logic_error("invalid rns bases");
            }
            neg_inv_prod_q_mod_m_tilde_.set(negate_uint_mod(temp, m_tilde_), m_tilde_);

            // Compute prod(q) mod Bsk
            prod_q_mod_Bsk_ = allocate_uint(base_Bsk_size, pool_);
            SEAL_ITERATE(iter(prod_q_mod_Bsk_, base_Bsk_->base()), base_Bsk_size, [&](auto I) {
                get<0>(I) = modulo_uint(base_q_->base_prod(), base_q_size, get<1>(I));
            });

            error = cudaPointerGetAttributes(&attributes, d_prod_B_mod_Bsk_);
            if (error == cudaSuccess)
            {
                if (attributes.devicePointer == NULL)
                {
                    // printf("Pointer is not allocated on the device\n");
                    checkCudaErrors(cudaMalloc((void **)&d_prod_B_mod_Bsk_, base_Bsk_size * sizeof(uint64_t)));
                }
            }
            checkCudaErrors(cudaMemcpy(d_prod_B_mod_Bsk_, prod_q_mod_Bsk_.get(), base_Bsk_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            if (base_t_gamma_)
            {
                // Compute gamma^(-1) mod t
                if (!try_invert_uint_mod(barrett_reduce_64(gamma_.value(), t_), t_, temp))
                {
                    throw logic_error("invalid rns bases");
                }
                inv_gamma_mod_t_.set(temp, t_);

                // Compute prod({t, gamma}) mod q
                prod_t_gamma_mod_q_ = allocate<MultiplyUIntModOperand>(base_q_size, pool_);
                SEAL_ITERATE(iter(prod_t_gamma_mod_q_, base_q_->base()), base_q_size, [&](auto I) {
                    get<0>(I).set(
                        multiply_uint_mod((*base_t_gamma_)[0].value(), (*base_t_gamma_)[1].value(), get<1>(I)),
                        get<1>(I));
                });

                // Compute -prod(q)^(-1) mod {t, gamma}
                neg_inv_q_mod_t_gamma_ = allocate<MultiplyUIntModOperand>(base_t_gamma_size, pool_);
                SEAL_ITERATE(iter(neg_inv_q_mod_t_gamma_, base_t_gamma_->base()), base_t_gamma_size, [&](auto I) {
                    get<0>(I).operand = modulo_uint(base_q_->base_prod(), base_q_size, get<1>(I));
                    if (!try_invert_uint_mod(get<0>(I).operand, get<1>(I), get<0>(I).operand))
                    {
                        throw logic_error("invalid rns bases");
                    }
                    get<0>(I).set(negate_uint_mod(get<0>(I).operand, get<1>(I)), get<1>(I));
                });
            }

            // Compute q[last]^(-1) mod q[i] for i = 0..last-1
            // This is used by modulus switching and rescaling
            inv_q_last_mod_q_ = allocate<MultiplyUIntModOperand>(base_q_size - 1, pool_);
            SEAL_ITERATE(iter(inv_q_last_mod_q_, base_q_->base()), base_q_size - 1, [&](auto I) {
                if (!try_invert_uint_mod((*base_q_)[base_q_size - 1].value(), get<1>(I), temp))
                {
                    throw logic_error("invalid rns bases");
                }
                get<0>(I).set(temp, get<1>(I));
            });



            uint64_t inv_q_last_mod_q_operand[base_q_size - 1];
            uint64_t inv_q_last_mod_q_quotient[base_q_size - 1];
            for (size_t i = 0; i < base_q_size - 1; i++)
            {
                inv_q_last_mod_q_operand[i] = inv_q_last_mod_q_[i].operand;
                inv_q_last_mod_q_quotient[i] = inv_q_last_mod_q_[i].quotient;
            }


            error = cudaPointerGetAttributes(&attributes, d_inv_q_last_mod_q_operand_);
            if (error == cudaSuccess)
            {
                if (attributes.devicePointer == NULL)
                {
                    // printf("Pointer is not allocated on the device\n");
                    checkCudaErrors(cudaMalloc((void **)&d_inv_q_last_mod_q_operand_, (base_q_size - 1) * sizeof(std::uint64_t)));
                    checkCudaErrors(cudaMalloc((void **)&d_inv_q_last_mod_q_quotient_, (base_q_size - 1) * sizeof(std::uint64_t)));
                }
            }
            checkCudaErrors(cudaMemcpy(
                d_inv_q_last_mod_q_operand_, inv_q_last_mod_q_operand, (base_q_size - 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_inv_q_last_mod_q_quotient_, inv_q_last_mod_q_quotient, (base_q_size - 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));






            if (t_.value() != 0)
            {
                if (!try_invert_uint_mod(base_q_->base()[base_q_size - 1].value(), t_, inv_q_last_mod_t_))
                {
                    throw logic_error("invalid rns bases");
                }

                q_last_mod_t_ = barrett_reduce_64(base_q_->base()[base_q_size - 1].value(), t_);
            }
        }

        void RNSTool::divide_and_round_q_last_inplace(RNSIter input, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            CoeffIter last_input = input[base_q_size - 1];
            
            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;
            add_poly_scalar_coeffmod(last_input, coeff_count_, half, last_modulus, last_input);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            SEAL_ITERATE(iter(input, inv_q_last_mod_q_, base_q_->base()), base_q_size - 1, [&](auto I) {
                // (ct mod qk) mod qi
                modulo_poly_coeffs(last_input, coeff_count_, get<2>(I), temp);

                // Subtract rounding correction here; the negative sign will turn into a plus in the next subtraction
                uint64_t half_mod = barrett_reduce_64(half, get<2>(I));
                sub_poly_scalar_coeffmod(temp, coeff_count_, half_mod, get<2>(I), temp);
                // (ct mod qi) - (ct mod qk) mod qi
                sub_poly_coeffmod(get<0>(I), temp, coeff_count_, get<2>(I), get<0>(I));

                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<0>(I));
            });

        }


        void RNSTool::divide_and_rount_q_last_inplace_cuda(uint64_t *input) const {
            size_t base_q_size = base_q_->size();
            uint64_t *last_input = input + (base_q_size - 1) * coeff_count_;

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;

            eltwiseAddModScalarKernel<<<(coeff_count_ + 255) / 256, 256>>>(last_input, 
                                                                            last_input,
                                                                            half, 
                                                                            last_modulus.value(), 
                                                                            coeff_count_);

            uint64_t *temp = nullptr;
            allocate_gpu<uint64_t>(&temp, coeff_count_);
            for(int i = 0; i < (base_q_size - 1); i++) {
                modulo_poly_coeffs_kernel<<<(coeff_count_ + 255) / 256, 256>>>(last_input, 
                                                                                coeff_count_, 
                                                                                base_q_->base()[i].value(), 
                                                                                base_q_->base()[i].const_ratio().data()[1],
                                                                                temp);
                uint64_t half_mod = barrett_reduce_64(half, base_q_->base()[i]);
                eltwiseSubModScalarKernel<<<(coeff_count_ + 255) / 256, 256>>>(temp, 
                                                                                temp,
                                                                                half_mod,
                                                                                base_q_->base()[i].value(), 
                                                                                coeff_count_                                                                                
                                                                                );

                eltwiseSubModKernel<<<(coeff_count_ + 255) / 256, 256>>>(input + i * coeff_count_, 
                                                                        input + i * coeff_count_, 
                                                                        temp, 
                                                                        base_q_->base()[i].value(), 
                                                                        coeff_count_);

                multiply_poly_scalar_kernel<<<(coeff_count_ + 255) / 256, 256>>>(
                    input + i * coeff_count_, coeff_count_, 1, inv_q_last_mod_q_[i].quotient,
                    inv_q_last_mod_q_[i].operand, base_q_->base()[i].value(), input + i * coeff_count_);
            }

        }

        void RNSTool::divide_and_round_q_last_ntt_inplace(
            RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!rns_ntt_tables)
            {
                throw invalid_argument("rns_ntt_tables cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            CoeffIter last_input = input[base_q_size - 1];

            // Convert to non-NTT form
            inverse_ntt_negacyclic_harvey(last_input, rns_ntt_tables[base_q_size - 1]);

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;
            add_poly_scalar_coeffmod(last_input, coeff_count_, half, last_modulus, last_input);

            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            SEAL_ITERATE(iter(input, inv_q_last_mod_q_, base_q_->base(), rns_ntt_tables), base_q_size - 1, [&](auto I) {
                // (ct mod qk) mod qi
                if (get<2>(I).value() < last_modulus.value())
                {
                    modulo_poly_coeffs(last_input, coeff_count_, get<2>(I), temp);
                }
                else
                {
                    set_uint(last_input, coeff_count_, temp);
                }

                // Lazy subtraction here. ntt_negacyclic_harvey_lazy can take 0 < x < 4*qi input.
                uint64_t neg_half_mod = get<2>(I).value() - barrett_reduce_64(half, get<2>(I));

                // Note: lambda function parameter must be passed by reference here
                SEAL_ITERATE(temp, coeff_count_, [&](auto &J) { J += neg_half_mod; });
#if SEAL_USER_MOD_BIT_COUNT_MAX <= 60
                // Since SEAL uses at most 60-bit moduli, 8*qi < 2^63.
                // This ntt_negacyclic_harvey_lazy results in [0, 4*qi).
                uint64_t qi_lazy = get<2>(I).value() << 2;
                ntt_negacyclic_harvey_lazy(temp, get<3>(I));
#else
                // 2^60 < pi < 2^62, then 4*pi < 2^64, we perfrom one reduction from [0, 4*qi) to [0, 2*qi) after ntt.
                uint64_t qi_lazy = get<2>(I).value() << 1;
                ntt_negacyclic_harvey_lazy(temp, get<3>(I));

                // Note: lambda function parameter must be passed by reference here
                SEAL_ITERATE(temp, coeff_count_, [&](auto &J) {
                    J -= (qi_lazy & static_cast<uint64_t>(-static_cast<int64_t>(J >= qi_lazy)));
                });
#endif
                // Lazy subtraction again, results in [0, 2*qi_lazy),
                // The reduction [0, 2*qi_lazy) -> [0, qi) is done implicitly in multiply_poly_scalar_coeffmod.
                SEAL_ITERATE(iter(get<0>(I), temp), coeff_count_, [&](auto J) { get<0>(J) += qi_lazy - get<1>(J); });

                // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<0>(I));
            });
        }

        void inverse_ntt_cuda_2(
            uint64_t *d_tmp_dest_modq, int coeff_count, int coeff_modulus_size, uint64_t *d_inv_root_powers,
            Modulus coeff_modulus)
        {
            cudaStream_t ntt = 0;
            uint64_t temp_mu;
            for (int i = 0; i < coeff_modulus_size; i++)
            {
                k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus.bit_count() * 2);
                temp_mu = (mu1 / coeff_modulus.value()).low;
                inverseNTT(
                    d_tmp_dest_modq + coeff_count * i, coeff_count, ntt, coeff_modulus.value(), temp_mu,
                    coeff_modulus.bit_count(), d_inv_root_powers + coeff_count * i);
            }
        }

        void RNSTool::divide_and_round_q_last_ntt_inplace_cuda_test(
            uint64_t *d_input, uint64_t *matrix_n1, uint64_t *matrix_n2, uint64_t *matrix_n12, uint64_t *modulu, uint64_t *ratio0, uint64_t *ratio1,
            uint64_t *roots, int *bits, std::pair<int, int> split_coeff,uint64_t *d_inv_root_powers, ConstNTTTablesIter rns_ntt_tables) const
        {
            size_t base_q_size = base_q_->size();
            uint64_t *last_input = d_input + (base_q_size - 1) * coeff_count_;
            uint64_t *d_temp = nullptr;
            checkCudaErrors(cudaMalloc((void **)&d_temp, coeff_count_ * sizeof(uint64_t)));

            uint64_t *ntt_temp = nullptr;
            checkCudaErrors(cudaMalloc((void **)&ntt_temp, coeff_count_ * sizeof(uint64_t)));

            // Convert to non-NTT form
            inverse_ntt_cuda_2(
                last_input, coeff_count_, 1, d_inv_root_powers + (base_q_size - 1) * coeff_count_,
                rns_ntt_tables[base_q_size - 1].modulus());
            // rns_ntt_tables[base_q_size - 1].get_inv_root(), rns_ntt_tables[base_q_size - 1].coeff_count_power());

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;

            const std::size_t threadsPerBlock = 256;
            const std::size_t blocksPerGrid = (coeff_count_ + threadsPerBlock - 1) / threadsPerBlock;
            eltwiseAddModScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(
                last_input, last_input, half, last_modulus.value(), coeff_count_);

            const int threads_per_block = 256;
            const int blocks_per_grid = (coeff_count_ + threads_per_block - 1) / threads_per_block;
            
            dim3 block(16, 16);
            int n1 = split_coeff.first, n2 = split_coeff.second;
            dim3 grid_batch((n2 - 1) / block.x + 1, (n1 - 1) / block.y + 1);

            for (int i = 0; i < base_q_size - 1; i++)
            {
                if (base_q_->base()[i].value() < last_modulus.value())
                {
                    modulo_poly_coeffs_kernel<<<blocks_per_grid, threads_per_block>>>(
                        last_input, coeff_count_, base_q_->base()[i].value(),
                        base_q_->base()[i].const_ratio().data()[1], d_temp);
                }
                else
                {
                    fill_kernel<<<blocks_per_grid, threads_per_block>>>(last_input, coeff_count_, d_temp);
                }

                // Lazy subtraction here. ntt_negacyclic_harvey_lazy can take 0 < x < 4*qi input.
                uint64_t neg_half_mod = base_q_->base()[i].value() - barrett_reduce_64(half, base_q_->base()[i]);



                // Note: lambda function parameter must be passed by reference here
                add_neg_kernel<<<blocks_per_grid, threads_per_block>>>(d_temp, coeff_count_, neg_half_mod);
                uint64_t qi_lazy = base_q_->base()[i].value() << 2;
              
                matrix_multi_elemul_merge_batch_test<<<grid_batch, block>>>(matrix_n1 + i * n1 * n1, 
                                                                        d_temp , 
                                                                        ntt_temp,
                                                                        matrix_n12 + i * n1 * n2,
                                                                        n1, n1, n2,
                                                                        1,
                                                                        modulu + i,
                                                                        ratio0 + i,
                                                                        ratio1 + i,
                                                                        roots +i
                                                                        );

                matrix_multi_transpose_batch<<<grid_batch, block>>>(ntt_temp, 
                                                                    matrix_n2 + i * n2 * n2, 
                                                                    d_temp, 
                                                                    n1, n2, n2, 
                                                                    1,
                                                                    modulu + i, 
                                                                    bits + i, 
                                                                    ratio0 + i,
                                                                    ratio1 +i
                                                                    );

                add_qi_lazy_kernel<<<blocks_per_grid, threads_per_block>>>(
                    d_input + i * coeff_count_, d_temp, coeff_count_, qi_lazy);

                multiply_poly_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(
                    d_input + i * coeff_count_, coeff_count_, 1, inv_q_last_mod_q_[i].quotient,
                    inv_q_last_mod_q_[i].operand, base_q_->base()[i].value(), d_input + i * coeff_count_);
            }
        }

        void RNSTool::divide_and_round_q_last_ntt_inplace_cuda_v1(
            uint64_t *d_input, uint64_t *d_root_powers, uint64_t *d_inv_root_powers, ConstNTTTablesIter rns_ntt_tables, cudaStream_t *streams, int stream_num) const
        {
            size_t base_q_size = base_q_->size();
            uint64_t *last_input = d_input + (base_q_size - 1) * coeff_count_;

            // ensure_size(&d_temp_, coeff_count_ * (base_q_size - 1), d_temp_size_);
            // uint64_t *d_temp = d_temp_;

            uint64_t *d_temp = nullptr;
            allocate_gpu<uint64_t>(&d_temp, coeff_count_ * (base_q_size - 1));


            // Convert to non-NTT form
            inverse_ntt_cuda_2(
                last_input, coeff_count_, 1, d_inv_root_powers + (base_q_size - 1) * coeff_count_,
                rns_ntt_tables[base_q_size - 1].modulus());
            // rns_ntt_tables[base_q_size - 1].get_inv_root(), rns_ntt_tables[base_q_size - 1].coeff_count_power());

            // Add (qi-1)/2 to change from flooring to rounding
            Modulus last_modulus = (*base_q_)[base_q_size - 1];
            uint64_t half = last_modulus.value() >> 1;

            const std::size_t threadsPerBlock = 256;
            const std::size_t blocksPerGrid = (coeff_count_ + threadsPerBlock - 1) / threadsPerBlock;
            eltwiseAddModScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(
                last_input, last_input, half, last_modulus.value(), coeff_count_);

            const int threads_per_block = 256;
            const int blocks_per_grid = (coeff_count_ + threads_per_block - 1) / threads_per_block;

            uint64_t temp_mu;
            k_uint128_t mu1;
            for (int i = 0; i < base_q_size - 1; i++)
            {
                uint64_t qi_lazy = base_q_->base()[i].value() << 2;
                uint64_t neg_half_mod = base_q_->base()[i].value() - barrett_reduce_64(half, base_q_->base()[i]);

                k_uint128_t mu1 = k_uint128_t::exp2(rns_ntt_tables[i].modulus().bit_count() * 2);
                temp_mu = (mu1 / rns_ntt_tables[i].modulus().value()).low;

                if (base_q_->base()[i].value() < last_modulus.value())
                {
                    modulo_poly_coeffs_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i % stream_num]>>>(
                        last_input, coeff_count_, base_q_->base()[i].value(),
                        base_q_->base()[i].const_ratio().data()[1], d_temp + i * coeff_count_);
                }
                else
                {
                    fill_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i % stream_num]>>>(last_input, coeff_count_, d_temp + i * coeff_count_);
                }

                add_neg_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i % stream_num]>>>(d_temp + i * coeff_count_, coeff_count_, neg_half_mod);

                forwardNTT(
                    d_temp + i * coeff_count_, coeff_count_, streams[i % stream_num], rns_ntt_tables[i].modulus().value(), temp_mu,
                    rns_ntt_tables[i].modulus().bit_count(), d_root_powers + i * coeff_count_);

                add_qi_lazy_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i % stream_num]>>>(
                    d_input + i * coeff_count_, d_temp + i * coeff_count_, coeff_count_, qi_lazy);

                multiply_poly_scalar_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i % stream_num]>>>(
                    d_input + i * coeff_count_, coeff_count_, 1, inv_q_last_mod_q_[i].quotient,
                    inv_q_last_mod_q_[i].operand, base_q_->base()[i].value(), d_input + i * coeff_count_);
            }
            cudaDeviceSynchronize();
            deallocate_gpu<uint64_t>(&d_temp, coeff_count_ * (base_q_size - 1));

        }

        void RNSTool::fastbconv_sk(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (!input)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base Bsk
            Ensure: Output in base q
            */

            size_t base_q_size = base_q_->size();
            size_t base_B_size = base_B_->size();

            // Fast convert B -> q; input is in Bsk but we only use B
            base_B_to_q_conv_->fast_convert_array(input, destination, pool);

            // Compute alpha_sk
            // Fast convert B -> {m_sk}; input is in Bsk but we only use B
            SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_count_, pool);
            base_B_to_m_sk_conv_->fast_convert_array(input, RNSIter(temp, coeff_count_), pool);

            // Take the m_sk part of input, subtract from temp, and multiply by inv_prod_B_mod_m_sk_
            // Note: input_sk is allocated in input[base_B_size]
            SEAL_ALLOCATE_GET_COEFF_ITER(alpha_sk, coeff_count_, pool);
            SEAL_ITERATE(iter(alpha_sk, temp, input[base_B_size]), coeff_count_, [&](auto I) {
                // It is not necessary for the negation to be reduced modulo the small prime
                get<0>(I) = multiply_uint_mod(get<1>(I) + (m_sk_.value() - get<2>(I)), inv_prod_B_mod_m_sk_, m_sk_);
            });

            // alpha_sk is now ready for the Shenoy-Kumaresan conversion; however, note that our
            // alpha_sk here is not a centered reduction, so we need to apply a correction below.
            const uint64_t m_sk_div_2 = m_sk_.value() >> 1;
            SEAL_ITERATE(iter(prod_B_mod_q_, base_q_->base(), destination), base_q_size, [&](auto I) {
                // Set up the multiplication helpers
                MultiplyUIntModOperand prod_B_mod_q_elt;
                prod_B_mod_q_elt.set(get<0>(I), get<1>(I));

                MultiplyUIntModOperand neg_prod_B_mod_q_elt;
                neg_prod_B_mod_q_elt.set(get<1>(I).value() - get<0>(I), get<1>(I));

                SEAL_ITERATE(iter(alpha_sk, get<2>(I)), coeff_count_, [&](auto J) {
                    // Correcting alpha_sk since it represents a negative value
                    if (get<0>(J) > m_sk_div_2)
                    {
                        get<1>(J) = multiply_add_uint_mod(
                            negate_uint_mod(get<0>(J), m_sk_), prod_B_mod_q_elt, get<1>(J), get<1>(I));
                    }
                    // No correction needed
                    else
                    {
                        // It is not necessary for the negation to be reduced modulo the small prime
                        get<1>(J) = multiply_add_uint_mod(get<0>(J), neg_prod_B_mod_q_elt, get<1>(J), get<1>(I));
                    }
                });
            });
        }

        void RNSTool::fastbconv_sk_cuda(uint64_t *d_in, uint64_t *d_destination) const
        {
            /*
            Require: Input in base Bsk
            Ensure: Output in base q
            */

            size_t base_q_size = base_q_->size();
            size_t base_B_size = base_B_->size();

            // ensure_size(&d_temp_, coeff_count_, d_temp_size_);
            allocate_gpu<uint64_t>(&d_temp_, coeff_count_);

            uint64_t *d_base_q_value = base_q_->d_base();
            uint64_t *d_base_q_ratio = base_q_->d_ratio1();


            uint64_t *d_alpha_sk = nullptr;
            allocate_gpu<uint64_t>(&d_alpha_sk, coeff_count_);
            // checkCudaErrors(cudaMalloc((void **)&d_alpha_sk, coeff_count_ * sizeof(uint64_t)));

// 计算过程
            // Fast convert B -> q; input is in Bsk but we only use B
            base_B_to_q_conv_->fast_convert_array_cuda(d_in, d_destination, coeff_count_);

            // Compute alpha_sk
            // Fast convert B -> {m_sk}; input is in Bsk but we only use B
            base_B_to_m_sk_conv_->fast_convert_array_cuda(d_in, d_temp_, coeff_count_);

            // Take the m_sk part of input, subtract from temp, and multiply by inv_prod_B_mod_m_sk_
            // Note: input_sk is allocated in input[base_B_size]

// 两个可以合一起
            fastbconv_sk_kernel_helper1<<<(coeff_count_ + 255) / 256, 256>>>(d_in + base_B_size * coeff_count_, 
                                                                            d_temp_, coeff_count_, 
                                                                            d_alpha_sk, m_sk_.value(), 
                                                                            inv_prod_B_mod_m_sk_.operand, 
                                                                            inv_prod_B_mod_m_sk_.quotient);

            fastbconv_sk_kernel_helper2<<<(base_q_size * coeff_count_ + 255) / 256, 256>>>(d_alpha_sk, d_destination, 
                                                                                            base_q_size, coeff_count_,
                                                                                             m_sk_.value() >> 1, m_sk_.value(), 
                                                                                             d_prod_B_mod_q_, 
                                                                                            d_base_q_value, d_base_q_ratio);
            deallocate_gpu<uint64_t>(&d_alpha_sk, coeff_count_);
            deallocate_gpu<uint64_t>(&d_temp_, coeff_count_);

        }

        void RNSTool::sm_mrq(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base Bsk U {m_tilde}
            Ensure: Output in base Bsk
            */

            size_t base_Bsk_size = base_Bsk_->size();

            // The last component of the input is mod m_tilde
            ConstCoeffIter input_m_tilde = input[base_Bsk_size];
            const uint64_t m_tilde_div_2 = m_tilde_.value() >> 1;

            // Compute r_m_tilde
            SEAL_ALLOCATE_GET_COEFF_ITER(r_m_tilde, coeff_count_, pool);
            multiply_poly_scalar_coeffmod(
                input_m_tilde, coeff_count_, neg_inv_prod_q_mod_m_tilde_, m_tilde_, r_m_tilde);

            SEAL_ITERATE(
                iter(input, prod_q_mod_Bsk_, inv_m_tilde_mod_Bsk_, base_Bsk_->base(), destination), base_Bsk_size,
                [&](auto I) {
                    MultiplyUIntModOperand prod_q_mod_Bsk_elt;
                    prod_q_mod_Bsk_elt.set(get<1>(I), get<3>(I));
                    SEAL_ITERATE(iter(get<0>(I), r_m_tilde, get<4>(I)), coeff_count_, [&](auto J) {
                        // We need centered reduction of r_m_tilde modulo Bsk. Note that m_tilde is chosen
                        // to be a power of two so we have '>=' below.
                        uint64_t temp = get<1>(J);
                        if (temp >= m_tilde_div_2)
                        {
                            temp += get<3>(I).value() - m_tilde_.value();
                        }

                        // Compute (input + q*r_m_tilde)*m_tilde^(-1) mod Bsk
                        get<2>(J) = multiply_uint_mod(
                            multiply_add_uint_mod(temp, prod_q_mod_Bsk_elt, get<0>(J), get<3>(I)), get<2>(I),
                            get<3>(I));
                    });
                });

        }

        void RNSTool::sm_mrq_cuda(uint64_t *input, uint64_t *destination) const
        {
            /*
            Require: Input in base Bsk U {m_tilde}
            Ensure: Output in base Bsk
            */

            size_t base_Bsk_size = base_Bsk_->size();
            
            uint64_t *d_prod_B_mod_Bsk = d_prod_B_mod_Bsk_;
            uint64_t *d_base_Bsk_value = base_Bsk_->d_base();
            uint64_t *d_base_Bsk_ratio = base_Bsk_->d_ratio1();

            multiply_poly_scalar_coeffmod_kernel_rns<<<(coeff_count_ + 255) / 256, 256>>>(
                input + base_Bsk_size * coeff_count_, coeff_count_, neg_inv_prod_q_mod_m_tilde_.operand, 
                neg_inv_prod_q_mod_m_tilde_.quotient, m_tilde_.value(), 
                input + base_Bsk_size * coeff_count_);



            sm_mrq_kernel<<<(base_Bsk_size * coeff_count_ + 255) / 256, 256>>>(input, input + base_Bsk_size * coeff_count_, destination,
                                                                                base_Bsk_size, coeff_count_, m_tilde_.value(),
                                                                                d_prod_B_mod_Bsk, 
                                                                                d_base_Bsk_value, d_base_Bsk_ratio, 
                                                                                d_inv_m_tilde_mod_Bsk_operand_, d_inv_m_tilde_mod_Bsk_quotient_);
           

        }

        void RNSTool::fast_floor(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in base q U Bsk
            Ensure: Output in base Bsk
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();

            // Convert q -> Bsk
            base_q_to_Bsk_conv_->fast_convert_array(input, destination, pool);

            // Move input pointer to past the base q components
            input += base_q_size;
            SEAL_ITERATE(iter(input, inv_prod_q_mod_Bsk_, base_Bsk_->base(), destination), base_Bsk_size, [&](auto I) {
                SEAL_ITERATE(iter(get<0>(I), get<3>(I)), coeff_count_, [&](auto J) {
                    // It is not necessary for the negation to be reduced modulo base_Bsk_elt
                    get<1>(J) = multiply_uint_mod(get<0>(J) + (get<2>(I).value() - get<1>(J)), get<1>(I), get<2>(I));
                });
            });
        }

        void RNSTool::fast_floor_cuda(uint64_t *d_in, uint64_t *d_destination) const
        {

            /*
            Require: Input in base q U Bsk
            Ensure: Output in base Bsk
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();
            size_t count = coeff_count_;

            uint64_t *d_base_bsk_value = base_Bsk_->d_base();
            
            
            base_q_to_Bsk_conv_->fast_convert_array_cuda(d_in, d_destination, count);

            fast_floor_kernel<<<(count * base_Bsk_size + 255) / 256, 256>>>(
                d_in + base_q_size * count , d_destination, 
                base_Bsk_size, count, 
                d_inv_prod_q_mod_Bsk_operand_, d_inv_prod_q_mod_Bsk_quotient_, 
                d_base_bsk_value);


        }

        void RNSTool::fastbconv_m_tilde(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (destination.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            /*
            Require: Input in q
            Ensure: Output in Bsk U {m_tilde}
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();

            // We need to multiply first the input with m_tilde mod q
            // This is to facilitate Montgomery reduction in the next step of multiplication
            // This is NOT an ideal approach: as mentioned in BEHZ16, multiplication by
            // m_tilde can be easily merge into the base conversion operation; however, then
            // we could not use the BaseConverter as below without modifications.
            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count_, base_q_size, pool);
            multiply_poly_scalar_coeffmod(input, base_q_size, m_tilde_.value(), base_q_->base(), temp);   

            // Now convert to Bsk
            base_q_to_Bsk_conv_->fast_convert_array(temp, destination, pool);

            // Finally convert to {m_tilde}
            base_q_to_m_tilde_conv_->fast_convert_array(temp, destination + base_Bsk_size, pool);
        }

        void RNSTool::fastbconv_m_tilde_cuda(uint64_t *input, uint64_t *destination) const
        {
            /*
            Require: Input in q
            Ensure: Output in Bsk U {m_tilde}
            */

            size_t base_q_size = base_q_->size();
            size_t base_Bsk_size = base_Bsk_->size();

            // We need to multiply first the input with m_tilde mod q
            // This is to facilitate Montgomery reduction in the next step of multiplication
            // This is NOT an ideal approach: as mentioned in BEHZ16, multiplication by
            // m_tilde can be easily merge into the base conversion operation; however, then
            // we could not use the BaseConverter as below without modifications.

            ensure_size(&d_temp_, base_q_size * coeff_count_, d_temp_size_);


            uint64_t *d_base_q_value = base_q_->d_base();
            uint64_t *d_base_q_ratio = base_q_->d_ratio1();
            
            multiply_poly_scalar_coeffmod_kernel_kernel<<<(base_q_size * coeff_count_ + 255) / 256, 256>>> (
                input, d_temp_, coeff_count_, base_q_size, 
                d_base_q_value, d_base_q_ratio, m_tilde_.value());

            // Now convert to Bsk
            base_q_to_Bsk_conv_->fast_convert_array_cuda(d_temp_, destination, coeff_count_);

            // Finally convert to {m_tilde}
            base_q_to_m_tilde_conv_->fast_convert_array_cuda(d_temp_, destination + base_Bsk_size * coeff_count_, coeff_count_);
        }

        void RNSTool::decrypt_scale_and_round(ConstRNSIter input, CoeffIter destination, MemoryPoolHandle pool) const
        {
#ifdef SEAL_DEBUG
            if (input == nullptr)
            {
                throw invalid_argument("input cannot be null");
            }
            if (input.poly_modulus_degree() != coeff_count_)
            {
                throw invalid_argument("input is not valid for encryption parameters");
            }
            if (!destination)
            {
                throw invalid_argument("destination cannot be null");
            }
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            size_t base_q_size = base_q_->size();
            size_t base_t_gamma_size = base_t_gamma_->size();

            // Compute |gamma * t|_qi * ct(s)
            SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count_, base_q_size, pool);
            SEAL_ITERATE(iter(input, prod_t_gamma_mod_q_, base_q_->base(), temp), base_q_size, [&](auto I) {
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<3>(I));
            });

            // Make another temp destination to get the poly in mod {t, gamma}
            SEAL_ALLOCATE_GET_RNS_ITER(temp_t_gamma, coeff_count_, base_t_gamma_size, pool);

            // Convert from q to {t, gamma}
            base_q_to_t_gamma_conv_->fast_convert_array(temp, temp_t_gamma, pool);

            // Multiply by -prod(q)^(-1) mod {t, gamma}
            SEAL_ITERATE(
                iter(temp_t_gamma, neg_inv_q_mod_t_gamma_, base_t_gamma_->base(), temp_t_gamma), base_t_gamma_size,
                [&](auto I) {
                    multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<1>(I), get<2>(I), get<3>(I));
                });

            // Need to correct values in temp_t_gamma (gamma component only) which are
            // larger than floor(gamma/2)
            uint64_t gamma_div_2 = (*base_t_gamma_)[1].value() >> 1;

            // Now compute the subtraction to remove error and perform final multiplication by
            // gamma inverse mod t
            SEAL_ITERATE(iter(temp_t_gamma[0], temp_t_gamma[1], destination), coeff_count_, [&](auto I) {
                // Need correction because of centered mod
                if (get<1>(I) > gamma_div_2)
                {
                    // Compute -(gamma - a) instead of (a - gamma)
                    get<2>(I) = add_uint_mod(get<0>(I), barrett_reduce_64(gamma_.value() - get<1>(I), t_), t_);
                }
                // No correction needed
                else
                {
                    get<2>(I) = sub_uint_mod(get<0>(I), barrett_reduce_64(get<1>(I), t_), t_);
                }

                // If this coefficient was non-zero, multiply by gamma^(-1)
                if (0 != get<2>(I))
                {
                    // Perform final multiplication by gamma inverse mod t
                    get<2>(I) = multiply_uint_mod(get<2>(I), inv_gamma_mod_t_, t_);
                }
            });
        }

        void RNSTool::decrypt_scale_and_round_cuda(uint64_t *input, uint64_t *destination) const{
            size_t base_q_size = base_q_->size();
            size_t base_t_gamma_size = base_t_gamma_->size();

            uint64_t *d_temp = nullptr;
            checkCudaErrors(cudaMalloc((void **)&d_temp, coeff_count_ * base_q_size * sizeof(uint64_t)));
            for(int i = 0; i < base_q_size; i++){
                multiply_poly_scalar_coeffmod_kernel_one_modulu<<<(coeff_count_ + 255) / 256, 256>>>(
                    input + i * coeff_count_, d_temp + i * coeff_count_, coeff_count_, prod_t_gamma_mod_q_[i].operand, prod_t_gamma_mod_q_[i].quotient, 
                    base_q_->base()[i].value());
            }

            uint64_t *d_temp_t_gamma = nullptr;
            checkCudaErrors(cudaMalloc((void **)&d_temp_t_gamma, coeff_count_ * base_t_gamma_size * sizeof(uint64_t)));
            base_q_to_t_gamma_conv_->fast_convert_array_cuda(d_temp, d_temp_t_gamma, coeff_count_);

        }

        void RNSTool::mod_t_and_divide_q_last_ntt_inplace(
            RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const
        {
            size_t modulus_size = base_q_->size();
            const Modulus *curr_modulus = base_q_->base();
            const Modulus plain_modulus = t_;
            uint64_t last_modulus_value = curr_modulus[modulus_size - 1].value();

            SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(neg_c_last_mod_t, coeff_count_, pool);
            // neg_c_last_mod_t = - c_last (mod t)
            CoeffIter c_last = input[modulus_size - 1];
            inverse_ntt_negacyclic_harvey(c_last, rns_ntt_tables[modulus_size - 1]);

            modulo_poly_coeffs(c_last, coeff_count_, plain_modulus, neg_c_last_mod_t);
            negate_poly_coeffmod(neg_c_last_mod_t, coeff_count_, plain_modulus, neg_c_last_mod_t);
            if (inv_q_last_mod_t_ != 1)
            {
                // neg_c_last_mod_t *= q_last^(-1) (mod t)
                multiply_poly_scalar_coeffmod(
                    neg_c_last_mod_t, coeff_count_, inv_q_last_mod_t_, plain_modulus, neg_c_last_mod_t);
            }

            SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(delta_mod_q_i, coeff_count_, pool);

            SEAL_ITERATE(iter(input, curr_modulus, inv_q_last_mod_q_, rns_ntt_tables), modulus_size - 1, [&](auto I) {
                // delta_mod_q_i = neg_c_last_mod_t (mod q_i)
                modulo_poly_coeffs(neg_c_last_mod_t, coeff_count_, get<1>(I), delta_mod_q_i);

                // delta_mod_q_i *= q_last (mod q_i)
                multiply_poly_scalar_coeffmod(
                    delta_mod_q_i, coeff_count_, last_modulus_value, get<1>(I), delta_mod_q_i);

                // c_i = c_i - c_last - neg_c_last_mod_t * q_last (mod 2q_i)
                SEAL_ITERATE(iter(delta_mod_q_i, c_last), coeff_count_, [&](auto J) {
                    get<0>(J) = add_uint_mod(get<0>(J), barrett_reduce_64(get<1>(J), get<1>(I)), get<1>(I));
                });
                ntt_negacyclic_harvey(delta_mod_q_i, get<3>(I));
                SEAL_ITERATE(iter(get<0>(I), delta_mod_q_i), coeff_count_, [&](auto J) {
                    get<0>(J) = sub_uint_mod(get<0>(J), get<1>(J), get<1>(I));
                });

                // c_i = c_i * inv_q_last_mod_q_i (mod q_i)
                multiply_poly_scalar_coeffmod(get<0>(I), coeff_count_, get<2>(I), get<1>(I), get<0>(I));
            });
        }

        __device__ void  multiply_poly_scalar_coeffmod_helper_kernelr(uint64_t poly, uint64_t scalar, uint64_t modulus_value, uint64_t ratio1, uint64_t *result) {
            
            uint64_t operand = barrett_reduce_64_kernel(scalar, modulus_value, ratio1);

            std::uint64_t wide_quotient[2]{ 0, 0 };
            std::uint64_t wide_coeff[2]{ 0, operand };
            divide_uint128_inplace_kernel(wide_coeff, modulus_value, wide_quotient);
            uint64_t quotient = wide_quotient[0];

            unsigned long long tmp1, tmp2;
            multiply_uint64_hw64_kernel(poly, quotient, &tmp1);
            tmp2 = operand * poly - tmp1 * modulus_value;
            *result = tmp2 >= modulus_value ? tmp2 - modulus_value : tmp2;
        }

        __global__ void mod_t_and_divide_q_helper(uint64_t *input, uint64_t *result,
        size_t coeff_count, size_t modulu_size,
        uint64_t *modulu_value, uint64_t *modulu_ratio1,
        uint64_t plain_value, uint64_t plain_ratio1,
        uint64_t inv_q_last_mod_t){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < coeff_count * modulu_size) {
                size_t coeff_idx = idx % coeff_count;
                size_t modulu_idx = idx / coeff_count;

                uint64_t neg_c_last_mod_t = barrett_reduce_64_kernel(input[coeff_idx], plain_value, plain_ratio1);

                neg_c_last_mod_t = (plain_value - neg_c_last_mod_t) & static_cast<std::uint64_t>(-(neg_c_last_mod_t != 0));
                if (inv_q_last_mod_t != 1){
                    multiply_poly_scalar_coeffmod_helper_kernelr(neg_c_last_mod_t, inv_q_last_mod_t, plain_value, plain_ratio1, &neg_c_last_mod_t);
                }

                uint64_t delta_mod_q_i = barrett_reduce_64_kernel(neg_c_last_mod_t, modulu_value[modulu_idx], modulu_ratio1[modulu_idx]);

                multiply_poly_scalar_coeffmod_helper_kernelr(delta_mod_q_i, modulu_value[modulu_size],  modulu_value[modulu_idx], modulu_ratio1[modulu_idx], &delta_mod_q_i);

                delta_mod_q_i = add_uint_mod_kernel(delta_mod_q_i, input[coeff_idx], modulu_value[modulu_idx]);

                result[idx] = delta_mod_q_i;

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void mod_t_and_divide_q_helper2( uint64_t *input, uint64_t *operand, uint64_t *destination, 
                                size_t coeff_count, size_t modulu_size,
                                uint64_t *modulus_value, uint64_t *modulus_ratio,
                                uint64_t *modq_operand, uint64_t *modq_quotient)
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while(index < coeff_count*modulu_size)
            {
                size_t modulu_index = index / coeff_count;

                uint64_t x = sub_uint_mod_kernel(input[index], operand[index], modulus_value[modulu_index]);

                destination[index] = multiply_uint_mod_kernel(x, modq_quotient[modulu_index], modq_operand[modulu_index], modulus_value[modulu_index]);

                index += blockDim.x * gridDim.x;
            }
        }

        void RNSTool::mod_t_and_divide_q_last_ntt_inplace_cuda(
            uint64_t *input, uint64_t *matrix_n1, uint64_t *matrix_n2, uint64_t *matrix_n12,uint64_t *modulu, uint64_t *ratio0, uint64_t *ratio1,
            uint64_t *roots, int *bits, std::pair<int, int> split_coeff,uint64_t *d_inv_root_powers,ConstNTTTablesIter rns_ntt_tables) const
        {

            size_t modulus_size = base_q_->size();
            const Modulus *curr_modulus = base_q_->base();
            const Modulus plain_modulus = t_;
            uint64_t last_modulus_value = curr_modulus[modulus_size - 1].value();

            ensure_size(&d_temp_, coeff_count_ * (modulus_size - 1), d_temp_size_);

            uint64_t *c_last = input + (modulus_size - 1) * coeff_count_;

            cudaStream_t ntt = 0;
            uint64_t temp_mu;
            k_uint128_t mu1 = k_uint128_t::exp2(curr_modulus[modulus_size - 1].bit_count() * 2);
            temp_mu = (mu1 / curr_modulus[modulus_size - 1].value()).low;
            inverseNTT(
                c_last, coeff_count_, ntt, rns_ntt_tables[modulus_size - 1].modulus().value(), temp_mu,
                rns_ntt_tables[modulus_size - 1].modulus().bit_count(), d_inv_root_powers + coeff_count_ * (modulus_size - 1));

            uint64_t plain_value = plain_modulus.value();
            uint64_t plain_ratio1 = plain_modulus.const_ratio().data()[1];


            ensure_size(&d_input_m_tilde_, (modulus_size - 1) * coeff_count_, d_input_m_tilde_size_);
            uint64_t *d_delta_mod_q_i = d_input_m_tilde_;


            mod_t_and_divide_q_helper<<<(coeff_count_ * (modulus_size - 1) + 255) / 256, 256>>>(c_last, d_delta_mod_q_i,
                                                                                                coeff_count_, (modulus_size - 1),
                                                                                                base_q_->d_base(), base_q_->d_ratio1(),
                                                                                                plain_value, plain_ratio1,
                                                                                                inv_q_last_mod_t_);




            dim3 block(16, 16);
            int n1 = split_coeff.first, n2 = split_coeff.second;
            dim3 grid_batch((n2 - 1) / block.x + 1, (n1* (modulus_size - 1) - 1) / block.y + 1);
            
            // matrix_multi_elemul_merge_batch<<<grid_batch, block>>>(matrix_n1, 
            //                                                     d_delta_mod_q_i, 
            //                                                     d_temp_,
            //                                                     n1, n1, n2,
            //                                                     (modulus_size - 1),
            //                                                     modulu,
            //                                                     ratio0,
            //                                                     ratio1,
            //                                                     roots
            //                                                 );
            matrix_multi_elemul_merge_batch_test<<<grid_batch, block>>>(matrix_n1, 
                                                                d_delta_mod_q_i, 
                                                                d_temp_,
                                                                matrix_n12,
                                                                n1, n1, n2,
                                                                (modulus_size - 1),
                                                                modulu,
                                                                ratio0,
                                                                ratio1,
                                                                roots
                                                            );

            matrix_multi_transpose_batch<<<grid_batch, block>>>(
                                                            d_temp_, 
                                                            matrix_n2, 
                                                            d_delta_mod_q_i, 
                                                            n1, n2, n2, 
                                                            (modulus_size - 1),
                                                            modulu, 
                                                            bits, 
                                                            ratio0,
                                                            ratio1);



            mod_t_and_divide_q_helper2<<<(coeff_count_ * (modulus_size - 1) + 255) / 256, 256>>>(input, d_delta_mod_q_i, input,
                                                                                                coeff_count_, (modulus_size - 1), 
                                                                                                base_q_->d_base(),base_q_->d_ratio1(),
                                                                                                d_inv_q_last_mod_q_operand_,
                                                                                                d_inv_q_last_mod_q_quotient_);

        }

        void RNSTool::mod_t_and_divide_q_last_ntt_inplace_cuda_v1(
            uint64_t *input, uint64_t *d_root_powers, uint64_t *d_inv_root_powers,ConstNTTTablesIter rns_ntt_tables, cudaStream_t *streams, int stream_num) const
        {

            size_t modulus_size = base_q_->size();
            const Modulus *curr_modulus = base_q_->base();
            const Modulus plain_modulus = t_;
            uint64_t last_modulus_value = curr_modulus[modulus_size - 1].value();

            ensure_size(&d_temp_, coeff_count_ * (modulus_size - 1), d_temp_size_);

            uint64_t *c_last = input + (modulus_size - 1) * coeff_count_;

            cudaStream_t ntt = 0;
            uint64_t temp_mu;
            k_uint128_t mu1 = k_uint128_t::exp2(curr_modulus[modulus_size - 1].bit_count() * 2);
            temp_mu = (mu1 / curr_modulus[modulus_size - 1].value()).low;
            inverseNTT(
                c_last, coeff_count_, ntt, rns_ntt_tables[modulus_size - 1].modulus().value(), temp_mu,
                rns_ntt_tables[modulus_size - 1].modulus().bit_count(), d_inv_root_powers + coeff_count_ * (modulus_size - 1));

            uint64_t plain_value = plain_modulus.value();
            uint64_t plain_ratio1 = plain_modulus.const_ratio().data()[1];


            ensure_size(&d_input_m_tilde_, (modulus_size - 1) * coeff_count_, d_input_m_tilde_size_);
            uint64_t *d_delta_mod_q_i = d_input_m_tilde_;


            mod_t_and_divide_q_helper<<<(coeff_count_ * (modulus_size - 1) + 255) / 256, 256>>>(c_last, d_delta_mod_q_i,
                                                                                                coeff_count_, (modulus_size - 1),
                                                                                                base_q_->d_base(), base_q_->d_ratio1(),
                                                                                                plain_value, plain_ratio1,
                                                                                                inv_q_last_mod_t_);



            for(int i = 0; i < (modulus_size - 1); i++){

                k_uint128_t mu1 = k_uint128_t::exp2(rns_ntt_tables[i].modulus().bit_count() * 2);
                temp_mu = (mu1 / rns_ntt_tables[i].modulus().value()).low;

                forwardNTT(
                    d_delta_mod_q_i + i * coeff_count_, coeff_count_, streams[i % stream_num], rns_ntt_tables[i].modulus().value(), temp_mu,
                    rns_ntt_tables[i].modulus().bit_count(), d_root_powers + i * coeff_count_);

            }

            cudaDeviceSynchronize();

            mod_t_and_divide_q_helper2<<<(coeff_count_ * (modulus_size - 1) + 255) / 256, 256>>>(input, d_delta_mod_q_i, input,
                                                                                                coeff_count_, (modulus_size - 1), 
                                                                                                base_q_->d_base(),base_q_->d_ratio1(),
                                                                                                d_inv_q_last_mod_q_operand_,
                                                                                                d_inv_q_last_mod_q_quotient_);

        }

        void RNSTool::decrypt_modt(RNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const
        {
            // Use exact base convension rather than convert the base through the compose API
            base_q_to_t_conv_->exact_convert_array(phase, destination, pool);
        }

        void RNSTool::decrypt_modt_cuda(uint64_t *phase, uint64_t *destination, size_t count) const
        {
            // Use exact base convension rather than convert the base through the compose API
            base_q_to_t_conv_->exact_convert_array_cuda(phase, destination, count);
        }
    } // namespace util
} // namespace seal
