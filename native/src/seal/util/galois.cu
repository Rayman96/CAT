// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/galois.h"
#include "seal/util/numth.h"
#include "seal/util/uintcore.h"
#include "seal/util/common.cuh"
#include "cuda_runtime.h"
// #include "device_launch_parameters.h"
using namespace std;

namespace seal
{
    namespace util
    {

        __global__ void generate_table_ntt_kernel(
            uint32_t galois_elt, uint32_t coeff_count, int coeff_count_power_, uint64_t *operand, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t temp_table;
            while (idx < coeff_count)
            {
                size_t i = idx + coeff_count;
                uint32_t coeff_count_minus_one = static_cast<uint32_t>(coeff_count) - 1;
                uint32_t reversed = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                temp_table = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power_);
                result[idx] = operand[temp_table];

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void generate_table_ntt_batch_kernel(
            uint32_t galois_elt, uint32_t coeff_count, int coeff_count_power_, size_t modulu_size, uint64_t *operand, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t temp_table;
            while (idx < coeff_count * modulu_size)
            {
                size_t i = (idx % coeff_count) + coeff_count;
                uint32_t coeff_count_minus_one = static_cast<uint32_t>(coeff_count) - 1;
                uint32_t reversed = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                temp_table = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power_);
                result[idx] = operand[temp_table + (idx / coeff_count) * coeff_count];

                idx += blockDim.x * gridDim.x;
            }
        }

        constexpr uint32_t GaloisTool::generator_;

        void GaloisTool::generate_table_ntt(uint32_t galois_elt, Pointer<uint32_t> &result) const
        {
#ifdef SEAL_DEBUG
            if (!(galois_elt & 1) || (galois_elt >= 2 * (uint64_t(1) << coeff_count_power_)))
            {
                throw invalid_argument("Galois element is not valid");
            }
#endif
            ReaderLock reader_lock(permutation_tables_locker_.acquire_read());
            if (result)
            {
                return;
            }
            reader_lock.unlock();

            auto temp(allocate<uint32_t>(coeff_count_, pool_));
            auto temp_ptr = temp.get();

            uint32_t coeff_count_minus_one = safe_cast<uint32_t>(coeff_count_) - 1;
            for (size_t i = coeff_count_; i < coeff_count_ << 1; i++)
            {
                uint32_t reversed = reverse_bits<uint32_t>(safe_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                *temp_ptr++ = reverse_bits<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power_);
            }

            WriterLock writer_lock(permutation_tables_locker_.acquire_write());
            if (result)
            {
                return;
            }
            result.acquire(move(temp));
        }

        uint32_t GaloisTool::get_elt_from_step(int step) const
        {
            uint32_t n = safe_cast<uint32_t>(coeff_count_);
            uint32_t m32 = mul_safe(n, uint32_t(2));
            uint64_t m = static_cast<uint64_t>(m32);

            if (step == 0)
            {
                return static_cast<uint32_t>(m - 1);
            }
            else
            {
                // Extract sign of steps. When steps is positive, the rotation
                // is to the left; when steps is negative, it is to the right.
                bool sign = step < 0;
                uint32_t pos_step = safe_cast<uint32_t>(abs(step));

                if (pos_step >= (n >> 1))
                {
                    throw invalid_argument("step count too large");
                }

                pos_step &= m32 - 1;
                if (sign)
                {
                    step = safe_cast<int>(n >> 1) - safe_cast<int>(pos_step);
                }
                else
                {
                    step = safe_cast<int>(pos_step);
                }

                // Construct Galois element for row rotation
                uint64_t gen = static_cast<uint64_t>(generator_);
                uint64_t galois_elt = 1;
                while (step--)
                {
                    galois_elt *= gen;
                    galois_elt &= m - 1;
                }
                return static_cast<uint32_t>(galois_elt);
            }
        }

        vector<uint32_t> GaloisTool::get_elts_from_steps(const vector<int> &steps) const
        {
            vector<uint32_t> galois_elts;
            transform(steps.begin(), steps.end(), back_inserter(galois_elts), [&](auto s) {
                return this->get_elt_from_step(s);
            });
            return galois_elts;
        }

        vector<uint32_t> GaloisTool::get_elts_all() const noexcept
        {
            uint32_t m = safe_cast<uint32_t>(static_cast<uint64_t>(coeff_count_) << 1);
            vector<uint32_t> galois_elts{};

            // Generate Galois keys for m - 1 (X -> X^{m-1})
            galois_elts.push_back(m - 1);

            // Generate Galois key for power of generator_ mod m (X -> X^{3^k}) and
            // for negative power of generator_ mod m (X -> X^{-3^k})
            uint64_t pos_power = generator_;
            uint64_t neg_power = 0;
            try_invert_uint_mod(generator_, m, neg_power);
            for (int i = 0; i < coeff_count_power_ - 1; i++)
            {
                galois_elts.push_back(static_cast<uint32_t>(pos_power));
                pos_power *= pos_power;
                pos_power &= (m - 1);

                galois_elts.push_back(static_cast<uint32_t>(neg_power));
                neg_power *= neg_power;
                neg_power &= (m - 1);
            }

            return galois_elts;
        }

        void GaloisTool::initialize(int coeff_count_power)
        {
            if ((coeff_count_power < get_power_of_two(SEAL_POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > get_power_of_two(SEAL_POLY_MOD_DEGREE_MAX))
            {
                throw invalid_argument("coeff_count_power out of range");
            }

            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;

            // Capacity for coeff_count_ number of tables
            permutation_tables_ = allocate<Pointer<uint32_t>>(coeff_count_, pool_);
        }

        void GaloisTool::apply_galois(
            ConstCoeffIter operand, uint32_t galois_elt, const Modulus &modulus, CoeffIter result) const
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw invalid_argument("operand");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (operand == result)
            {
                throw invalid_argument("result cannot point to the same value as operand");
            }
            // Verify coprime conditions.
            if (!(galois_elt & 1) || (galois_elt >= 2 * (uint64_t(1) << coeff_count_power_)))
            {
                throw invalid_argument("Galois element is not valid");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
            const uint64_t modulus_value = modulus.value();
            const uint64_t coeff_count_minus_one = coeff_count_ - 1;
            uint64_t index_raw = 0;
            for (uint64_t i = 0; i <= coeff_count_minus_one; i++, ++operand, index_raw += galois_elt)
            {
                uint64_t index = index_raw & coeff_count_minus_one;
                uint64_t result_value = *operand;
                if ((index_raw >> coeff_count_power_) & 1)
                {
                    // Explicit inline
                    // result[index] = negate_uint_mod(result[index], modulus);
                    int64_t non_zero = (result_value != 0);
                    result_value = (modulus_value - result_value) & static_cast<uint64_t>(-non_zero);
                }
                result[index] = result_value;
            }
        }

         void GaloisTool::apply_galois_ntt(ConstCoeffIter operand, uint32_t galois_elt, CoeffIter result) const
        {
            // malloc
            uint64_t *d_result = nullptr;
            uint64_t *d_operand = nullptr;
            allocate_gpu<uint64_t>(&d_result, coeff_count_);
            allocate_gpu<uint64_t>(&d_operand, coeff_count_);

            checkCudaErrors(cudaMemcpy(d_operand, operand.ptr(), coeff_count_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
            generate_table_ntt_kernel<<<(coeff_count_ + 1023) / 1023, 1024>>>(
                galois_elt, coeff_count_, coeff_count_power_, d_operand, d_result);
            checkCudaErrors(cudaMemcpy(result, d_result, coeff_count_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));


            deallocate_gpu<uint64_t>(&d_result, coeff_count_);
            deallocate_gpu<uint64_t>(&d_operand, coeff_count_);
        }

        void GaloisTool::apply_galois_ntt_cuda(uint64_t *d_operand, uint32_t galois_elt, uint64_t *d_result) const
        {

            generate_table_ntt_kernel<<<(coeff_count_ + 1023) / 1023, 1024>>>(
                            galois_elt, coeff_count_, coeff_count_power_, d_operand, d_result);
        }

        void GaloisTool::apply_galois_ntt_batch_cuda(uint64_t *d_operand, uint32_t galois_elt, uint64_t *d_result, size_t modulu_size) const
        {

            generate_table_ntt_batch_kernel<<<(coeff_count_ * modulu_size + 255) / 255, 256>>>(
                            galois_elt, 
                            coeff_count_, 
                            coeff_count_power_,
                            modulu_size, 
                            d_operand, 
                            d_result);
        }


    } // namespace util
} // namespace seal
