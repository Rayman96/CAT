// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/evaluator.h"
#include "seal/keygenerator.h"
#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/galois.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.cuh"
#include "seal/util/rlwe.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        template <typename T, typename S>
        SEAL_NODISCARD inline bool are_same_scale(const T &value1, const S &value2) noexcept
        {
            return util::are_close<double>(value1.scale(), value2.scale());
        }

        SEAL_NODISCARD inline bool is_scale_within_bounds(
            double scale, const SEALContext::ContextData &context_data) noexcept
        {
            int scale_bit_count_bound = 0;
            switch (context_data.parms().scheme())
            {
            case scheme_type::bfv:
            case scheme_type::bgv:
                scale_bit_count_bound = context_data.parms().plain_modulus().bit_count();
                break;
            case scheme_type::ckks:
                scale_bit_count_bound = context_data.total_coeff_modulus_bit_count();
                break;
            default:
                // Unsupported scheme; check will fail
                scale_bit_count_bound = -1;
            };

            return !(scale <= 0 || (static_cast<int>(log2(scale)) >= scale_bit_count_bound));
        }

        /**
        Returns (f, e1, e2) such that
        (1) e1 * factor1 = e2 * factor2 = f mod p;
        (2) gcd(e1, p) = 1 and gcd(e2, p) = 1;
        (3) abs(e1_bal) + abs(e2_bal) is minimal, where e1_bal and e2_bal represent e1 and e2 in (-p/2, p/2].
        */
        SEAL_NODISCARD inline auto balance_correction_factors(
            uint64_t factor1, uint64_t factor2, const Modulus &plain_modulus) -> tuple<uint64_t, uint64_t, uint64_t>
        {
            uint64_t t = plain_modulus.value();
            uint64_t half_t = t / 2;

            auto sum_abs = [&](uint64_t x, uint64_t y) {
                int64_t x_bal = static_cast<int64_t>(x > half_t ? x - t : x);
                int64_t y_bal = static_cast<int64_t>(y > half_t ? y - t : y);
                return abs(x_bal) + abs(y_bal);
            };

            // ratio = f2 / f1 mod p
            uint64_t ratio = 1;
            if (!try_invert_uint_mod(factor1, plain_modulus, ratio))
            {
                throw logic_error("invalid correction factor1");
            }
            ratio = multiply_uint_mod(ratio, factor2, plain_modulus);
            uint64_t e1 = ratio;
            uint64_t e2 = 1;
            int64_t sum = sum_abs(e1, e2);

            // Extended Euclidean
            int64_t prev_a = static_cast<int64_t>(plain_modulus.value());
            int64_t prev_b = static_cast<int64_t>(0);
            int64_t a = static_cast<int64_t>(ratio);
            int64_t b = 1;

            while (a != 0)
            {
                int64_t q = prev_a / a;
                int64_t temp = prev_a % a;
                prev_a = a;
                a = temp;

                temp = sub_safe(prev_b, mul_safe(b, q));
                prev_b = b;
                b = temp;

                uint64_t a_mod = barrett_reduce_64(static_cast<uint64_t>(abs(a)), plain_modulus);
                if (a < 0)
                {
                    a_mod = negate_uint_mod(a_mod, plain_modulus);
                }
                uint64_t b_mod = barrett_reduce_64(static_cast<uint64_t>(abs(b)), plain_modulus);
                if (b < 0)
                {
                    b_mod = negate_uint_mod(b_mod, plain_modulus);
                }
                if (a_mod != 0 && gcd(a_mod, t) == 1) // which also implies gcd(b_mod, t) == 1
                {
                    int64_t new_sum = sum_abs(a_mod, b_mod);
                    if (new_sum < sum)
                    {
                        sum = new_sum;
                        e1 = a_mod;
                        e2 = b_mod;
                    }
                }
            }
            return make_tuple(multiply_uint_mod(e1, factor1, plain_modulus), e1, e2);
        }

        __global__ void computeNonZeroCoefficientsKernel(
            uint64_t *poly, uint64_t *modulus_value, size_t encrypted_size, size_t coeff_count,
            size_t coeff_modulus_size)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            while (idx < encrypted_size * coeff_count * coeff_modulus_size)
            {
                uint64_t coeff = poly[idx];
                std::int64_t non_zero = (coeff != 0);
                poly[idx] = (modulus_value[(idx / coeff_count) % coeff_modulus_size] - coeff) &
                            static_cast<std::uint64_t>(-non_zero);
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void ckks_bgv_multiply_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, size_t coeff_modulus_size, uint64_t *modulus,
            uint64_t *modulus_ratio_0, uint64_t *modulus_ratio_1, uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size)
            {
                const int modulu_index = (idx / coeff_count) % coeff_modulus_size;
                const int index_1 = idx + coeff_count * coeff_modulus_size;
                const int index_2 = idx + 2 * coeff_count * coeff_modulus_size;
                const uint64_t modulus_value = modulus[modulu_index];
                const uint64_t const_ratio_0 = modulus_ratio_0[modulu_index];
                const uint64_t const_ratio_1 = modulus_ratio_1[modulu_index];
                unsigned long long z[2], tmp1, tmp2[2], tmp3, tmpx0y1, tmpx1y0, carry;

                // Compute x1^y1----------------------------------------
                multiply_uint64_kernel2(operand1[index_1], operand2[index_1], z);
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
                result[index_2] = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

                // Compute x0*y1 ----------------------------------------

                multiply_uint64_kernel2(operand1[idx], operand2[index_1], z);
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
                tmpx0y1 = (z[0] - tmp1 * modulus_value) >= modulus_value ? (z[0] - tmp1 * modulus_value) - modulus_value
                                                                         : (z[0] - tmp1 * modulus_value);

                // Compute x1*y0----------------------------------------

                multiply_uint64_kernel2(operand1[index_1], operand2[idx], z);
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
                tmpx1y0 = (z[0] - tmp1 * modulus_value) >= modulus_value ? (z[0] - tmp1 * modulus_value) - modulus_value
                                                                         : (z[0] - tmp1 * modulus_value);
                result[index_1] =
                    tmpx0y1 + tmpx1y0 >= modulus_value ? tmpx0y1 + tmpx1y0 - modulus_value : tmpx0y1 + tmpx1y0;

                // Compute x0*y0----------------------------------------
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

        __global__ void apply_galois_helper(
            uint32_t galois_elt, uint32_t coeff_count, int coeff_count_power_, int coeff_modulus_size,
            uint64_t *operand, uint64_t *result, uint64_t *temp)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t temp_table;
            while (idx < coeff_count * coeff_modulus_size)
            {
                // 计算encrypte_0
                size_t i = idx % coeff_count + coeff_count;
                uint32_t coeff_count_minus_one = static_cast<uint32_t>(coeff_count) - 1;
                uint32_t reversed = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                temp_table = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power_);

                temp[idx] = operand[temp_table + (idx / coeff_count) * coeff_count];
                __syncthreads();
                result[idx] = temp[idx];
                // 用encrypte_1计算temp，并把1赋值为0
                temp[idx] = operand[temp_table + coeff_modulus_size * coeff_count + (idx / coeff_count) * coeff_count];
                __syncthreads();
                // Wipe encrypted.data(1)
                operand[coeff_modulus_size * coeff_count + idx] = 0;

                idx += blockDim.x * gridDim.x;
            }
        }
        
        __global__ void apply_galois_helper_single(
            uint32_t galois_elt, uint32_t coeff_count, int coeff_count_power_, int coeff_modulus_size,
            uint64_t *operand, uint64_t *temp)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t temp_table;
            while (idx < coeff_count * coeff_modulus_size)
            {
                // 计算encrypte_0
                size_t i = idx % coeff_count + coeff_count;
                uint32_t coeff_count_minus_one = static_cast<uint32_t>(coeff_count) - 1;
                uint32_t reversed = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(i), coeff_count_power_ + 1);
                uint64_t index_raw = (static_cast<uint64_t>(galois_elt) * static_cast<uint64_t>(reversed)) >> 1;
                index_raw &= static_cast<uint64_t>(coeff_count_minus_one);
                temp_table = reverse_bits_kernel<uint32_t>(static_cast<uint32_t>(index_raw), coeff_count_power_);

                temp[idx] = operand[temp_table + (idx / coeff_count) * coeff_count];

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void eltwiseAddModSizeKernel(
            uint64_t *result, const uint64_t *operand1, const uint64_t *operand2, const uint64_t *modulus,
            const std::size_t coeff_count, const std::size_t coeff_modulus_size, const std::size_t encrypted_size)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < coeff_count * coeff_modulus_size * encrypted_size)
            {
                const size_t modulus_index = (index / coeff_count) % coeff_modulus_size;
                result[index] = operand1[index] + operand2[index] >= modulus[modulus_index]
                                    ? operand1[index] + operand2[index] - modulus[modulus_index]
                                    : operand1[index] + operand2[index];
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void eltwiseSubModSizeKernel(
            uint64_t *result, const uint64_t *operand1, const uint64_t *operand2, const uint64_t *modulus,
            const std::size_t coeff_count, const std::size_t coeff_modulus_size, const std::size_t encrypted_size)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < coeff_count * coeff_modulus_size * encrypted_size)
            {
                const size_t modulus_value = modulus[(index / coeff_count) % coeff_modulus_size];

                unsigned long long temp_result;
                std::int64_t borrow = sub_uint64_kernel(operand1[index], operand2[index], &temp_result);
                result[index] = temp_result + (modulus_value & static_cast<std::uint64_t>(-borrow));

                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void negatePolyKernel(
            uint64_t *operand1, const uint64_t *modulus, const std::size_t coeff_count,
            const std::size_t coeff_modulus_size, const std::size_t encrypted_size, uint64_t *result)
        {
            std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            while (index < coeff_count * coeff_modulus_size * encrypted_size)
            {
                const size_t modulus_value = modulus[(index / coeff_count) % coeff_modulus_size];

                uint64_t coeff = operand1[index];
                int64_t non_zero = (coeff != 0);
                result[index] = (modulus_value - coeff) & static_cast<std::uint64_t>(-non_zero);

                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void barrett_reduce_64_helper(
            uint64_t *input, uint64_t modulus_value, uint64_t ratio, uint64_t qk_half, uint64_t *result, size_t size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < size)
            {
                result[idx] = barrett_reduce_64_kernel(input[idx] + qk_half, modulus_value, ratio);
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper1(
            uint64_t *input, size_t coeff_count, uint64_t modulus_value, uint64_t modulus_ratio, uint64_t compare_value,
            uint64_t *result)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count)
            {
                if (compare_value > modulus_ratio)
                {
                    result[index] = barrett_reduce_64_kernel(input[index], modulus_value, modulus_ratio);
                }
                else
                {
                    result[index] = input[index];
                }
                uint64_t fix =
                    modulus_value - barrett_reduce_64_kernel(compare_value >> 1, modulus_value, modulus_ratio);
                result[index] += fix;
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper1_batch(
            uint64_t *input, size_t coeff_count, int modulu_size, uint64_t *modulus_value, uint64_t *modulus_ratio, uint64_t compare_value,
            uint64_t *result)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count * modulu_size)
            {
                uint64_t input_index = index % coeff_count;
                int modulu_index = index / coeff_count;
                uint64_t modulus_value_ = modulus_value[modulu_index];
                uint64_t modulus_ratio_ = modulus_ratio[modulu_index];

                if (compare_value > modulus_ratio_)
                {
                    result[index] = barrett_reduce_64_kernel(input[input_index], modulus_value_, modulus_ratio_);
                }
                else
                {
                    result[index] = input[index];
                }
                uint64_t fix =
                    modulus_value_ - barrett_reduce_64_kernel(compare_value >> 1, modulus_value_, modulus_ratio_);
                result[index] += fix;
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper2(
            uint64_t *operand1, uint64_t *temp_result, size_t coeff_count, uint64_t operand, uint64_t quotient,
            uint64_t modulus_value, uint64_t *result)
        {
            uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                uint64_t qi = modulus_value;

#if SEAL_USER_MOD_BIT_COUNT_MAX > 60
                // 目前限制了，不会经过这里
                uint64_t qi_lazy = qi << 1;
                operand1[idx] = operand1[idx] >= qi_lazy ? qi_lazy : 0;
#else
                // Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
                uint64_t qi_lazy = qi << 2;
#endif

                temp_result[idx] += qi_lazy - operand1[idx];
                // multiply_poly_scalar_coeffmod
                unsigned long long tmp1, tmp2;
                multiply_uint64_hw64_kernel(temp_result[idx], quotient, &tmp1);
                tmp2 = operand * temp_result[idx] - tmp1 * modulus_value;
                temp_result[idx] = tmp2 >= modulus_value ? tmp2 - modulus_value : tmp2;
                // add_poly_coeffmod_kernel
                uint64_t sum = temp_result[idx] + result[idx];
                result[idx] = sum >= modulus_value ? sum - modulus_value : sum;
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper2_batch(
            uint64_t *operand1, uint64_t *temp_result, size_t coeff_count, uint64_t operand, uint64_t quotient,
            uint64_t modulus_value, uint64_t *result)
        {
            uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                uint64_t qi = modulus_value;

#if SEAL_USER_MOD_BIT_COUNT_MAX > 60
                // 目前限制了，不会经过这里
                uint64_t qi_lazy = qi << 1;
                operand1[idx] = operand1[idx] >= qi_lazy ? qi_lazy : 0;
#else
                // Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
                uint64_t qi_lazy = qi << 2;
#endif

                temp_result[idx] += qi_lazy - operand1[idx];
                // multiply_poly_scalar_coeffmod
                unsigned long long tmp1, tmp2;
                multiply_uint64_hw64_kernel(temp_result[idx], quotient, &tmp1);
                tmp2 = operand * temp_result[idx] - tmp1 * modulus_value;
                temp_result[idx] = tmp2 >= modulus_value ? tmp2 - modulus_value : tmp2;
                // add_poly_coeffmod_kernel
                uint64_t sum = temp_result[idx] + result[idx];
                result[idx] = sum >= modulus_value ? sum - modulus_value : sum;
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper3(
            uint64_t *last_input, uint64_t coeff_count, uint64_t base_q_value, uint64_t base_q_ratio,
            uint64_t *temp_result, bool isLess)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count)
            {
                if (isLess)
                {
                    temp_result[index] = last_input[index];
                }
                else
                {
                    temp_result[index] = barrett_reduce_64_kernel(last_input[index], base_q_value, base_q_ratio);
                }
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void switch_key_helper3_batch(
            uint64_t *last_input, uint64_t coeff_count, uint64_t modulu_size, uint64_t base_q_value, uint64_t base_q_ratio,
            uint64_t *temp_result, uint64_t *modulu_value, uint64_t key_index_modulu)
        {
            uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count * modulu_size)
            {
                uint64_t modulu = modulu_value[index /coeff_count];
                if (modulu <= key_index_modulu)
                {
                    temp_result[index] = last_input[index];
                }
                else
                {
                    temp_result[index] = barrett_reduce_64_kernel(last_input[index], base_q_value, base_q_ratio);
                }
                index += blockDim.x * gridDim.x;
            }
        }

        __global__ void lazy_reduction_counter_kernel(
            uint64_t *value, uint64_t *result, uint64_t coeff_count, uint64_t key_modulus_value, uint64_t ratio_0,
            uint64_t ratio_1, int key_component_count, int rns_modulus_size, bool is_equal)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * key_component_count)
            {
                size_t result_index = (idx / coeff_count) * (rns_modulus_size * coeff_count) + (idx % coeff_count);
                size_t value_index = (idx / coeff_count) * 2 * coeff_count + (idx % coeff_count) * 2;
                if (is_equal)
                {
                    result[result_index] = static_cast<uint64_t>(value[value_index]);
                }
                else
                {
                    uint64_t ratio[2] = { ratio_0, ratio_1 };
                    result[result_index] = barrett_reduce_128_kernel3(value + value_index, key_modulus_value, ratio);
                }
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void lazy_reduction_counter_kernel2(
            uint64_t *d_temp_operand, uint64_t *d_key_vector, uint64_t *d_temp_poly_lazy, uint64_t coeff_count,
            uint64_t key_modulus_value, uint64_t *ratio, bool lazy_reduction_counter)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count)
            {
                unsigned long long qword[2]{ 0, 0 };
                multiply_uint64_kernel(d_temp_operand[idx], d_key_vector[idx], qword);
                // Accumulate product of t_operand and t_key_acc to t_poly_lazy and reduce
                add_uint128_kernel(qword, d_temp_poly_lazy + 2 * idx, qword);

                if (!lazy_reduction_counter)
                {
                    d_temp_poly_lazy[2 * idx] = barrett_reduce_128_kernel3(qword, key_modulus_value, ratio);
                    d_temp_poly_lazy[2 * idx + 1] = 0;
                }
                else
                {
                    d_temp_poly_lazy[2 * idx] = qword[0];
                    d_temp_poly_lazy[2 * idx + 1] = qword[1];
                }
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void lazy_reduction_counter_kernel3(
            uint64_t *d_temp_operand, uint64_t *d_key_vector, uint64_t *d_temp_poly_lazy, uint64_t coeff_count,
            uint64_t key_modulus_value, uint64_t ratio_0, uint64_t ratio_1, bool lazy_reduction_counter,
            int key_component_count, size_t key_modulus_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * key_component_count)
            {
                size_t index_1 = idx % coeff_count;
                size_t index_result = (idx / coeff_count) * (2 * coeff_count) + 2 * (idx % coeff_count);
                size_t index_key = (idx / coeff_count) * (coeff_count * key_modulus_size) + idx % coeff_count;

                unsigned long long qword[2]{ 0, 0 };
                multiply_uint64_kernel(d_temp_operand[index_1], d_key_vector[index_key], qword);
                // Accumulate product of t_operand and t_key_acc to t_poly_lazy and reduce
                add_uint128_kernel(qword, d_temp_poly_lazy + index_result, qword);

                if (!lazy_reduction_counter)
                {
                    uint64_t ratio[2] = { ratio_0, ratio_1 };
                    d_temp_poly_lazy[index_result] = barrett_reduce_128_kernel3(qword, key_modulus_value, ratio);
                    d_temp_poly_lazy[index_result + 1] = 0;
                }
                else
                {
                    d_temp_poly_lazy[index_result] = qword[0];
                    d_temp_poly_lazy[index_result + 1] = qword[1];
                }
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void lazy_reduction_counter_kernel4(
            uint64_t *d_temp_operand, uint64_t** d_key_vector, uint64_t *d_temp_poly_lazy, uint64_t coeff_count,
            uint64_t decomp_modulu_size, uint64_t key_modulus_value, uint64_t ratio_0, uint64_t ratio_1,
            size_t lazy_reduction_counter, size_t lazy_reduction_summand_bound, int key_component_count, size_t key_modulus_size,
            uint64_t key_index)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * key_component_count)
            {
                size_t index_1 = idx % coeff_count;
                size_t index_result = (idx / coeff_count) * (2 * coeff_count) + 2 * (idx % coeff_count);
                size_t index_key = (idx / coeff_count) * (coeff_count * key_modulus_size) + idx % coeff_count + key_index * coeff_count;

                uint64_t ratio[2] = { ratio_0, ratio_1 };
                unsigned long long qword[2]{ 0, 0 };

                for(int i = 0; i < decomp_modulu_size; i++){
                    multiply_uint64_kernel(d_temp_operand[i * coeff_count + index_1], d_key_vector[i][index_key], qword);
                    // Accumulate product of t_operand and t_key_acc to t_poly_lazy and reduce
                    add_uint128_kernel(qword, d_temp_poly_lazy + index_result, qword);

                    if (!lazy_reduction_counter)
                    {
                        d_temp_poly_lazy[index_result] = barrett_reduce_128_kernel3(qword, key_modulus_value, ratio);
                        d_temp_poly_lazy[index_result + 1] = 0;
                    }
                    else
                    {
                        d_temp_poly_lazy[index_result] = qword[0];
                        d_temp_poly_lazy[index_result + 1] = qword[1];
                    }

                    if (!--lazy_reduction_counter)
                    {
                        lazy_reduction_counter = lazy_reduction_summand_bound;
                    }
                }
                   
                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void diff_modulus_size_copy_kernel(
            uint64_t *origin, uint64_t *destination, size_t encrypted_size, size_t coeff_count,
            size_t old_coeff_modulus_size, size_t new_coeff_modulus_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < encrypted_size * coeff_count * new_coeff_modulus_size)
            {
                size_t encrypted_shift =
                    (idx / (new_coeff_modulus_size * coeff_count)) * (old_coeff_modulus_size * coeff_count);
                size_t modulu_shift =
                    idx - (idx / (new_coeff_modulus_size * coeff_count)) * (new_coeff_modulus_size * coeff_count);
                size_t old_idx = encrypted_shift + modulu_shift;

                destination[idx] = origin[old_idx];
                idx += blockDim.x * gridDim.x;
            }
        }

        __device__ void behz_ciphertext_product_kernel_helper(
            uint64_t *operand1, uint64_t *operand2, uint64_t *operand3, uint64_t modulus, uint64_t modulus_ratio_0,
            uint64_t modulus_ratio_1, uint64_t *result)
        {
            const uint64_t modulus_value = modulus;
            const uint64_t const_ratio_0 = modulus_ratio_0;
            const uint64_t const_ratio_1 = modulus_ratio_1;

            unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
            multiply_uint64_kernel2(*operand1, *operand2, z);
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
            std::uint64_t temp_result = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

            std::uint64_t sum = temp_result + *operand3;
            *result = sum >= modulus_value ? sum - modulus_value : sum;
        }

        __device__ void  multiply_poly_scalar_coeffmod_helper_kernel(uint64_t poly, uint64_t scalar, uint64_t modulus_value, uint64_t ratio1, uint64_t *result) {
            
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

        __global__ void bfv_multiply_helper1(
            uint64_t *d_encrypted1_q, uint64_t *d_encrypted2_q, uint64_t *d_encrypted1_Bsk, uint64_t *d_encrypted2_Bsk,
            uint64_t *d_temp_dest_q, uint64_t *d_temp_dest_Bsk, size_t dest_size, size_t encrypted1_size,
            size_t encrypted2_size, size_t coeff_count, size_t base_q_size, size_t base_Bsk_size,
            uint64_t *q_modulus_value, uint64_t *q_modulus_ratio0, uint64_t *q_modulus_ratio1,
            uint64_t *Bsk_modulus_value, uint64_t *Bsk_modulus_ratio0, uint64_t *Bsk_modulus_ratio1)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < dest_size * coeff_count * base_Bsk_size)
            {
                const size_t dest_size_idx = idx / (coeff_count * base_Bsk_size);
                const size_t dest_modulu_q_idx = (idx - dest_size_idx * coeff_count * base_q_size) / coeff_count;
                const size_t dest_modulu_Bsk_idx = (idx - dest_size_idx * coeff_count * base_Bsk_size) / coeff_count;
                const size_t coeff_idx = idx % coeff_count;

                const size_t curr_encrypted1_last =
                    dest_size_idx > encrypted1_size - 1 ? encrypted1_size - 1 : dest_size_idx;
                const size_t curr_encrypted2_first =
                    dest_size_idx > encrypted2_size - 1 ? encrypted2_size - 1 : dest_size_idx;

                const size_t curr_encrypted1_first = dest_size_idx - curr_encrypted2_first;
                const size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                // 几个数据的起点坐标,dest是结果, 2需要反转一下
                const size_t encrypted1_q_index = curr_encrypted1_first * coeff_count * base_q_size;
                const size_t encrypted1_Bsk_index = curr_encrypted1_first * coeff_count * base_Bsk_size;

                const size_t encrypted2_q_index = curr_encrypted2_first * coeff_count * base_q_size;
                const size_t encrypted2_Bsk_index = curr_encrypted2_first * coeff_count * base_Bsk_size;

                const size_t temp_dest_q_index = dest_size_idx * coeff_count * base_q_size;
                const size_t temp_dest_Bsk_index = dest_size_idx * coeff_count * base_Bsk_size;

                for (int i = 0; i < steps; i++){
                
                    if (idx < dest_size * coeff_count * base_q_size) {
                        behz_ciphertext_product_kernel_helper(
                            d_encrypted1_q + encrypted1_q_index + i * base_q_size * coeff_count +
                                dest_modulu_q_idx * coeff_count + coeff_idx,
                            d_encrypted2_q + encrypted2_q_index - i * base_q_size * coeff_count +
                                dest_modulu_q_idx * coeff_count + coeff_idx,
                            d_temp_dest_q + temp_dest_q_index + dest_modulu_q_idx * coeff_count + coeff_idx,
                            q_modulus_value[dest_modulu_q_idx], q_modulus_ratio0[dest_modulu_q_idx],
                            q_modulus_ratio1[dest_modulu_q_idx],
                            d_temp_dest_q + temp_dest_q_index + dest_modulu_q_idx * coeff_count + coeff_idx);
                    }
 
                    behz_ciphertext_product_kernel_helper(
                        d_encrypted1_Bsk + encrypted1_Bsk_index + i * base_Bsk_size * coeff_count +
                            dest_modulu_Bsk_idx * coeff_count + coeff_idx,
                        d_encrypted2_Bsk + encrypted2_Bsk_index - i * base_Bsk_size * coeff_count +
                            dest_modulu_Bsk_idx * coeff_count + coeff_idx,
                        d_temp_dest_Bsk + temp_dest_Bsk_index + dest_modulu_Bsk_idx * coeff_count + coeff_idx,
                        Bsk_modulus_value[dest_modulu_Bsk_idx], Bsk_modulus_ratio0[dest_modulu_Bsk_idx],
                        Bsk_modulus_ratio1[dest_modulu_Bsk_idx],
                        d_temp_dest_Bsk + temp_dest_Bsk_index + dest_modulu_Bsk_idx * coeff_count + coeff_idx);
                }
                idx += blockDim.x * gridDim.x;
            }
        }


        __global__ void bfv_multiply_helper2(uint64_t *d_temp_dest_q, uint64_t *d_temp_dest_Bsk, uint64_t *d_temp_q_Bsk,
            uint64_t scalar,
            size_t coeff_count, size_t dest_size, size_t base_q_size, size_t base_Bsk_size, 
            uint64_t *q_modulus_value, uint64_t *q_modulus_ratio1,
            uint64_t *Bsk_modulus_value, uint64_t *Bsk_modulus_ratio1) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * dest_size * base_Bsk_size){
                size_t dest_size_idx = idx / (coeff_count * base_Bsk_size);
                size_t dest_size_q_idx = idx / (coeff_count * base_q_size);


                size_t dest_Bsk_count = coeff_count * base_Bsk_size;
                size_t dest_q_count = coeff_count * base_q_size;


                size_t dest_modulu_Bsk_idx = (idx - dest_size_idx * dest_Bsk_count) / coeff_count;
                size_t dest_modulu_q_idx = (idx - dest_size_q_idx * dest_q_count) / coeff_count;
                size_t coeff_idx = idx % coeff_count;
                
                

                multiply_poly_scalar_coeffmod_helper_kernel(
                                    *(d_temp_dest_Bsk + dest_size_idx * coeff_count * base_Bsk_size +  dest_modulu_Bsk_idx * coeff_count + coeff_idx ), 
                                    scalar,
                                    Bsk_modulus_value[dest_modulu_Bsk_idx],
                                    Bsk_modulus_ratio1[dest_modulu_Bsk_idx],
                                    d_temp_q_Bsk + dest_size_idx * coeff_count * (base_Bsk_size + base_q_size) + base_q_size * coeff_count + dest_modulu_Bsk_idx * coeff_count + coeff_idx);
                if (idx < coeff_count * dest_size * base_q_size) {
                    multiply_poly_scalar_coeffmod_helper_kernel(
                                        *(d_temp_dest_q + dest_size_q_idx * dest_q_count +  dest_modulu_q_idx * coeff_count + coeff_idx), 
                                        scalar,
                                        q_modulus_value[dest_modulu_q_idx],
                                        q_modulus_ratio1[dest_modulu_q_idx],
                                        d_temp_q_Bsk + dest_size_q_idx * coeff_count * (base_Bsk_size + base_q_size) +  dest_modulu_q_idx * coeff_count + coeff_idx);
                }
                // d_temp_q_Bsk -> floor -> d_temp_Bsk

                idx += blockDim.x * gridDim.x;

            }

        }


        __global__ void transform_helper(uint64_t *plain, uint64_t *temp, 
                                        size_t coeff_count, size_t coeff_modulus_size,
                                        uint64_t *plain_upper_half_increment,
                                        uint64_t plain_upper_half_threshold) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count){
                if (plain[idx] < plain_upper_half_threshold) {
                    temp[idx] = plain[idx];
                } else {
                    unsigned char carry = add_uint64_kernel(plain_upper_half_increment[idx], plain[idx], temp + idx);
                    idx += coeff_count;
                    for (; --coeff_modulus_size; idx+=coeff_count) {
                        unsigned long long temp_result;
                        carry = add_uint64_carry_kernel(plain_upper_half_increment[idx], std::uint64_t(0), carry, &temp_result);
                        temp[idx] = temp_result;
                    }
                }
                idx += blockDim.x * gridDim.x;
            }
        }


        __global__ void negacyclic_multiply_poly_mono_coeffmod_kernel(uint64_t *input, uint64_t plain,
                                                                     size_t size, size_t coeff_count, size_t coeff_modulus_size,
                                                                     size_t mono_exponent, uint64_t *modulus_value, uint64_t *ratio1){
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < size * coeff_modulus_size * coeff_count){
                size_t modulu_idx = (idx / coeff_count) % coeff_modulus_size;
                uint64_t temp_result;
                multiply_poly_scalar_coeffmod_helper_kernel(input[idx], plain, modulus_value[modulu_idx], ratio1[modulu_idx], &temp_result);

                if (mono_exponent == 0) {
                    input[idx] = temp_result;
                    return ;
                } 

                size_t coeff_idx = idx % coeff_count;
                size_t index_shift = (idx / coeff_count) * coeff_count;
                uint64_t index_raw = mono_exponent + coeff_idx;
                uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
                uint64_t index = index_raw & coeff_count_mod_mask;
                if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !input[idx]) {
                    input[index_shift + index] = temp_result;
                } else {
                    input[index_shift + index] = modulus_value[modulu_idx] - temp_result;
                }

                idx += blockDim.x * gridDim.x;
            }

        }

    // 输入k_last，输出delta
        __global__ void bgv_switch_key_helper1(uint64_t *input, uint64_t *output, size_t coeff_count, size_t modulu_size,
                                            uint64_t modulus_value, uint64_t ratio_1, uint64_t qk_inv_qp, uint64_t qk,
                                            uint64_t *key_modulu_value, uint64_t *key_ratio_1){
            uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * modulu_size) {
                size_t coeff_idx = idx % coeff_count;
                size_t modulu_idx = idx / coeff_count;
                uint64_t k = barrett_reduce_64_kernel(input[coeff_idx], modulus_value, ratio_1);

                uint64_t temp_neg = k;
                const uint64_t non_zero = (temp_neg != 0);
                k = (modulus_value - temp_neg) & static_cast<std::uint64_t>(-non_zero);

                if (qk_inv_qp != 1){
                    multiply_poly_scalar_coeffmod_helper_kernel(k, qk_inv_qp, modulus_value, ratio_1, &k);
                }

                output[idx] = barrett_reduce_64_kernel(k, key_modulu_value[modulu_idx], key_ratio_1[modulu_idx]);
                multiply_poly_scalar_coeffmod_helper_kernel(output[idx], qk, key_modulu_value[modulu_idx], key_ratio_1[modulu_idx], output+idx);

                uint64_t temp_c_mod_qi = barrett_reduce_64_kernel(input[coeff_idx], key_modulu_value[modulu_idx], key_ratio_1[modulu_idx]);
                
                output[idx] = add_uint_mod_kernel(output[idx], temp_c_mod_qi, key_modulu_value[modulu_idx]);

                idx += blockDim.x * gridDim.x;
            }    
        }
        

        __global__ void bgv_switch_key_helper2(uint64_t *input, uint64_t *d_t_poly_prod, uint64_t *output, 
                                                size_t coeff_count, size_t modulu_size, size_t rns_modulus_size,
                                                uint64_t *modulu_value, uint64_t *modulu_ratio_0, uint64_t *modulu_ratio_1,
                                                uint64_t *modswitch_factors_operand, uint64_t *modswitch_factors_quotient){
            uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * modulu_size) {
                size_t coeff_idx = idx % coeff_count;
                size_t modulu_idx = idx / coeff_count;

                d_t_poly_prod[idx] = sub_uint_mod_kernel(d_t_poly_prod[idx], input[idx], modulu_value[modulu_idx]);

                d_t_poly_prod[idx] = multiply_uint_mod_kernel(d_t_poly_prod[idx], 
                                modswitch_factors_quotient[modulu_idx], modswitch_factors_operand[modulu_idx],modulu_value[modulu_idx]);

                uint64_t copy_output = output[idx];
                uint64_t sum = d_t_poly_prod[idx] + output[idx];
                output[idx] = sum >= modulu_value[modulu_idx] ? sum - modulu_value[modulu_idx] : sum;

                idx += blockDim.x * gridDim.x;
            }
        }


        __global__ void mod_switch_helper(uint64_t *input, uint64_t *output, size_t encrypted_size, 
                                            size_t coeff_count, size_t coeff_modulus_size, size_t next_coeff_modulus_size){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < encrypted_size * coeff_count * next_coeff_modulus_size){
                size_t n = index / (coeff_count * next_coeff_modulus_size);
                size_t l = (index - n * coeff_count * next_coeff_modulus_size) / coeff_count;
                size_t p = index % coeff_count;

                output[index] = input[n * coeff_count * coeff_modulus_size + l * coeff_count + p];
                index += blockDim.x * gridDim.x;
            }
        }


        __global__ void transform_helper2(uint64_t *input, uint64_t *output, size_t plain_coeff_count, size_t coeff_modulus_size,
                                        size_t coeff_count,
                                        uint64_t *plain_upper_half_increment, uint64_t plain_upper_half_threshold){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count * coeff_modulus_size){
                
                size_t shift_modulu = (coeff_modulus_size- 1 - (index / coeff_count)) % coeff_modulus_size;

                size_t output_index =  shift_modulu * coeff_count + index % coeff_count;

                if(index % coeff_count >= plain_coeff_count){
                    output[output_index] = 0;
                    return;
                }
                uint64_t plain_value = input[0];
                if (plain_value >= plain_upper_half_threshold){
                        output[output_index] = plain_value + plain_upper_half_increment[shift_modulu];
                }
                else{
                    output[output_index] = plain_value;
                }
                index += blockDim.x * gridDim.x;
            }

        }

        __global__ void multiply_plain_normal_helper(uint64_t *encrypt, uint64_t *plain, uint64_t *output, size_t coeff_count, 
                                                    int coeff_modulu_size, uint64_t *plain_upper_half_increment, uint64_t plain_upper_half_threshold){
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            while (index < coeff_count * coeff_modulu_size){
                size_t plain_index = index % coeff_count;
                uint64_t plain_value = plain[plain_index];
                if (plain_value >= plain_upper_half_threshold){
                    output[index] = encrypt[index] + plain_upper_half_increment[index / coeff_count];
                }
                else{
                    output[index] = encrypt[index];
                }   
                
                
                index += blockDim.x * gridDim.x;

            }
        
        }


        __global__ void multiply_plain_normal_helper2(uint64_t *plain, uint64_t *output, size_t coeff_count, 
            size_t coeff_modulus_size, uint64_t plain_upper_half_threshold, uint64_t *plain_upper_half_increment) {
                size_t index = blockIdx.x * blockDim.x + threadIdx.x;
                while (index < coeff_count * coeff_modulus_size){
                    if (plain[index % coeff_count] >= plain_upper_half_threshold){
                        output[index] = plain[index % coeff_count] + plain_upper_half_increment[index / coeff_count];
                    } else {
                        output[index] = plain[index % coeff_count];
                    }
                    
                    index += blockDim.x * gridDim.x;

                }
            }


    } // namespace

    Evaluator::Evaluator(const SEALContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
    }

    void Evaluator::ensure_size(uint64_t **input, size_t current_size, size_t &size) const
    {
        if (current_size > size)
        {
            checkCudaErrors(cudaMalloc((void **)input, current_size * sizeof(uint64_t)));
            size = current_size;
        }
    }

    void Evaluator::negate_inplace(Ciphertext &encrypted) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        uint64_t *d_encrypted = encrypted.d_data();
        uint64_t *d_modulus = parms.d_coeff_modulus_value();

        constexpr int threadsPerBlock = 256;
        int blocksPerGrid = (encrypted_size * coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;

        computeNonZeroCoefficientsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_encrypted, d_modulus, encrypted_size, coeff_count, coeff_modulus_size);

        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::add_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        // Size check
        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        if (encrypted1.correction_factor() != encrypted2.correction_factor())
        {
            // printf("add goes here\n");

            // Balance correction factors and multiply by scalars before addition in BGV
            auto factors = balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(), plain_modulus);
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted1.data(), coeff_count, coeff_modulus_size), encrypted1.size(), get<1>(factors),
                coeff_modulus, PolyIter(encrypted1.data(), coeff_count, coeff_modulus_size));

            Ciphertext encrypted2_copy = encrypted2;
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted2.data(), coeff_count, coeff_modulus_size), encrypted2.size(), get<2>(factors),
                coeff_modulus, PolyIter(encrypted2_copy.data(), coeff_count, coeff_modulus_size));

            // Set new correction factor
            encrypted1.correction_factor() = get<0>(factors);
            encrypted2_copy.correction_factor() = get<0>(factors);

            add_inplace(encrypted1, encrypted2_copy);
        }
        else
        {
            // Prepare destination
            // encrypted1.resize(context_, context_data.parms_id(), max_count);
            encrypted1.resize_pure_gpu(context_, context_data.parms_id(), max_count);

            uint64_t *d_encrypted1 = encrypted1.d_data();
            uint64_t *d_encrypted2 = encrypted2.d_data();
            uint64_t *d_modulus = parms.d_coeff_modulus_value();
            // Add ciphertexts
            const size_t threadsPerBlock = 256;
            const size_t blocksPerGrid =
                (min_count * coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;
            eltwiseAddModSizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_encrypted1, d_encrypted1, d_encrypted2, d_modulus, coeff_count, coeff_modulus_size, min_count);

            // Copy the remainding polys of the array with larger count into encrypted1
            if (encrypted1_size < encrypted2_size)
            {
                checkCudaErrors(cudaMemcpy(
                    d_encrypted1 + encrypted1_size * coeff_count * coeff_modulus_size,
                    d_encrypted2 + encrypted1_size * coeff_count * coeff_modulus_size,
                    (encrypted2_size - encrypted1_size) * coeff_count * coeff_modulus_size * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice));
            }
        }

        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted1.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::add_many(const vector<Ciphertext> &encrypteds, Ciphertext &destination) const
    {
        if (encrypteds.empty())
        {
            throw invalid_argument("encrypteds cannot be empty");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }

        destination = encrypteds[0];
        for (size_t i = 1; i < encrypteds.size(); i++)
        {
            add_inplace(destination, encrypteds[i]);
        }
    }

    void Evaluator::sub_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }
        if (encrypted1.is_ntt_form() != encrypted2.is_ntt_form())
        {
            throw invalid_argument("NTT form mismatch");
        }
        if (!are_same_scale(encrypted1, encrypted2))
        {
            throw invalid_argument("scale mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        size_t max_count = max(encrypted1_size, encrypted2_size);
        size_t min_count = min(encrypted1_size, encrypted2_size);

        // Size check
        if (!product_fits_in(max_count, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        if (encrypted1.correction_factor() != encrypted2.correction_factor())
        {
            // Balance correction factors and multiply by scalars before subtraction in BGV
            auto factors = balance_correction_factors(
                encrypted1.correction_factor(), encrypted2.correction_factor(), plain_modulus);

            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted1.data(), coeff_count, coeff_modulus_size), encrypted1.size(), get<1>(factors),
                coeff_modulus, PolyIter(encrypted1.data(), coeff_count, coeff_modulus_size));

            Ciphertext encrypted2_copy = encrypted2;
            multiply_poly_scalar_coeffmod(
                ConstPolyIter(encrypted2.data(), coeff_count, coeff_modulus_size), encrypted2.size(), get<2>(factors),
                coeff_modulus, PolyIter(encrypted2_copy.data(), coeff_count, coeff_modulus_size));

            // Set new correction factor
            encrypted1.correction_factor() = get<0>(factors);
            encrypted2_copy.correction_factor() = get<0>(factors);

            sub_inplace(encrypted1, encrypted2_copy);
        }
        else
        {
            // Prepare destination
            encrypted1.resize(context_, context_data.parms_id(), max_count);

            // Subtract ciphertexts
            uint64_t *d_encrypted1 = encrypted1.d_data();
            uint64_t *d_encrypted2 = encrypted2.d_data();
            uint64_t *d_modulus = parms.d_coeff_modulus_value();
            // Add ciphertexts
            const size_t threadsPerBlock = 256;
            size_t blocksPerGrid =
                (min_count * coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;
            eltwiseSubModSizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_encrypted1, d_encrypted1, d_encrypted2, d_modulus, coeff_count, coeff_modulus_size, min_count);

            // If encrypted2 has larger count, negate remaining entries
            if (encrypted1_size < encrypted2_size)
            {
                // 待测试
                blocksPerGrid =
                    ((encrypted2_size - min_count) * coeff_count * coeff_modulus_size + threadsPerBlock - 1) /
                    threadsPerBlock;
                negatePolyKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    d_encrypted2 + min_count * coeff_count * coeff_modulus_size, d_modulus, coeff_count,
                    coeff_modulus_size, min_count, d_encrypted1 + min_count * coeff_count * coeff_modulus_size);

            }
        }

        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted1.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::multiply_inplace(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted1, context_) || !is_buffer_valid(encrypted1))
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(encrypted2, context_) || !is_buffer_valid(encrypted2))
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (encrypted1.parms_id() != encrypted2.parms_id())
        {
            throw invalid_argument("encrypted1 and encrypted2 parameter mismatch");
        }

        auto context_data_ptr = context_.first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::bfv:
            bfv_multiply(encrypted1, encrypted2, pool);
            break;

        case scheme_type::ckks:
            ckks_multiply(encrypted1, encrypted2, pool);
            break;

        case scheme_type::bgv:
            bgv_multiply(encrypted1, encrypted2, pool);
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted1.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::bfv_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (encrypted1.is_ntt_form() || encrypted2.is_ntt_form())
        {
            throw invalid_argument("encrypted1 or encrypted2 cannot be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t base_q_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();
        uint64_t plain_modulus = parms.plain_modulus().value();

        auto rns_tool = context_data.rns_tool();
        size_t base_Bsk_size = rns_tool->base_Bsk()->size();
        size_t base_Bsk_m_tilde_size = rns_tool->base_Bsk_m_tilde()->size();

        // Determine destination.size()
        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_size, coeff_count, base_Bsk_m_tilde_size))
        {
            throw logic_error("invalid parameters");
        }

        // Set up iterators for bases
        auto base_q = iter(parms.coeff_modulus());
        auto &coeff_q_modulus = parms.coeff_modulus();
        uint64_t *base_q_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *base_q_modulus_ratio0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *base_q_modulus_ratio1 = parms.d_coeff_modulus_ratio_1();

        auto base_Bsk = iter(rns_tool->base_Bsk()->base());
        auto base_Bsk_modulus = rns_tool->base_Bsk()->base();

        uint64_t *d_base_Bsk_modulus_value = rns_tool->base_Bsk()->d_base();
        uint64_t *d_base_Bsk_modulus_ratio0 = rns_tool->base_Bsk()->d_ratio0();
        uint64_t *d_base_Bsk_modulus_ratio1 = rns_tool->base_Bsk()->d_ratio1();

        // Set up iterators for NTT tables
        auto base_q_ntt_tables = iter(context_data.small_ntt_tables());
        auto base_Bsk_ntt_tables = iter(rns_tool->base_Bsk_ntt_tables());

        uint64_t *d_inv_root_powers_Bsk = rns_tool->d_base_Bsk_root_powers();            

      
        // Microsoft SEAL uses BEHZ-style RNS multiplication. This process is somewhat complex and consists of the
        // following steps:
        //
        // (1) Lift encrypted1 and encrypted2 (initially in base q) to an extended base q U Bsk U {m_tilde}
        // (2) Remove extra multiples of q from the results with Montgomery reduction, switching base to q U Bsk
        // (3) Transform the data to NTT form
        // (4) Compute the ciphertext polynomial product using dyadic multiplication
        // (5) Transform the data back from NTT form
        // (6) Multiply the result by t (plain_modulus)
        // (7) Scale the result by q using a divide-and-floor algorithm, switching base to Bsk
        // (8) Use Shenoy-Kumaresan method to convert the result to base q

        print_helper<<<1,3>>>(encrypted1.d_data(), 3);


        // Resize encrypted1 to destination size
        encrypted1.resize_pure_gpu(context_, context_data.parms_id(), dest_size);

        // This lambda function takes as input an IterTuple with three components:
        //
        // 1. (Const)RNSIter to read an input polynomial from
        // 2. RNSIter for the output in base q
        // 3. RNSIter for the output in base Bsk
        //
        // It performs steps (1)-(3) of the BEHZ multiplication (see above) on the given input polynomial (given as an
        // RNSIter or ConstRNSIter) and writes the results in base q and base Bsk to the given output
        // iterators.

        // Allocate space for a base q output of behz_extend_base_convert_to_ntt for encrypted1
        // Allocate space for a base Bsk output of behz_extend_base_convert_to_ntt for encrypted1

        ensure_size(&d_encrypted1_q, encrypted1_size * coeff_count * base_q_size, d_encrypted1_q_size);
        ensure_size(&d_encrypted1_Bsk, encrypted1_size * coeff_count * base_Bsk_size, d_encrypted1_Bsk_size);
        ensure_size(&d_encrypted2_q, encrypted2_size * coeff_count * base_q_size, d_encrypted2_q_size);
        ensure_size(&d_encrypted2_Bsk, encrypted2_size * coeff_count * base_Bsk_size, d_encrypted2_Bsk_size);
        ensure_size(&d_temp_bfv_multiply, encrypted1_size * coeff_count * base_Bsk_m_tilde_size, d_temp_bfv_multiply_size);
        ensure_size(&d_temp_dest_q, dest_size * coeff_count * base_q_size, d_temp_dest_q_size);
        ensure_size(&d_temp_dest_Bsk, dest_size * coeff_count * base_Bsk_size, d_temp_dest_Bsk_size);
        ensure_size(&d_temp_q_Bsk, coeff_count * (base_q_size + base_Bsk_size) * dest_size, d_temp_q_Bsk_size);
        ensure_size(&d_temp_Bsk, coeff_count * base_Bsk_size, d_temp_Bsk_size);

        checkCudaErrors(cudaMemset(d_temp_dest_q, 0, dest_size * coeff_count * base_q_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(d_temp_dest_Bsk, 0, dest_size * coeff_count * base_Bsk_size * sizeof(uint64_t)));


        // Perform BEHZ steps (1)-(3) for encrypted1
        uint64_t *d_root_matrix = context_data.d_root_matrix();
        int *d_bit_count = context_data.d_bit_count();

        cudaStream_t ntt = 0;
        uint64_t temp_mu;



        printf("bfv multiply ntt test\n ");
        for(size_t i = 0; i < encrypted1_size; i++){
            checkCudaErrors(cudaMemcpy(d_encrypted1_q + i * coeff_count * base_q_size, encrypted1.d_data() + i * coeff_count * base_q_size, 
                coeff_count * base_q_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

            cudaDeviceSynchronize();
# if NTT_VERSION == 3
            ntt_v3(context_, encrypted1.parms_id(), d_encrypted1_q + i * coeff_count * base_q_size, base_q_size);
# else
            ntt_v1(context_, encrypted1.parms_id(), d_encrypted1_q + i * coeff_count * base_q_size, base_q_size);
# endif
            cudaDeviceSynchronize();
        }


        for (int i = 0; i < encrypted1_size; i++) {
            rns_tool->fastbconv_m_tilde_cuda(encrypted1.d_data() + i * coeff_count * base_q_size, 
                                            d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size);
            rns_tool->sm_mrq_cuda(d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size, 
                                            d_encrypted1_Bsk + i * coeff_count * base_Bsk_size);

        }

        for (size_t i = 0; i < encrypted1_size * base_Bsk_size; i++)
        {
            size_t j = i % base_Bsk_size;

            k_uint128_t mu1 = k_uint128_t::exp2(base_Bsk_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / base_Bsk_modulus[j].value()).low;
            forwardNTT(
                d_encrypted1_Bsk + i * coeff_count, 
                coeff_count, ntt, 
                base_Bsk_modulus[j].value(), temp_mu,
                base_Bsk_modulus[j].bit_count(), 
                d_inv_root_powers_Bsk + coeff_count * j);
            
        }

        for(size_t i = 0; i < encrypted2_size; i++){
            checkCudaErrors(cudaMemcpy(d_encrypted2_q + i * coeff_count * base_q_size, encrypted2.d_data() + i * coeff_count * base_q_size, 
                coeff_count * base_q_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

# if NTT_VERSION == 3
            ntt_v3(context_, encrypted2.parms_id(), d_encrypted2_q + i * coeff_count * base_q_size, base_q_size);
# else
            ntt_v1(context_, encrypted2.parms_id(), d_encrypted2_q + i * coeff_count * base_q_size, base_q_size);
# endif
        }

        for (int i = 0; i < encrypted2_size; i++) {
            rns_tool->fastbconv_m_tilde_cuda(encrypted2.d_data() + i * coeff_count * base_Bsk_m_tilde_size, 
                                            d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size);
            rns_tool->sm_mrq_cuda(d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size, 
                                            d_encrypted2_Bsk + i * coeff_count * base_Bsk_m_tilde_size);

        }

        for (size_t i = 0; i < encrypted2_size * base_Bsk_size; i++)
        {
            size_t j = i % base_Bsk_size;

            k_uint128_t mu1 = k_uint128_t::exp2(base_Bsk_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / base_Bsk_modulus[j].value()).low;
            forwardNTT(
                d_encrypted2_Bsk + i *coeff_count, coeff_count, ntt, base_Bsk_modulus[j].value(), temp_mu,
                base_Bsk_modulus[j].bit_count(), d_inv_root_powers_Bsk + coeff_count * j);
            
        }

        // Allocate temporary space for the output of step (4)

        bfv_multiply_helper1<<<(dest_size * base_Bsk_size * coeff_count + 255) / 256, 256>>>(
            d_encrypted1_q, d_encrypted2_q, d_encrypted1_Bsk, d_encrypted2_Bsk, d_temp_dest_q, d_temp_dest_Bsk,
            dest_size, encrypted1_size, encrypted2_size, coeff_count, base_q_size, base_Bsk_size, base_q_modulus_value,
            base_q_modulus_ratio0, base_q_modulus_ratio1, d_base_Bsk_modulus_value, d_base_Bsk_modulus_ratio0,
            d_base_Bsk_modulus_ratio1);


        // Perform BEHZ step (5): transform data from NTT form
        // Lazy reduction here. The following multiply_poly_scalar_coeffmod will correct the value back to [0, p)       
        uint64_t *d_inv_root_powers = context_data.d_root_powers();


        for (size_t i = 0; i < dest_size * base_q_size; i++)
        {
            size_t j = i % base_q_size;

            k_uint128_t mu1 = k_uint128_t::exp2(coeff_q_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / coeff_q_modulus[j].value()).low;
            inverseNTT(
                d_temp_dest_q + i *coeff_count, coeff_count, ntt, coeff_q_modulus[j].value(), temp_mu,
                coeff_q_modulus[j].bit_count(), d_inv_root_powers + coeff_count * j);
            
        }

        for (size_t i = 0; i < dest_size * base_Bsk_size; i++)
        {
            size_t j = i % base_Bsk_size;

            k_uint128_t mu1 = k_uint128_t::exp2(base_Bsk_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / base_Bsk_modulus[j].value()).low;
            inverseNTT(
                d_temp_dest_Bsk + i *coeff_count, coeff_count, ntt, base_Bsk_modulus[j].value(), temp_mu,
                base_Bsk_modulus[j].bit_count(), d_inv_root_powers_Bsk + coeff_count * j);
            
        }

        bfv_multiply_helper2<<<(coeff_count * dest_size * base_Bsk_size +255) /256 , 256>>>(
                                                                            d_temp_dest_q, 
                                                                            d_temp_dest_Bsk,
                                                                            d_temp_q_Bsk, 
                                                                            plain_modulus,
                                                                            coeff_count, 
                                                                            dest_size, 
                                                                            base_q_size, 
                                                                            base_Bsk_size,  
                                                                            base_q_modulus_value, 
                                                                            base_q_modulus_ratio1,
                                                                            d_base_Bsk_modulus_value,
                                                                            d_base_Bsk_modulus_ratio1);
        // Perform BEHZ steps (6)-(8)
        for(int i = 0; i < dest_size; i++) {
            rns_tool->fast_floor_cuda(d_temp_q_Bsk + i * coeff_count*(base_q_size + base_Bsk_size), d_temp_Bsk);
            // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
            rns_tool->fastbconv_sk_cuda(d_temp_Bsk, encrypted1.d_data() + base_q_size * coeff_count * i);
        }

        print_helper<<<1,3>>>(encrypted1.d_data(), 3);


    }

    void Evaluator::ckks_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (!(encrypted1.is_ntt_form() && encrypted2.is_ntt_form()))
        {
            throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Set up iterator for the base
        auto coeff_modulus = iter(parms.coeff_modulus());

        // Prepare destination
        encrypted1.resize_pure_gpu(context_, context_data.parms_id(), dest_size);


        // Set up iterators for input ciphertexts
        PolyIter encrypted1_iter = iter(encrypted1);
        ConstPolyIter encrypted2_iter = iter(encrypted2);

        uint64_t *d_encrypted1 = encrypted1.d_data();
        uint64_t *d_encrypted2 = encrypted2.d_data();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();

        if (dest_size == 3)
        {
            std::size_t threadsPerBlock = 256;
            std::size_t blocksPerGrid = (coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;

            ckks_bgv_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_encrypted1, d_encrypted2, coeff_count, coeff_modulus_size, d_coeff_modulus_value,
                d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1, d_encrypted1);
        }
        else
        {
            // Allocate temporary space for the result

            uint64_t *d_temp = nullptr;
            uint64_t *d_prod = nullptr;
            allocate_gpu<uint64_t>(&d_temp, dest_size * coeff_count * coeff_modulus_size);
            allocate_gpu<uint64_t>(&d_prod, coeff_count);
            for(int i = 0; i < dest_size; i++){
                size_t curr_encrypted1_last = min<size_t>(i, encrypted1_size - 1);
                size_t curr_encrypted2_first = min<size_t>(i, encrypted2_size - 1);
                size_t curr_encrypted1_first = i - curr_encrypted2_first;
                size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                uint64_t *shifted_encrypted1 = encrypted1.d_data() + (curr_encrypted1_first + coeff_modulus_size * i) * coeff_count;
                // 2要反向遍历
                uint64_t *shifted_encrypted2 = encrypted2.d_data() + (curr_encrypted2_first + coeff_modulus_size * i) * coeff_count;


                for(int j = 0; j < steps; j++) {
                    dyadic_product_coeffmod_kernel<<<(coeff_count + 255) / 256, 256>>>(shifted_encrypted1 + j * coeff_count, 
                                                                                        shifted_encrypted2 - j * coeff_count, 
                                                                                        coeff_count, 
                                                                                        coeff_modulus[j].value(), 
                                                                                        coeff_modulus[j].const_ratio().data()[0],
                                                                                        coeff_modulus[j].const_ratio().data()[1],
                                                                                        d_prod);
                    add_poly_coeffmod_kernel<<<(coeff_count + 255) / 256, 256>>>(d_prod, 
                                                                                d_temp + (i * coeff_modulus_size + j) * coeff_count, 
                                                                                coeff_count, 
                                                                                coeff_modulus[j].value(), 
                                                                                d_temp + (i * coeff_modulus_size + j) * coeff_count);
                }
            }


            checkCudaErrors(cudaMemcpy(encrypted1.d_data(), d_temp, dest_size * coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
            deallocate_gpu<uint64_t>(&d_temp, dest_size * coeff_count * coeff_modulus_size);
            deallocate_gpu<uint64_t>(&d_prod, coeff_count);
        }

        // Set the scale
        encrypted1.scale() *= encrypted2.scale();
        if (!is_scale_within_bounds(encrypted1.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::bgv_multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, MemoryPoolHandle pool) const
    {
        if (!encrypted1.is_ntt_form() || !encrypted2.is_ntt_form())
        {
            throw invalid_argument("encrypted1 or encrypted2 must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted1.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted1_size = encrypted1.size();
        size_t encrypted2_size = encrypted2.size();

        uint64_t *d_encrypted1 = encrypted1.d_data();
        uint64_t *d_encrypted2 = encrypted2.d_data();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();
        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_size = sub_safe(add_safe(encrypted1_size, encrypted2_size), size_t(1));

        // Set up iterator for the base
        auto coeff_modulus = iter(parms.coeff_modulus());

        // Prepare destination
        encrypted1.resize_pure_gpu(context_, context_data.parms_id(), dest_size);
        // Convert c0 and c1 to ntt
        // Set up iterators for input ciphertexts
        PolyIter encrypted1_iter = iter(encrypted1);
        ConstPolyIter encrypted2_iter = iter(encrypted2);

        if (dest_size == 3)
        {
            std::size_t threadsPerBlock = 256;
            std::size_t blocksPerGrid = (coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;

            ckks_bgv_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_encrypted1, d_encrypted2, coeff_count, coeff_modulus_size, d_coeff_modulus_value,
                d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1, d_encrypted1);

        }
        else
        {
            // Allocate temporary space for the result
            uint64_t *d_temp = nullptr;
            uint64_t *d_prod = nullptr;
            allocate_gpu<uint64_t>(&d_temp, dest_size * coeff_count * coeff_modulus_size);
            allocate_gpu<uint64_t>(&d_prod, coeff_count);
            for(int i = 0; i < dest_size; i++){
                size_t curr_encrypted1_last = min<size_t>(i, encrypted1_size - 1);
                size_t curr_encrypted2_first = min<size_t>(i, encrypted2_size - 1);
                size_t curr_encrypted1_first = i - curr_encrypted2_first;
                size_t steps = curr_encrypted1_last - curr_encrypted1_first + 1;

                uint64_t *shifted_encrypted1 = encrypted1.d_data() + (curr_encrypted1_first + coeff_modulus_size * i) * coeff_count;
                // 2要反向遍历
                uint64_t *shifted_encrypted2 = encrypted2.d_data() + (curr_encrypted2_first + coeff_modulus_size * i) * coeff_count;


                for(int j = 0; j < steps; j++) {
                    dyadic_product_coeffmod_kernel<<<(coeff_count + 255) / 256, 256>>>(shifted_encrypted1 + j * coeff_count, 
                                                                                        shifted_encrypted2 - j * coeff_count, 
                                                                                        coeff_count, 
                                                                                        coeff_modulus[j].value(), 
                                                                                        coeff_modulus[j].const_ratio().data()[0],
                                                                                        coeff_modulus[j].const_ratio().data()[1],
                                                                                        d_prod);
                    add_poly_coeffmod_kernel<<<(coeff_count + 255) / 256, 256>>>(d_prod, 
                                                                                d_temp + (i * coeff_modulus_size + j) * coeff_count, 
                                                                                coeff_count, 
                                                                                coeff_modulus[j].value(), 
                                                                                d_temp + (i * coeff_modulus_size + j) * coeff_count);
                }
            }



            checkCudaErrors(cudaMemcpy(encrypted1.d_data(), d_temp, dest_size * coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
            deallocate_gpu<uint64_t>(&d_temp, dest_size * coeff_count * coeff_modulus_size);
            deallocate_gpu<uint64_t>(&d_prod, coeff_count);

        }


        // Set the correction factor
        encrypted1.correction_factor() =
            multiply_uint_mod(encrypted1.correction_factor(), encrypted2.correction_factor(), parms.plain_modulus());
    }

    void Evaluator::square_inplace(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.first_context_data();
        switch (context_data_ptr->parms().scheme())
        {
            // 直接用乘法可以吗？
        case scheme_type::bfv:
            bfv_square(encrypted, move(pool));
            break;

        case scheme_type::ckks:
            ckks_square(encrypted, move(pool));
            break;

        case scheme_type::bgv:
            bgv_square(encrypted, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::bfv_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t base_q_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        uint64_t plain_modulus = parms.plain_modulus().value();

        auto rns_tool = context_data.rns_tool();
        size_t base_Bsk_size = rns_tool->base_Bsk()->size();
        size_t base_Bsk_m_tilde_size = rns_tool->base_Bsk_m_tilde()->size();

        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_size != 2)
        {
            bfv_multiply(encrypted, encrypted, move(pool));
            return;
        }

        // Determine destination.size()
        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_size, coeff_count, base_Bsk_m_tilde_size))
        {
            throw logic_error("invalid parameters");
        }

        // Set up iterators for bases
        auto base_q = iter(parms.coeff_modulus());
        
        auto &coeff_q_modulus = parms.coeff_modulus();
        uint64_t *base_q_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *base_q_modulus_ratio0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *base_q_modulus_ratio1 = parms.d_coeff_modulus_ratio_1();


        auto base_Bsk = iter(rns_tool->base_Bsk()->base());

        auto base_Bsk_modulus = rns_tool->base_Bsk()->base();
        uint64_t *d_base_Bsk_modulus_value = rns_tool->base_Bsk()->d_base();
        uint64_t *d_base_Bsk_modulus_ratio0 = rns_tool->base_Bsk()->d_ratio0();
        uint64_t *d_base_Bsk_modulus_ratio1 = rns_tool->base_Bsk()->d_ratio1();


        // Set up iterators for NTT tables
        auto base_q_ntt_tables = iter(context_data.small_ntt_tables());
        auto base_Bsk_ntt_tables = iter(rns_tool->base_Bsk_ntt_tables());

        // Microsoft SEAL uses BEHZ-style RNS multiplication. For details, see Evaluator::bfv_multiply. This function
        // uses additionally Karatsuba multiplication to reduce the complexity of squaring a size-2 ciphertext, but the
        // steps are otherwise the same as in Evaluator::bfv_multiply.

        // Resize encrypted to destination size
        encrypted.resize_pure_gpu(context_, context_data.parms_id(), dest_size);

        // This lambda function takes as input an IterTuple with three components:
        //
        // 1. (Const)RNSIter to read an input polynomial from
        // 2. RNSIter for the output in base q
        // 3. RNSIter for the output in base Bsk
        //
        // It performs steps (1)-(3) of the BEHZ multiplication on the given input polynomial (given as an RNSIter
        // or ConstRNSIter) and writes the results in base q and base Bsk to the given output iterators.

        ensure_size(&d_encrypted1_q, encrypted_size * coeff_count * base_q_size, d_encrypted1_q_size);
        ensure_size(&d_encrypted1_Bsk, encrypted_size * coeff_count * base_Bsk_size, d_encrypted1_Bsk_size);
        ensure_size(&d_temp_bfv_multiply, encrypted_size * coeff_count * base_Bsk_m_tilde_size, d_temp_bfv_multiply_size);
        ensure_size(&d_temp_dest_q, dest_size * coeff_count * base_q_size, d_temp_dest_q_size);
        ensure_size(&d_temp_dest_Bsk, dest_size * coeff_count * base_Bsk_size, d_temp_dest_Bsk_size);
        ensure_size(&d_temp_q_Bsk, coeff_count * (base_q_size + base_Bsk_size) * dest_size, d_temp_q_Bsk_size);
        ensure_size(&d_temp_Bsk, coeff_count * base_Bsk_size, d_temp_Bsk_size);

        uint64_t *d_root_matrix = context_data.d_root_matrix();
        int *d_bit_count = context_data.d_bit_count();
        uint64_t *d_inv_root_powers_Bsk = rns_tool->d_base_Bsk_root_powers();            

        cudaStream_t ntt = 0;
        uint64_t temp_mu;

        for(size_t i = 0; i < encrypted_size; i++){
            checkCudaErrors(cudaMemcpy(d_encrypted1_q + i * coeff_count * base_q_size, encrypted.d_data() + i * coeff_count * base_q_size, 
                coeff_count * base_q_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
            cudaDeviceSynchronize();

# if NTT_VERSION == 3
            ntt_v3(context_, encrypted.parms_id(), d_encrypted1_q + i * coeff_count * base_q_size, base_q_size);
# else
            ntt_v1(context_, encrypted.parms_id(), d_encrypted1_q + i * coeff_count * base_q_size, base_q_size);
# endif
            cudaDeviceSynchronize();

        }

        for (int i = 0; i < encrypted_size; i++) {
            rns_tool->fastbconv_m_tilde_cuda(encrypted.d_data() + i * coeff_count * base_q_size, 
                                            d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size);
            rns_tool->sm_mrq_cuda(d_temp_bfv_multiply + i * coeff_count * base_Bsk_m_tilde_size, 
                                            d_encrypted1_Bsk + i * coeff_count * base_Bsk_size);

        }

        for (size_t i = 0; i < encrypted_size * base_Bsk_size; i++)
        {
            size_t j = i % base_Bsk_size;

            k_uint128_t mu1 = k_uint128_t::exp2(base_Bsk_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / base_Bsk_modulus[j].value()).low;
            forwardNTT(
                d_encrypted1_Bsk + i *coeff_count, 
                coeff_count, ntt, 
                base_Bsk_modulus[j].value(), temp_mu,
                base_Bsk_modulus[j].bit_count(), 
                d_inv_root_powers_Bsk + coeff_count * j);
            
        }

        // Perform BEHZ step (4): dyadic Karatsuba-squaring on size-2 ciphertexts

        // This lambda function computes the size-2 ciphertext square for BFV multiplication. Since we use the BEHZ
        // approach, the multiplication of individual polynomials is done using a dyadic product where the inputs
        // are already in NTT form. The arguments of the lambda function are expected to be as follows:
        //
        // 1. a ConstPolyIter pointing to the beginning of the input ciphertext (in NTT form)
        // 3. a ConstModulusIter pointing to an array of Modulus elements for the base
        // 4. the size of the base
        // 5. a PolyIter pointing to the beginning of the output ciphertext

        auto behz_ciphertext_square_cuda = [&](uint64_t *input, 
                                            uint64_t *modulu_value, uint64_t *modulu_ratio0, uint64_t *modulu_ratio1, 
                                            size_t base_size, uint64_t *output) {
            // Compute c0^2
            dyadic_product_coeffmod_kernel<<<(base_size * coeff_count + 255) / 255, 256>>>(input, input, coeff_count, base_size, 1, modulu_value, modulu_ratio0, modulu_ratio1, output);

            // Compute 2*c0*c1
            dyadic_product_coeffmod_kernel<<<(base_size * coeff_count + 255) / 255, 256>>>(input, input + base_size*coeff_count, coeff_count, base_size, 1, modulu_value, modulu_ratio0, modulu_ratio1, output + base_size*coeff_count);
            add_poly_coeffmod_kernel<<<(base_size * coeff_count + 255) / 255, 256>>>(output + base_size*coeff_count, output + base_size*coeff_count, coeff_count, base_size, modulu_value, output + base_size*coeff_count);

            // Compute c1^2
            dyadic_product_coeffmod_kernel<<<(base_size * coeff_count + 255) / 255, 256>>>(input + base_size*coeff_count, input + base_size*coeff_count, coeff_count, base_size, 1, modulu_value, modulu_ratio0, modulu_ratio1, output + 2*base_size*coeff_count);;
        };

        // Perform the BEHZ ciphertext square both for base q and base Bsk
        behz_ciphertext_square_cuda(d_encrypted1_q, base_q_modulus_value, base_q_modulus_ratio0, base_q_modulus_ratio1, base_q_size, d_temp_dest_q);
        behz_ciphertext_square_cuda(d_encrypted1_Bsk, d_base_Bsk_modulus_value, d_base_Bsk_modulus_ratio0, d_base_Bsk_modulus_ratio1, base_Bsk_size, d_temp_dest_Bsk);

        // Perform BEHZ step (5): transform data from NTT form

        uint64_t *d_inv_root_powers = context_data.d_root_powers();
        
        for (size_t i = 0; i < dest_size * base_q_size; i++)
        {
            size_t j = i % base_q_size;

            k_uint128_t mu1 = k_uint128_t::exp2(coeff_q_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / coeff_q_modulus[j].value()).low;
            inverseNTT(
                d_temp_dest_q + i *coeff_count, coeff_count, ntt, coeff_q_modulus[j].value(), temp_mu,
                coeff_q_modulus[j].bit_count(), d_inv_root_powers + coeff_count * j);
            
        }

 
        for (size_t i = 0; i < dest_size * base_Bsk_size; i++)
        {
            size_t j = i % base_Bsk_size;

            k_uint128_t mu1 = k_uint128_t::exp2(base_Bsk_modulus[j].bit_count() * 2);
            temp_mu = (mu1 / base_Bsk_modulus[j].value()).low;
            inverseNTT(
                d_temp_dest_Bsk + i *coeff_count, coeff_count, ntt, base_Bsk_modulus[j].value(), temp_mu,
                base_Bsk_modulus[j].bit_count(), d_inv_root_powers_Bsk + coeff_count * j);
            
        }

        bfv_multiply_helper2<<<(coeff_count * dest_size * base_Bsk_size +255) /256 , 256>>>(
                                                                            d_temp_dest_q, 
                                                                            d_temp_dest_Bsk,
                                                                            d_temp_q_Bsk, 
                                                                            plain_modulus,
                                                                            coeff_count, 
                                                                            dest_size, 
                                                                            base_q_size, 
                                                                            base_Bsk_size,  
                                                                            base_q_modulus_value, 
                                                                            base_q_modulus_ratio1,
                                                                            d_base_Bsk_modulus_value,
                                                                            d_base_Bsk_modulus_ratio1);
        // Perform BEHZ steps (6)-(8)
        for(int i = 0; i < dest_size; i++) {
            rns_tool->fast_floor_cuda(d_temp_q_Bsk + i * coeff_count*(base_q_size + base_Bsk_size), d_temp_Bsk);
            // Step (8): use Shenoy-Kumaresan method to convert the result to base q and write to encrypted1
            rns_tool->fastbconv_sk_cuda(d_temp_Bsk, encrypted.d_data() + base_q_size * coeff_count * i);
        }


    }

    void Evaluator::ckks_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();

        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();

        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_size != 2)
        {
            // printf("encrypted size != 2\n");
            ckks_multiply(encrypted, encrypted, move(pool));
            return;
        }

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Set up iterator for the base

        // Prepare destination
        encrypted.resize_pure_gpu(context_, context_data.parms_id(), dest_size);
        uint64_t *d_encrypted = encrypted.d_data();

        // Set up iterators for input ciphertext

        std::size_t threadsPerBlock = 256;
        std::size_t blocksPerGrid = (coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;

        ckks_bgv_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_encrypted, d_encrypted, coeff_count, coeff_modulus_size, d_coeff_modulus_value, d_coeff_modulus_ratio_0,
            d_coeff_modulus_ratio_1, d_encrypted);

        // Set the scale
        encrypted.scale() *= encrypted.scale();
        if (!is_scale_within_bounds(encrypted.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::bgv_square(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_size = encrypted.size();

        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();
        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_size != 2)
        {
            bgv_multiply(encrypted, encrypted, move(pool));
            return;
        }

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        size_t dest_size = sub_safe(add_safe(encrypted_size, encrypted_size), size_t(1));

        // Size check
        if (!product_fits_in(dest_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Set up iterator for the base
        auto coeff_modulus = iter(parms.coeff_modulus());

        // Prepare destination
        encrypted.resize_pure_gpu(context_, context_data.parms_id(), dest_size);
        uint64_t *d_encrypted = encrypted.d_data();

        // Set up iterators for input ciphertext

        std::size_t threadsPerBlock = 256;
        std::size_t blocksPerGrid = (coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;

        ckks_bgv_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_encrypted, d_encrypted, coeff_count, coeff_modulus_size, d_coeff_modulus_value, d_coeff_modulus_ratio_0,
            d_coeff_modulus_ratio_1, d_encrypted);
        
        // Set the correction factor
        encrypted.correction_factor() =
            multiply_uint_mod(encrypted.correction_factor(), encrypted.correction_factor(), parms.plain_modulus());
    }

    void Evaluator::relinearize_internal(
        Ciphertext &encrypted, const RelinKeys &relin_keys, size_t destination_size, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (relin_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }

        size_t encrypted_size = encrypted.size();

        // Verify parameters.
        if (destination_size < 2 || destination_size > encrypted_size)
        {
            throw invalid_argument("destination_size must be at least 2 and less than or equal to current count");
        }
        if (relin_keys.size() < sub_safe(encrypted_size, size_t(2)))
        {
            throw invalid_argument("not enough relinearization keys");
        }

        // If encrypted is already at the desired level, return
        if (destination_size == encrypted_size)
        {
            return;
        }

        // Calculate number of relinearize_one_step calls needed
        size_t relins_needed = encrypted_size - destination_size;

        // Iterator pointing to the last component of encrypted
        auto encrypted_iter = iter(encrypted);
        encrypted_iter += encrypted_size - 1;

        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        SEAL_ITERATE(iter(size_t(0)), relins_needed, [&](auto I) {
            if ((context_data_ptr->parms().scheme() == scheme_type::ckks) || (context_data_ptr->parms().scheme() == scheme_type::bfv)){
                this->switch_key_inplace_cuda(
                    encrypted, encrypted.d_data() + (encrypted_size - 1) * coeff_count * coeff_modulus_size,
                    static_cast<const KSwitchKeys &>(relin_keys), RelinKeys::get_index(encrypted_size - 1 - I), pool);
            } else {
                this->switch_key_inplace_bgv(
                    encrypted, encrypted.d_data() + (encrypted_size - 1) * coeff_count * coeff_modulus_size,
                    static_cast<const KSwitchKeys &>(relin_keys), RelinKeys::get_index(encrypted_size - 1 - I), pool);
            }

        });

        // Put the output of final relinearization into destination.
        // Prepare destination only at this point because we are resizing down
        encrypted.resize(context_, context_data_ptr->parms_id(), destination_size);


        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::mod_switch_scale_to_next(const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
        // Assuming at this point encrypted is already validated.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_data_ptr->parms().scheme() == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (context_data_ptr->parms().scheme() == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (context_data_ptr->parms().scheme() == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &next_context_data = *context_data.next_context_data();
        auto &next_parms = next_context_data.parms();
        auto &parms_data = context_data.parms();
        auto rns_tool = context_data.rns_tool();

        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();
        uint64_t *d_root_powers = context_data.d_root_powers();

        size_t coeff_modulus_size = context_data.parms().coeff_modulus().size();

        size_t encrypted_size = encrypted.size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();



        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();

        std::pair<int, int> split_result = context_data.split_degree();
        uint64_t *prev_modulu_value = parms_data.d_coeff_modulus_value();
        uint64_t *prev_ratio0 = parms_data.d_coeff_modulus_ratio_0();
        uint64_t *prev_ratio1 = parms_data.d_coeff_modulus_ratio_1();
        int *d_bit_count = context_data.d_bit_count();
        uint64_t *d_roots = context_data.d_roots();
        uint64_t *d_root_matrix_n1 = context_data.d_root_matrix_n1();
        uint64_t *d_root_matrix_n2 = context_data.d_root_matrix_n2();
        uint64_t *d_root_matrix_n12 = context_data.d_root_matrix_n12();


        uint64_t *d_encrypted = encrypted.d_data();
        uint64_t *d_copy = nullptr;
        allocate_gpu<uint64_t>(&d_copy, encrypted_size * coeff_count * coeff_modulus_size);

        checkCudaErrors(cudaMemcpy(
            d_copy, d_encrypted, encrypted_size * coeff_count * coeff_modulus_size * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));

        Ciphertext encrypted_copy(pool);

        switch (next_parms.scheme())
        {
        case scheme_type::bfv:
            for (int i = 0; i < encrypted_size; i++) {
                rns_tool->divide_and_rount_q_last_inplace_cuda(d_copy + i * coeff_count * coeff_modulus_size);
            }

            destination.resize_pure_gpu(context_, next_context_data.parms_id(), encrypted_size);
            diff_modulus_size_copy_kernel<<<
                            (encrypted_size * coeff_count * next_coeff_modulus_size + 255) / 256, 256>>>(
                            d_copy, destination.d_data(), encrypted_size, coeff_count, coeff_modulus_size, next_coeff_modulus_size);

            break;


        case scheme_type::ckks:
            for (int i = 0; i < encrypted_size; i++)
            {
#if NTT_VERSION == 3
                rns_tool->divide_and_round_q_last_ntt_inplace_cuda_test(
                    d_copy + i * coeff_count * coeff_modulus_size, 
                    d_root_matrix_n1, d_root_matrix_n2, d_root_matrix_n12,
                    prev_modulu_value, prev_ratio0, prev_ratio1, d_roots, d_bit_count,
                    split_result,
                    d_inv_root_powers,
                    context_data.small_ntt_tables());
#else
                rns_tool->divide_and_round_q_last_ntt_inplace_cuda_v1(
                    d_copy + i * coeff_count * coeff_modulus_size, 
                    d_root_powers,
                    d_inv_root_powers,
                    context_data.small_ntt_tables(),
                    ntt_steam, stream_num);
#endif
            }

            destination.resize_pure_gpu(context_, next_context_data.parms_id(), encrypted_size);
            diff_modulus_size_copy_kernel<<<
                (encrypted_size * coeff_count * next_coeff_modulus_size + 255) / 256, 256>>>(
                d_copy, destination.d_data(), encrypted_size, coeff_count, coeff_modulus_size, next_coeff_modulus_size);

            break;

        case scheme_type::bgv:

            for (int i = 0; i < encrypted_size; i++)
            {
#if NTT_VERSION == 3
                rns_tool->mod_t_and_divide_q_last_ntt_inplace_cuda(
                    d_copy + i * coeff_count * coeff_modulus_size, 
                    d_root_matrix_n1, d_root_matrix_n2, d_root_matrix_n12,
                    prev_modulu_value, prev_ratio0, prev_ratio1, d_roots, d_bit_count,
                    split_result,
                    d_inv_root_powers,
                    context_data.small_ntt_tables());
#else
                rns_tool->mod_t_and_divide_q_last_ntt_inplace_cuda_v1(
                    d_copy + i * coeff_count * coeff_modulus_size, 
                    d_root_powers,
                    d_inv_root_powers,
                    context_data.small_ntt_tables(),
                    ntt_steam, stream_num);
#endif
            }

            destination.resize_pure_gpu(context_, next_context_data.parms_id(), encrypted_size);
            diff_modulus_size_copy_kernel<<<
                (encrypted_size * coeff_count * next_coeff_modulus_size + 255) / 256, 256>>>(
                d_copy, destination.d_data(), encrypted_size, coeff_count, coeff_modulus_size, next_coeff_modulus_size);
            
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }

        // Set other attributes
        destination.is_ntt_form() = encrypted.is_ntt_form();
        if (next_parms.scheme() == scheme_type::ckks)
        {
            // Change the scale when using CKKS
            destination.scale() =
                encrypted.scale() / static_cast<double>(context_data.parms().coeff_modulus().back().value());
        }
        else if (next_parms.scheme() == scheme_type::bgv)
        {
            // Change the correction factor when using BGV
            destination.correction_factor() = multiply_uint_mod(
                encrypted.correction_factor(), rns_tool->inv_q_last_mod_t(), next_parms.plain_modulus());
        }
        deallocate_gpu<uint64_t>(&d_copy, encrypted_size * coeff_count * coeff_modulus_size);

    }

    void Evaluator::mod_switch_drop_to_next(
        const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
        // Assuming at this point encrypted is already validated.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        size_t coeff_modulus_size = context_data_ptr->parms().coeff_modulus().size();
        if (context_data_ptr->parms().scheme() == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        
        // Extract encryption parameters.
        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = next_context_data.parms();

        if (!is_scale_within_bounds(encrypted.scale(), next_context_data))
        {
            throw invalid_argument("scale out of bounds");
        }

        // q_1,...,q_{k-1}
        size_t next_coeff_modulus_size = next_parms.coeff_modulus().size();
        size_t coeff_count = next_parms.poly_modulus_degree();
        size_t encrypted_size = encrypted.size();

        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, next_coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        auto drop_modulus_and_copy = [&](ConstPolyIter in_iter, PolyIter out_iter) {
            SEAL_ITERATE(iter(in_iter, out_iter), encrypted_size, [&](auto I) {
                SEAL_ITERATE(
                    iter(I), next_coeff_modulus_size, [&](auto J) { set_uint(get<0>(J), coeff_count, get<1>(J)); });
            });
        };

        if (&encrypted == &destination)
        {          
            uint64_t *d_temp = nullptr;
            allocate_gpu<uint64_t>(&d_temp, encrypted_size * coeff_count * next_coeff_modulus_size);

            // Switching in-place so need temporary space

            // Copy data over to temp; only copy the RNS components relevant after modulus drop
            mod_switch_helper<<<(encrypted_size*coeff_count*next_coeff_modulus_size + 255) / 256, 256>>>(
                encrypted.d_data(), d_temp, encrypted_size, coeff_count, coeff_modulus_size, next_coeff_modulus_size
            );

            // Resize destination before writing
            destination.resize_pure_gpu(context_, next_context_data.parms_id(), encrypted_size);
            checkCudaErrors(cudaMemcpy(destination.d_data(), d_temp, encrypted_size * coeff_count * next_coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

            // Copy data to destination
            // TODO: avoid copying and temporary space allocation
            deallocate_gpu<uint64_t>(&d_temp, encrypted_size * coeff_count * next_coeff_modulus_size);

        }
        else
        {
            // Resize destination before writing
            // destination.resize(context_, next_context_data.parms_id(), encrypted_size);
            destination.resize_pure_gpu(context_, next_context_data.parms_id(), encrypted_size);

            // Copy data over to destination; only copy the RNS components relevant after modulus drop
            mod_switch_helper<<<(encrypted_size*coeff_count*next_coeff_modulus_size + 255) / 256, 256>>>(
                encrypted.d_data(), destination.d_data(), encrypted_size, coeff_count, coeff_modulus_size, next_coeff_modulus_size
            );
        }
        destination.is_ntt_form() = true;
        destination.scale() = encrypted.scale();
        destination.correction_factor() = encrypted.correction_factor();
    }

    void Evaluator::mod_switch_drop_to_next(Plaintext &plain) const
    {
        // Assuming at this point plain is already validated.
        auto context_data_ptr = context_.get_context_data(plain.parms_id());
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (!context_data_ptr->next_context_data())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }

        // Extract encryption parameters.
        auto &next_context_data = *context_data_ptr->next_context_data();
        auto &next_parms = context_data_ptr->next_context_data()->parms();

        if (!is_scale_within_bounds(plain.scale(), next_context_data))
        {
            throw invalid_argument("scale out of bounds");
        }

        // q_1,...,q_{k-1}
        auto &next_coeff_modulus = next_parms.coeff_modulus();
        size_t next_coeff_modulus_size = next_coeff_modulus.size();
        size_t coeff_count = next_parms.poly_modulus_degree();

        // Compute destination size first for exception safety
        auto dest_size = mul_safe(next_coeff_modulus_size, coeff_count);

        plain.parms_id() = parms_id_zero;
        plain.resize(dest_size);
        plain.parms_id() = next_context_data.parms_id();
    }

    void Evaluator::mod_switch_to_next(
        const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_.first_context_data()->parms().scheme())
        {
        case scheme_type::bfv:
            // Modulus switching with scaling
            mod_switch_scale_to_next(encrypted, destination, move(pool));
            break;

        case scheme_type::ckks:
            // Modulus switching without scaling
            mod_switch_drop_to_next(encrypted, destination, move(pool));
            break;

        case scheme_type::bgv:
            mod_switch_scale_to_next(encrypted, destination, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (destination.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::mod_switch_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(encrypted, pool);
        }
    }

    // check
    void Evaluator::mod_switch_to_inplace(Plaintext &plain, parms_id_type parms_id) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(plain.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (!context_.get_context_data(parms_id))
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!plain.is_ntt_form())
        {
            throw invalid_argument("plain is not in NTT form");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (plain.parms_id() != parms_id)
        {
            mod_switch_to_next_inplace(plain);
        }
    }

    void Evaluator::rescale_to_next(const Ciphertext &encrypted, Ciphertext &destination, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_.first_context_data()->parms().scheme())
        {
        case scheme_type::bfv:
            /* Fall through */
        case scheme_type::bgv:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::ckks:
            // Modulus switching with scaling
            mod_switch_scale_to_next(encrypted, destination, move(pool));
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (destination.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::rescale_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        switch (context_data_ptr->parms().scheme())
        {
        case scheme_type::bfv:
            /* Fall through */
        case scheme_type::bgv:
            throw invalid_argument("unsupported operation for scheme type");

        case scheme_type::ckks:
            while (encrypted.parms_id() != parms_id)
            {
                // Modulus switching with scaling
                mod_switch_scale_to_next(encrypted, encrypted, pool);
            }
            break;

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::mod_reduce_to_next_inplace(Ciphertext &encrypted, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (context_.last_parms_id() == encrypted.parms_id())
        {
            throw invalid_argument("end of modulus switching chain reached");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        mod_switch_drop_to_next(encrypted, encrypted, std::move(pool));
// #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
//         // Transparent ciphertext output is not allowed.
//         if (encrypted.is_transparent())
//         {
//             throw logic_error("result ciphertext is transparent");
//         }
// #endif
    }

    void Evaluator::mod_reduce_to_inplace(Ciphertext &encrypted, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        auto target_context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (context_data_ptr->chain_index() < target_context_data_ptr->chain_index())
        {
            throw invalid_argument("cannot switch to higher level modulus");
        }

        while (encrypted.parms_id() != parms_id)
        {
            mod_reduce_to_next_inplace(encrypted, pool);
        }
    }

    void Evaluator::multiply_many(
        const vector<Ciphertext> &encrypteds, const RelinKeys &relin_keys, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (encrypteds.size() == 0)
        {
            throw invalid_argument("encrypteds vector must not be empty");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        for (size_t i = 0; i < encrypteds.size(); i++)
        {
            if (&encrypteds[i] == &destination)
            {
                throw invalid_argument("encrypteds must be different from destination");
            }
        }

        // There is at least one ciphertext
        auto context_data_ptr = context_.get_context_data(encrypteds[0].parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypteds is not valid for encryption parameters");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();

        if (parms.scheme() != scheme_type::bfv && parms.scheme() != scheme_type::bgv)
        {
            throw logic_error("unsupported scheme");
        }

        // If there is only one ciphertext, return it.
        if (encrypteds.size() == 1)
        {
            destination = encrypteds[0];
            return;
        }

        // Do first level of multiplications
        vector<Ciphertext> product_vec;
        for (size_t i = 0; i < encrypteds.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            if (encrypteds[i].data() == encrypteds[i + 1].data())
            {
                square(encrypteds[i], temp);
            }
            else
            {
                multiply(encrypteds[i], encrypteds[i + 1], temp);
            }
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(move(temp));
        }
        if (encrypteds.size() & 1)
        {
            product_vec.emplace_back(encrypteds.back());
        }

        // Repeatedly multiply and add to the back of the vector until the end is reached
        for (size_t i = 0; i < product_vec.size() - 1; i += 2)
        {
            Ciphertext temp(context_, context_data.parms_id(), pool);
            multiply(product_vec[i], product_vec[i + 1], temp);
            relinearize_inplace(temp, relin_keys, pool);
            product_vec.emplace_back(move(temp));
        }

        destination = product_vec.back();
    }

    void Evaluator::exponentiate_inplace(
        Ciphertext &encrypted, uint64_t exponent, const RelinKeys &relin_keys, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_.get_context_data(relin_keys.parms_id()))
        {
            throw invalid_argument("relin_keys is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be 0");
        }

        // Fast case
        if (exponent == 1)
        {
            return;
        }

        // Create a vector of copies of encrypted
        vector<Ciphertext> exp_vector(static_cast<size_t>(exponent), encrypted);
        multiply_many(exp_vector, relin_keys, encrypted, move(pool));
    }

    void Evaluator::add_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::bfv)
        {
            if (encrypted.is_ntt_form())
            {
                throw invalid_argument("BFV encrypted cannot be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BFV plain cannot be in NTT form");
            }
        }
        else if (parms.scheme() == scheme_type::ckks)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("CKKS encrypted must be in NTT form");
            }
            if (!plain.is_ntt_form())
            {
                throw invalid_argument("CKKS plain must be in NTT form");
            }
            if (encrypted.parms_id() != plain.parms_id())
            {
                throw invalid_argument("encrypted and plain parameter mismatch");
            }
            if (!are_same_scale(encrypted, plain))
            {
                throw invalid_argument("scale mismatch");
            }
        }
        else if (parms.scheme() == scheme_type::bgv)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("BGV encrypted must be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BGV plain cannot be in NTT form");
            }
        }

        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::bfv:
        {
            multiply_add_plain_with_scaling_variant_cuda(plain, context_data, encrypted.d_data());
            break;
        }

        case scheme_type::ckks:
        {
            add_poly_coeffmod_cuda(
                encrypted.d_data(), plain.d_data(), coeff_modulus_size, coeff_count, d_coeff_modulus_value,
                encrypted.d_data());
            break;
        }

        case scheme_type::bgv:
        {
            Plaintext plain_copy = plain;

            multiply_poly_scalar_coeffmod_kernel_one_modulu<<<(plain.coeff_count() + 255) / 256, 256>>>(
                                                                plain.d_data(), plain_copy.d_data(),
                                                                plain.coeff_count(), parms.plain_modulus().value(),
                                                                parms.plain_modulus().const_ratio().data()[0], 
                                                                encrypted.correction_factor());

            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), move(pool));

            add_poly_coeffmod_kernel<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(encrypted.d_data(),
                                                                                                        plain_copy.d_data(),
                                                                                                        coeff_count,
                                                                                                        coeff_modulus_size,
                                                                                                        d_coeff_modulus_value,
                                                                                                        encrypted.d_data());
            

            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::sub_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        if (parms.scheme() == scheme_type::bfv)
        {
            if (encrypted.is_ntt_form())
            {
                throw invalid_argument("BFV encrypted cannot be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BFV plain cannot be in NTT form");
            }
        }
        else if (parms.scheme() == scheme_type::ckks)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("CKKS encrypted must be in NTT form");
            }
            if (!plain.is_ntt_form())
            {
                throw invalid_argument("CKKS plain must be in NTT form");
            }
            if (encrypted.parms_id() != plain.parms_id())
            {
                throw invalid_argument("encrypted and plain parameter mismatch");
            }
            if (!are_same_scale(encrypted, plain))
            {
                throw invalid_argument("scale mismatch");
            }
        }
        else if (parms.scheme() == scheme_type::bgv)
        {
            if (!encrypted.is_ntt_form())
            {
                throw invalid_argument("BGV encrypted must be in NTT form");
            }
            if (plain.is_ntt_form())
            {
                throw invalid_argument("BGV plain cannot be in NTT form");
            }
        }

        // Extract encryption parameters.
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        switch (parms.scheme())
        {
        case scheme_type::bfv:
        {
            multiply_sub_plain_with_scaling_variant_cuda(plain, context_data, encrypted.d_data());
            break;
        }

        case scheme_type::ckks:
        {


            sub_poly_coeffmod_cuda(
                encrypted.d_data(), plain.d_data(), coeff_modulus_size, coeff_count, d_coeff_modulus_value,
                encrypted.d_data());

            break;
        }

        case scheme_type::bgv:
        {
            // Plaintext plain_copy = plain;
            // multiply_poly_scalar_coeffmod(
            //     plain.data(), plain.coeff_count(), encrypted.correction_factor(), parms.plain_modulus(),
            //     plain_copy.data());
            // transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), move(pool));
            // RNSIter encrypted_iter(encrypted.data(), coeff_count);
            // ConstRNSIter plain_iter(plain_copy.data(), coeff_count);
            // sub_poly_coeffmod(encrypted_iter, plain_iter, coeff_modulus_size, coeff_modulus, encrypted_iter);


            Plaintext plain_copy = plain;

            multiply_poly_scalar_coeffmod_kernel_one_modulu<<<(plain.coeff_count() + 255) / 256, 256>>>(
                                                                plain.d_data(), plain_copy.d_data(),
                                                                plain.coeff_count(), parms.plain_modulus().value(),
                                                                parms.plain_modulus().const_ratio().data()[0], 
                                                                encrypted.correction_factor());
            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), move(pool));
            sub_poly_coeffmod_cuda(encrypted.d_data(),
                                    plain_copy.d_data(),
                                    coeff_modulus_size,
                                    coeff_count,
                                    d_coeff_modulus_value,
                                    encrypted.d_data());

            break;
        }

        default:
            throw invalid_argument("unsupported scheme");
        }
        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::multiply_plain_inplace(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        // if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        // {
        //     throw invalid_argument("encrypted is not valid for encryption parameters");
        // }
        // if (!is_metadata_valid_for(plain, context_) || !is_buffer_valid(plain))
        // {
        //     throw invalid_argument("plain is not valid for encryption parameters");
        // }
        // if (!pool)
        // {
        //     throw invalid_argument("pool is uninitialized");
        // }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        if (encrypted.is_ntt_form() && plain.is_ntt_form())
        {
            // cout << "multiply_plain_inplace ntt ntt" << endl;
            multiply_plain_ntt(encrypted, plain);
        }
        else if (!encrypted.is_ntt_form() && !plain.is_ntt_form())
        {
            cout << "multiply_plain_inplace normal normal" << endl;
            multiply_plain_normal(encrypted, plain, move(pool));
        }
        else if (encrypted.is_ntt_form() && !plain.is_ntt_form())
        {
            cout << "multiply_plain_inplace ntt normal" << endl;
            
           Plaintext plain_copy = plain;
            transform_to_ntt_inplace(plain_copy, encrypted.parms_id(), move(pool));
            multiply_plain_ntt(encrypted, plain_copy);
        }
        else
        {
            cout << "multiply_plain_inplace normal ntt" << endl;
           transform_to_ntt_inplace(encrypted);
            multiply_plain_ntt(encrypted, plain);
            transform_from_ntt_inplace(encrypted);
        }

        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::multiply_plain_normal(Ciphertext &encrypted, const Plaintext &plain, MemoryPoolHandle pool) const
    {
        // Extract encryption parameters.
        printf("multiply_plain_normal\n");
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();
        uint64_t *d_plain_upper_half_increment = context_data.d_plain_upper_half_increment();
        auto ntt_tables = iter(context_data.small_ntt_tables());

        size_t encrypted_size = encrypted.size();
        size_t plain_coeff_count = plain.coeff_count();
        size_t plain_nonzero_coeff_count = plain.nonzero_coeff_count();

        // Size check
        if (!product_fits_in(encrypted_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        /*
        Optimizations for constant / monomial multiplication can lead to the presence of a timing side-channel in
        use-cases where the plaintext data should also be kept private.
        */

        uint64_t *d_modulu_value = parms.d_coeff_modulus_value();
        uint64_t *d_modulu_ratio0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_modulu_ratio1 = parms.d_coeff_modulus_ratio_1();

        if (plain_nonzero_coeff_count == 1)
        {
            printf("plain_nonzero_coeff_count == 1\n");
            // Multiplying by a monomial?
            size_t mono_exponent = plain.significant_coeff_count() - 1;

            if (plain[mono_exponent] >= plain_upper_half_threshold)
            {
                if (!context_data.qualifiers().using_fast_plain_lift)
                {
                    printf("not using_fast_plain_lift\n");
                    // Allocate temporary space for a single RNS coefficient
                    SEAL_ALLOCATE_GET_COEFF_ITER(temp, coeff_modulus_size, pool);

                    // We need to adjust the monomial modulo each coeff_modulus prime separately when the coeff_modulus
                    // primes may be larger than the plain_modulus. We add plain_upper_half_increment (i.e., q-t) to
                    // the monomial to ensure it is smaller than coeff_modulus and then do an RNS multiplication. Note
                    // that in this case plain_upper_half_increment contains a multi-precision integer, so after the
                    // addition we decompose the multi-precision integer into RNS components, and then multiply.
                    add_uint(plain_upper_half_increment, coeff_modulus_size, plain[mono_exponent], temp);
                    context_data.rns_tool()->base_q()->decompose(temp, pool);
                    negacyclic_multiply_poly_mono_coeffmod(
                        encrypted, encrypted_size, temp, mono_exponent, coeff_modulus, encrypted, pool);
                }
                else
                {
                    printf("using_fast_plain_lift\n");
                    // Every coeff_modulus prime is larger than plain_modulus, so there is no need to adjust the
                    // monomial. Instead, just do an RNS multiplication.
                    negacyclic_multiply_poly_mono_coeffmod_kernel<<<(encrypted_size * coeff_modulus_size * coeff_count + 255) / 256, 256>>> (
                        encrypted.d_data(), plain[mono_exponent], encrypted_size, coeff_count, coeff_modulus_size ,mono_exponent, d_modulu_value, d_modulu_ratio1);

                }
            }
            else
            {
                printf("plain[mono_exponent] < plain_upper_half_threshold\n");
                // The monomial represents a positive number, so no RNS multiplication is needed.
                negacyclic_multiply_poly_mono_coeffmod_kernel<<<(encrypted_size * coeff_modulus_size * coeff_count + 255) / 256, 256>>> (
                    encrypted.d_data(), plain[mono_exponent], encrypted_size, coeff_count, coeff_modulus_size ,mono_exponent, d_modulu_value, d_modulu_ratio1);

            }

            // Set the scale
            if (parms.scheme() == scheme_type::ckks)
            {
                encrypted.scale() *= plain.scale();
                if (!is_scale_within_bounds(encrypted.scale(), context_data))
                {
                    throw invalid_argument("scale out of bounds");
                }
            }

            return;
        }

        // Generic case: any plaintext polynomial
        // Allocate temporary space for an entire RNS polynomial
        uint64_t *d_temp = nullptr;
        allocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);
        uint64_t *d_encrypted = encrypted.d_data();
        uint64_t *d_plain = plain.d_data();

        if (!context_data.qualifiers().using_fast_plain_lift)
        {
            // StrideIter<uint64_t *> temp_iter(temp.get(), coeff_modulus_size);

            // SEAL_ITERATE(iter(plain_copy.data(), temp_iter), plain_coeff_count, [&](auto I) {
            //     auto plain_value = get<0>(I);
            //     if (plain_value >= plain_upper_half_threshold)
            //     {
            //         add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
            //     }
            //     else
            //     {
            //         *get<1>(I) = plain_value;
            //     }
            // });

            // context_data.rns_tool()->base_q()->decompose_array(temp_iter, coeff_count, pool);

            printf("bfv using fast plain lift\n");
            multiply_plain_normal_helper<<<(plain_coeff_count + 255) / 256, 256>>>(
                d_plain, d_temp, d_temp, plain_coeff_count, coeff_modulus_size, d_plain_upper_half_increment, plain_upper_half_threshold);

            context_data.rns_tool()->base_q()->decompose_array_cuda(d_temp, coeff_count);
        }
        else
        {
            // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the coeff_modulus
            multiply_plain_normal_helper2<<<(coeff_modulus_size * plain_coeff_count + 255) / 256, 256>>>(plain.d_data(), d_temp, 
            plain_coeff_count, coeff_modulus_size, plain_upper_half_threshold, d_plain_upper_half_increment);
        }

        // Need to multiply each component in encrypted with temp; first step is to transform to NTT form
        ntt_v1(context_, encrypted.parms_id(), d_temp, coeff_modulus_size, 0);
        uint64_t *d_inv_root_powers = context_data.d_root_powers();
        cudaStream_t ntt = 0;

        for(int i = 0; i < encrypted_size; i++){
            ntt_v1(context_, encrypted.parms_id(), d_encrypted + i * coeff_modulus_size * coeff_count, coeff_modulus_size, 0);
            dyadic_product_coeffmod_kernel<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(
                d_encrypted + i * coeff_modulus_size * coeff_count, 
                d_temp, 
                coeff_count, 
                coeff_modulus_size, 
                1,
                d_modulu_value,
                d_modulu_ratio0,
                d_modulu_ratio1,
                d_encrypted + i * coeff_modulus_size * coeff_count);
            

            for (int j = 0; j < coeff_modulus_size; j++) {
                k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[j].bit_count() * 2);
                uint64_t temp_mu = (mu1 / coeff_modulus[j].value()).low;

                inverseNTT(
                    d_encrypted + i * coeff_modulus_size * coeff_count + j * coeff_count, 
                    coeff_count, 
                    ntt, 
                    coeff_modulus[j].value(), 
                    temp_mu,
                    coeff_modulus[j].bit_count(), 
                    d_inv_root_powers + coeff_count * j);
            }
        }


        // Set the scale
        if (parms.scheme() == scheme_type::ckks)
        {
            encrypted.scale() *= plain.scale();
            if (!is_scale_within_bounds(encrypted.scale(), context_data))
            {
                throw invalid_argument("scale out of bounds");
            }
        }
    
        deallocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);
    }

    void Evaluator::multiply_plain_ntt(Ciphertext &encrypted_ntt, const Plaintext &plain_ntt) const
    {
        // Verify parameters.
        if (!plain_ntt.is_ntt_form())
        {
            throw invalid_argument("plain_ntt is not in NTT form");
        }
        if (encrypted_ntt.parms_id() != plain_ntt.parms_id())
        {
            throw invalid_argument("encrypted_ntt and plain_ntt parameter mismatch");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.get_context_data(encrypted_ntt.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();

        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_ntt_size = encrypted_ntt.size();
        size_t enc_coeff_modulus_size = encrypted_ntt.coeff_modulus_size();
        uint64_t *d_encrypted_ntt = encrypted_ntt.d_data();
        uint64_t *d_plain_ntt = plain_ntt.d_data();

        // Size check
        if (!product_fits_in(encrypted_ntt_size, coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        std::size_t threadsPerBlock = 256;
        std::size_t blocksPerGrid =
            (coeff_count * coeff_modulus_size * encrypted_ntt_size + threadsPerBlock - 1) / threadsPerBlock;

        dyadic_product_coeffmod_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_encrypted_ntt, d_plain_ntt, coeff_count, coeff_modulus_size, encrypted_ntt_size, d_coeff_modulus_value,
            d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1, d_encrypted_ntt);


        // Set the scale
        encrypted_ntt.scale() *= plain_ntt.scale();
        if (!is_scale_within_bounds(encrypted_ntt.scale(), context_data))
        {
            throw invalid_argument("scale out of bounds");
        }
    }

    void Evaluator::transform_to_ntt_inplace(Plaintext &plain, parms_id_type parms_id, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        // if (!is_valid_for(plain, context_))
        // {
        //     throw invalid_argument("plain is not valid for encryption parameters");
        // }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for the current context");
        }
        if (plain.is_ntt_form())
        {
            throw invalid_argument("plain is already in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t plain_coeff_count = plain.coeff_count();

        uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
        auto plain_upper_half_increment = context_data.plain_upper_half_increment();

        auto ntt_tables = iter(context_data.small_ntt_tables());

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Resize to fit the entire NTT transformed (ciphertext size) polynomial
        // Note that the new coefficients are automatically set to 0
        // plain.resize(coeff_count * coeff_modulus_size);
        plain.resize_gpu(coeff_count * coeff_modulus_size);
        // RNSIter plain_iter(plain.data(), coeff_count);

        uint64_t *d_plain_upper_half_increment = context_data.d_plain_upper_half_increment();

        // uint64_t *d_plain_upper_half_increment = nullptr;
        // cudaMalloc((void **)&d_plain_upper_half_increment, coeff_modulus_size * sizeof(uint64_t));
        // cudaMemcpy(d_plain_upper_half_increment, plain_upper_half_increment, coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

        if (!context_data.qualifiers().using_fast_plain_lift)
        {
            // 没有走这里的测试用例，待测试
            printf("using fast plain lift\n");
            // Allocate temporary space for an entire RNS polynomial
            // Slight semantic misuse of RNSIter here, but this works well
            SEAL_ALLOCATE_ZERO_GET_RNS_ITER(temp, coeff_modulus_size, coeff_count, pool);

            // uint64_t *d_temp = nullptr;
            // checkCudaErrors(cudaMalloc((void **)&d_temp, coeff_count * coeff_modulus_size * sizeof(uint64_t)));

            uint64_t *d_temp = nullptr;
            allocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);

            transform_helper<<<(plain_coeff_count + 255) / 256, 256>>> (plain.d_data(), d_temp, 
                                                                        plain_coeff_count, coeff_modulus_size,
                                                                        d_plain_upper_half_increment,
                                                                        plain_upper_half_threshold);




            // SEAL_ITERATE(iter(plain.data(), temp), plain_coeff_count, [&](auto I) {
            //     auto plain_value = get<0>(I);
            //     if (plain_value >= plain_upper_half_threshold)
            //     {
            //         add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
            //     }
            //     else
            //     {
            //         *get<1>(I) = plain_value;
            //     }
            // });

            checkCudaErrors(cudaMemcpy(temp, d_temp, coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            context_data.rns_tool()->base_q()->decompose_array(temp, coeff_count, pool);

            // Copy data back to plain
            set_poly(temp, coeff_count, coeff_modulus_size, plain.data());
            deallocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);

        }
        else
        {
            // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the coeff_modulus
            // primes.

            // Create a "reversed" helper iterator that iterates in the reverse order both plain RNS components and
            // the plain_upper_half_increment values.
            // auto helper_iter = reverse_iter(plain_iter, plain_upper_half_increment);
            // advance(helper_iter, -safe_cast<ptrdiff_t>(coeff_modulus_size - 1));

            transform_helper2<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(plain.d_data(), plain.d_data(), 
                                                                                            plain_coeff_count, 
                                                                                            coeff_modulus_size,
                                                                                            coeff_count,
                                                                                            d_plain_upper_half_increment,
                                                                                            plain_upper_half_threshold);

        }


        // cudaMemcpy(plain.d_data(), plain.data(), coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
        // Transform to NTT domain
        // ntt_negacyclic_harvey(plain_iter, coeff_modulus_size, ntt_tables);

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

#if NTT_VERSION == 3
        ntt_v3(context_, parms_id, plain.d_data(), coeff_modulus_size);
#else  
        ntt_v1(context_, parms_id, plain.d_data(), coeff_modulus_size);
#endif


        plain.parms_id() = parms_id;
    }

    void Evaluator::transform_to_ntt_inplace(Ciphertext &encrypted) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted is already in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Transform each polynomial to NTT domain

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        cudaStream_t ntt = 0;
        uint64_t temp_mu;
        for (int i = 0; i < encrypted_size; i++)
        {
#if NTT_VERSION == 3
            ntt_v3(context_, encrypted.parms_id(), encrypted.d_data() + i * coeff_count * coeff_modulus_size, coeff_modulus_size);
#else 
            ntt_v1(context_, encrypted.parms_id(), encrypted.d_data() + i * coeff_count * coeff_modulus_size, coeff_modulus_size);
#endif
        }


        // Finally change the is_ntt_transformed flag
        encrypted.is_ntt_form() = true;
// #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
//         // Transparent ciphertext output is not allowed.
//         if (encrypted.is_transparent())
//         {
//             throw logic_error("result ciphertext is transparent");
//         }
// #endif
    }

    void Evaluator::transform_from_ntt_inplace(Ciphertext &encrypted_ntt) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted_ntt, context_) || !is_buffer_valid(encrypted_ntt))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        auto context_data_ptr = context_.get_context_data(encrypted_ntt.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }
        if (!encrypted_ntt.is_ntt_form())
        {
            throw invalid_argument("encrypted_ntt is not in NTT form");
        }

        // Extract encryption parameters.
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t encrypted_ntt_size = encrypted_ntt.size();

        auto ntt_tables = iter(context_data.small_ntt_tables());
        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        cudaStream_t ntt = 0;
        uint64_t temp_mu;
        for (int i = 0; i < encrypted_ntt_size; i++)
        {
            for(int j = 0; j < coeff_modulus_size; j++)
            {
                k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[j].bit_count() * 2);
                temp_mu = (mu1 / coeff_modulus[j].value()).low;
                inverseNTT(
                    encrypted_ntt.d_data() + coeff_count * (i * coeff_modulus_size + j), 
                    coeff_count, ntt, coeff_modulus[j].value(), temp_mu,
                    coeff_modulus[j].bit_count(), d_inv_root_powers + coeff_count * j);
            }
        }
        
        // Transform each polynomial from NTT domain
        // inverse_ntt_negacyclic_harvey(encrypted_ntt, encrypted_ntt_size, ntt_tables);


        // Finally change the is_ntt_transformed flag
        encrypted_ntt.is_ntt_form() = false;
// #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
//         // Transparent ciphertext output is not allowed.
//         if (encrypted_ntt.is_transparent())
//         {
//             throw logic_error("result ciphertext is transparent");
//         }
// #endif
    }

    void Evaluator::apply_galois_inplace(
        Ciphertext &encrypted, uint32_t galois_elt, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Don't validate all of galois_keys but just check the parms_id.
        if (galois_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();
        // Use key_context_data where permutation tables exist since previous runs.
        auto galois_tool = context_.key_context_data()->galois_tool();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Check if Galois key is generated or not.
        if (!galois_keys.has_key(galois_elt))
        {
            throw invalid_argument("Galois key not present");
        }

        uint64_t m = mul_safe(static_cast<uint64_t>(coeff_count), uint64_t(2));

        // Verify parameters
        if (!(galois_elt & 1) || unsigned_geq(galois_elt, m))
        {
            throw invalid_argument("Galois element is not valid");
        }
        if (encrypted_size > 2)
        {
            throw invalid_argument("encrypted size must be 2");
        }

        SEAL_ALLOCATE_GET_RNS_ITER(temp, coeff_count, coeff_modulus_size, pool);
        auto step1 = chrono::high_resolution_clock::now();

        // DO NOT CHANGE EXECUTION ORDER OF FOLLOWING SECTION
        // BEGIN: Apply Galois for each ciphertext
        // Execution order is sensitive, since apply_galois is not inplace!

        if (parms.scheme() == scheme_type::bfv)
        {
            // !!! DO NOT CHANGE EXECUTION ORDER!!!
            // First transform encrypted.data(0)
            uint64_t *d_temp = nullptr;
            allocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);
            int coeff_count_power = get_power_of_two(coeff_count);


            apply_galois_helper_single<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(
                galois_elt, coeff_count, coeff_count_power, coeff_modulus_size, encrypted.d_data(),
                d_temp);

            checkCudaErrors(cudaMemcpy(encrypted.d_data(), d_temp, coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

            apply_galois_helper_single<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(
                galois_elt, coeff_count, coeff_count_power, coeff_modulus_size, 
                encrypted.d_data() + coeff_count * coeff_modulus_size , 
                d_temp);
            
            checkCudaErrors(cudaMemset(encrypted.d_data() + coeff_count * coeff_modulus_size, 0, coeff_count * coeff_modulus_size * sizeof(uint64_t)));

            switch_key_inplace_cuda(
                encrypted, d_temp, static_cast<const KSwitchKeys &>(galois_keys), GaloisKeys::get_index(galois_elt),
                pool);

            deallocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);


        }
        else if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            // First transform encrypted.data(0)

            uint64_t *d_temp = nullptr;
            allocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);
            int coeff_count_power = get_power_of_two(coeff_count);


            apply_galois_helper_single<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(
                galois_elt, coeff_count, coeff_count_power, coeff_modulus_size, encrypted.d_data(),
                d_temp);

            checkCudaErrors(cudaMemcpy(encrypted.d_data(), d_temp, coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

            apply_galois_helper_single<<<(coeff_modulus_size * coeff_count + 255) / 256, 256>>>(
                galois_elt, coeff_count, coeff_count_power, coeff_modulus_size, 
                encrypted.d_data() + coeff_count * coeff_modulus_size , 
                d_temp);
            
            checkCudaErrors(cudaMemset(encrypted.d_data() + coeff_count * coeff_modulus_size, 0, coeff_count * coeff_modulus_size * sizeof(uint64_t)));

            switch_key_inplace_cuda(
                encrypted, d_temp, static_cast<const KSwitchKeys &>(galois_keys), GaloisKeys::get_index(galois_elt),
                pool);

            deallocate_gpu<uint64_t>(&d_temp, coeff_count * coeff_modulus_size);


        }
        else
        {
            throw logic_error("scheme not implemented");
        }

        // #ifdef SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT
        //         // Transparent ciphertext output is not allowed.
        //         if (encrypted.is_transparent())
        //         {
        //             throw logic_error("result ciphertext is transparent");
        //         }
        // #endif
    }

    void Evaluator::rotate_internal(
        Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, MemoryPoolHandle pool) const
    {
        auto context_data_ptr = context_.get_context_data(encrypted.parms_id());
        if (!context_data_ptr)
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_data_ptr->qualifiers().using_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }
        if (galois_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }

        // Is there anything to do?
        if (steps == 0)
        {
            return;
        }

        size_t coeff_count = context_data_ptr->parms().poly_modulus_degree();
        auto galois_tool = context_data_ptr->galois_tool();

        auto &coeff_modulus = context_data_ptr->parms().coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t encrypted_size = encrypted.size();

        // Check if Galois key is generated or not.
        if (galois_keys.has_key(galois_tool->get_elt_from_step(steps)))
        {
            // Perform rotation and key switching
            apply_galois_inplace(encrypted, galois_tool->get_elt_from_step(steps), galois_keys, move(pool));
        }
        else
        {
            // Convert the steps to NAF: guarantees using smallest HW
            vector<int> naf_steps = naf(steps);

            // If naf_steps contains only one element, then this is a power-of-two
            // rotation and we would have expected not to get to this part of the
            // if-statement.
            if (naf_steps.size() == 1)
            {
                throw invalid_argument("Galois key not present");
            }

            SEAL_ITERATE(naf_steps.cbegin(), naf_steps.size(), [&](auto step) {
                // We might have a NAF-term of size coeff_count / 2; this corresponds
                // to no rotation so we skip it. Otherwise call rotate_internal.
                if (safe_cast<size_t>(abs(step)) != (coeff_count >> 1))
                {
                    // Apply rotation for this step
                    this->rotate_internal(encrypted, step, galois_keys, pool);
                }
            });
        }
    }

    void Evaluator::switch_key_inplace(
        Ciphertext &encrypted, ConstRNSIter target_iter, const KSwitchKeys &kswitch_keys, size_t kswitch_keys_index,
        MemoryPoolHandle pool) const
    {
        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto scheme = parms.scheme();

        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!target_iter)
        {
            throw invalid_argument("target_iter");
        }
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        // Don't validate all of kswitch_keys but just check the parms_id.
        if (kswitch_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }

        if (kswitch_keys_index >= kswitch_keys.data().size())
        {
            throw out_of_range("kswitch_keys_index");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (scheme == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        auto key_ntt_tables = iter(key_context_data.small_ntt_tables());
        auto modswitch_factors = key_context_data.rns_tool()->inv_q_last_mod_q();

        // Size check
        if (!product_fits_in(coeff_count, rns_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare input
        auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].data().size();

        for (auto &each_key : key_vector)
        {
            if (!is_metadata_valid_for(each_key, context_) || !is_buffer_valid(each_key))
            {
                throw invalid_argument("kswitch_keys is not valid for encryption parameters");
            }
        }

        // Create a copy of target_iter
        SEAL_ALLOCATE_GET_RNS_ITER(t_target, coeff_count, decomp_modulus_size, pool);
        set_uint(target_iter, decomp_modulus_size * coeff_count, t_target);

        // In CKKS or BGV, t_target is in NTT form; switch back to normal form
        if (scheme == scheme_type::ckks || scheme == scheme_type::bgv)
        {
            inverse_ntt_negacyclic_harvey(t_target, decomp_modulus_size, key_ntt_tables);
        }

        // Temporary result
        auto t_poly_prod(allocate_zero_poly_array(key_component_count, coeff_count, rns_modulus_size, pool));

        SEAL_ITERATE(iter(size_t(0)), rns_modulus_size, [&](auto I) {
            size_t key_index = (I == decomp_modulus_size ? key_modulus_size - 1 : I);

            // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
            size_t lazy_reduction_summand_bound = size_t(SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX);
            size_t lazy_reduction_counter = lazy_reduction_summand_bound;

            // Allocate memory for a lazy accumulator (128-bit coefficients)
            auto t_poly_lazy(allocate_zero_poly_array(key_component_count, coeff_count, 2, pool));

            // Semantic misuse of PolyIter; this is really pointing to the data for a single RNS factor
            PolyIter accumulator_iter(t_poly_lazy.get(), 2, coeff_count);

            // Multiply with keys and perform lazy reduction on product's coefficients
            SEAL_ITERATE(iter(size_t(0)), decomp_modulus_size, [&](auto J) {
                SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);
                ConstCoeffIter t_operand;

                // RNS-NTT form exists in input
                if ((scheme == scheme_type::ckks || scheme == scheme_type::bgv) && (I == J))
                {
                    t_operand = target_iter[J];
                }
                // Perform RNS-NTT conversion
                else
                {
                    // No need to perform RNS conversion (modular reduction)
                    if (key_modulus[J] <= key_modulus[key_index])
                    {
                        set_uint(t_target[J], coeff_count, t_ntt);
                    }
                    // Perform RNS conversion (modular reduction)
                    else
                    {
                        modulo_poly_coeffs(t_target[J], coeff_count, key_modulus[key_index], t_ntt);
                    }
                    // NTT conversion lazy outputs in [0, 4q)
                    ntt_negacyclic_harvey_lazy(t_ntt, key_ntt_tables[key_index]);
                    t_operand = t_ntt;
                }

                // Multiply with keys and modular accumulate products in a lazy fashion
                SEAL_ITERATE(iter(key_vector[J].data(), accumulator_iter), key_component_count, [&](auto K) {
                    if (!lazy_reduction_counter)
                    {
                        SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)), coeff_count, [&](auto L) {
                            unsigned long long qword[2]{ 0, 0 };
                            multiply_uint64(get<0>(L), get<1>(L), qword);

                            // Accumulate product of t_operand and t_key_acc to t_poly_lazy and reduce
                            add_uint128(qword, get<2>(L).ptr(), qword);
                            get<2>(L)[0] = barrett_reduce_128(qword, key_modulus[key_index]);
                            get<2>(L)[1] = 0;
                        });
                    }
                    else
                    {
                        // Same as above but no reduction
                        SEAL_ITERATE(iter(t_operand, get<0>(K)[key_index], get<1>(K)), coeff_count, [&](auto L) {
                            unsigned long long qword[2]{ 0, 0 };
                            multiply_uint64(get<0>(L), get<1>(L), qword);
                            add_uint128(qword, get<2>(L).ptr(), qword);
                            get<2>(L)[0] = qword[0];
                            get<2>(L)[1] = qword[1];
                        });
                    }
                });

                if (!--lazy_reduction_counter)
                {
                    lazy_reduction_counter = lazy_reduction_summand_bound;
                }
            });

            // PolyIter pointing to the destination t_poly_prod, shifted to the appropriate modulus
            PolyIter t_poly_prod_iter(t_poly_prod.get() + (I * coeff_count), coeff_count, rns_modulus_size);

            // Final modular reduction
            SEAL_ITERATE(iter(accumulator_iter, t_poly_prod_iter), key_component_count, [&](auto K) {
                if (lazy_reduction_counter == lazy_reduction_summand_bound)
                {
                    SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
                        get<1>(L) = static_cast<uint64_t>(*get<0>(L));
                    });
                }
                else
                {
                    // Same as above except need to still do reduction
                    SEAL_ITERATE(iter(get<0>(K), *get<1>(K)), coeff_count, [&](auto L) {
                        get<1>(L) = barrett_reduce_128(get<0>(L).ptr(), key_modulus[key_index]);
                    });
                }
            });
        });
        // Accumulated products are now stored in t_poly_prod

        // Perform modulus switching with scaling
        PolyIter t_poly_prod_iter(t_poly_prod.get(), coeff_count, rns_modulus_size);
        SEAL_ITERATE(iter(encrypted, t_poly_prod_iter), key_component_count, [&](auto I) {
            if (scheme == scheme_type::bgv)
            {
                const Modulus &plain_modulus = parms.plain_modulus();
                // qk is the special prime
                uint64_t qk = key_modulus[key_modulus_size - 1].value();
                uint64_t qk_inv_qp = context_.key_context_data()->rns_tool()->inv_q_last_mod_t();

                // Lazy reduction; this needs to be then reduced mod qi
                CoeffIter t_last(get<1>(I)[decomp_modulus_size]);
                inverse_ntt_negacyclic_harvey(t_last, key_ntt_tables[key_modulus_size - 1]);

                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(k, coeff_count, pool);
                modulo_poly_coeffs(t_last, coeff_count, plain_modulus, k);
                negate_poly_coeffmod(k, coeff_count, plain_modulus, k);
                if (qk_inv_qp != 1)
                {
                    multiply_poly_scalar_coeffmod(k, coeff_count, qk_inv_qp, plain_modulus, k);
                }

                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(delta, coeff_count, pool);
                SEAL_ALLOCATE_ZERO_GET_COEFF_ITER(c_mod_qi, coeff_count, pool);
                SEAL_ITERATE(iter(I, key_modulus, modswitch_factors, key_ntt_tables), decomp_modulus_size, [&](auto J) {
                    // delta = k mod q_i
                    modulo_poly_coeffs(k, coeff_count, get<1>(J), delta);
                    // delta = k * q_k mod q_i
                    multiply_poly_scalar_coeffmod(delta, coeff_count, qk, get<1>(J), delta);

                    // c mod q_i
                    modulo_poly_coeffs(t_last, coeff_count, get<1>(J), c_mod_qi);
                    // delta = c + k * q_k mod q_i
                    // c_{i} = c_{i} - delta mod q_i
                    SEAL_ITERATE(iter(delta, c_mod_qi), coeff_count, [&](auto K) {
                        get<0>(K) = add_uint_mod(get<0>(K), get<1>(K), get<1>(J));
                    });
                    ntt_negacyclic_harvey(delta, get<3>(J));
                    SEAL_ITERATE(iter(delta, get<0, 1>(J)), coeff_count, [&](auto K) {
                        get<1>(K) = sub_uint_mod(get<1>(K), get<0>(K), get<1>(J));
                    });

                    multiply_poly_scalar_coeffmod(get<0, 1>(J), coeff_count, get<2>(J), get<1>(J), get<0, 1>(J));

                    add_poly_coeffmod(get<0, 1>(J), get<0, 0>(J), coeff_count, get<1>(J), get<0, 0>(J));
                });
            }
            else
            {
                // Lazy reduction; this needs to be then reduced mod qi
                CoeffIter t_last(get<1>(I)[decomp_modulus_size]);
                inverse_ntt_negacyclic_harvey_lazy(t_last, key_ntt_tables[key_modulus_size - 1]);

                // Add (p-1)/2 to change from flooring to rounding.
                uint64_t qk = key_modulus[key_modulus_size - 1].value();
                uint64_t qk_half = qk >> 1;
                SEAL_ITERATE(t_last, coeff_count, [&](auto &J) {
                    J = barrett_reduce_64(J + qk_half, key_modulus[key_modulus_size - 1]);
                });

                SEAL_ITERATE(iter(I, key_modulus, key_ntt_tables, modswitch_factors), decomp_modulus_size, [&](auto J) {
                    SEAL_ALLOCATE_GET_COEFF_ITER(t_ntt, coeff_count, pool);

                    // (ct mod 4qk) mod qi
                    uint64_t qi = get<1>(J).value();
                    if (qk > qi)
                    {
                        // This cannot be spared. NTT only tolerates input that is less than 4*modulus (i.e. qk <=4*qi).
                        modulo_poly_coeffs(t_last, coeff_count, get<1>(J), t_ntt);
                    }
                    else
                    {
                        set_uint(t_last, coeff_count, t_ntt);
                    }

                    // Lazy substraction, results in [0, 2*qi), since fix is in [0, qi].
                    uint64_t fix = qi - barrett_reduce_64(qk_half, get<1>(J));
                    SEAL_ITERATE(t_ntt, coeff_count, [fix](auto &K) { K += fix; });

                    uint64_t qi_lazy = qi << 1; // some multiples of qi
                    if (scheme == scheme_type::ckks)
                    {
                        // This ntt_negacyclic_harvey_lazy results in [0, 4*qi).
                        ntt_negacyclic_harvey_lazy(t_ntt, get<2>(J));
#if SEAL_USER_MOD_BIT_COUNT_MAX > 60
                        // Reduce from [0, 4qi) to [0, 2qi)
                        SEAL_ITERATE(
                            t_ntt, coeff_count, [&](auto &K) { K -= SEAL_COND_SELECT(K >= qi_lazy, qi_lazy, 0); });
#else
                        // Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
                        qi_lazy = qi << 2;
#endif
                    }
                    else if (scheme == scheme_type::bfv)
                    {
                        inverse_ntt_negacyclic_harvey_lazy(get<0, 1>(J), get<2>(J));
                    }

                    // ((ct mod qi) - (ct mod qk)) mod qi with output in [0, 2 * qi_lazy)
                    SEAL_ITERATE(
                        iter(get<0, 1>(J), t_ntt), coeff_count, [&](auto K) { get<0>(K) += qi_lazy - get<1>(K); });

                    // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                    multiply_poly_scalar_coeffmod(get<0, 1>(J), coeff_count, get<3>(J), get<1>(J), get<0, 1>(J));
                    add_poly_coeffmod(get<0, 1>(J), get<0, 0>(J), coeff_count, get<1>(J), get<0, 0>(J));
                });
            }
        });
    }

    void Evaluator::switch_key_inplace_cuda(
        Ciphertext &encrypted, uint64_t *target_iter, const KSwitchKeys &kswitch_keys, size_t kswitch_keys_index,
        MemoryPoolHandle pool) const
    {
        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto key_parms_id = key_parms.parms_id();
        auto scheme = parms.scheme();

        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        // Don't validate all of kswitch_keys but just check the parms_id.
        if (kswitch_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }

        if (kswitch_keys_index >= kswitch_keys.data().size())
        {
            throw out_of_range("kswitch_keys_index");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (scheme == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();

        uint64_t *d_key_modulu_value = key_parms.d_coeff_modulus_value();
        uint64_t *d_key_ratio_0 = key_parms.d_coeff_modulus_ratio_0();
        uint64_t *d_key_ratio_1 = key_parms.d_coeff_modulus_ratio_1();

        size_t key_modulus_size = key_modulus.size();
        size_t key_coeff_count = key_parms.poly_modulus_degree();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        auto key_ntt_table_plain = key_context_data.small_ntt_tables();
        auto modswitch_factors = key_context_data.rns_tool()->inv_q_last_mod_q();
        uint64_t *d_root_matrix = key_context_data.d_root_matrix();
        uint64_t *d_root_power = key_context_data.d_root_powers();


        // Size check
        if (!product_fits_in(coeff_count, rns_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare input
        auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].data().size();

        // Check only the used component in KSwitchKeys.
        for (auto &each_key : key_vector)
        {
            if (!is_metadata_valid_for(each_key, context_) || !is_buffer_valid(each_key))
            {
                throw invalid_argument("kswitch_keys is not valid for encryption parameters");
            }
        }

        allocate_gpu<uint64_t>(&d_t_target_, coeff_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(&d_t_operand_, coeff_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(&d_t_poly_lazy_, coeff_count * 2 * key_component_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(
            &d_t_poly_prod_iter_, coeff_count * rns_modulus_size * key_component_count);

        uint64_t *d_target_iter = target_iter; // 输入数据
        uint64_t *d_t_target = d_t_target_; // copy了一份target_iter
        uint64_t *d_t_operand = d_t_operand_;
        uint64_t *d_t_poly_lazy = d_t_poly_lazy_; // d_t_ntt复用这一块内存
        uint64_t *d_t_poly_prod_iter = d_t_poly_prod_iter_; // ntt时复用这一块内存

        checkCudaErrors(cudaMemcpy(
            d_t_target, target_iter, coeff_count * decomp_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        uint64_t *d_inv_root_powers = key_context_data.d_inv_root_powers();
       
        uint64_t temp_mu;
        k_uint128_t mu1;
        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();

        for (int i = 0; i < decomp_modulus_size; ++i)
        {
            mu1 = k_uint128_t::exp2(key_modulus[i].bit_count() * 2);
            temp_mu = (mu1 / key_modulus[i].value()).low;
            inverseNTT(
                d_t_target + i * coeff_count, coeff_count, ntt_steam[i % stream_num], key_modulus[i].value(), temp_mu,
                key_modulus[i].bit_count(), d_inv_root_powers + i * coeff_count);
        }

        uint64_t** d_key_ptr_array = new uint64_t*[decomp_modulus_size];
        for (int j = 0; j < decomp_modulus_size; ++j)
        {
            d_key_ptr_array[j] = key_vector[j].data().d_data() ;
        }
        uint64_t** d_ptr_array = nullptr;
        // allocate_gpu<uint64_t *>(&d_ptr_array, decomp_modulus_size);
        checkCudaErrors(cudaMalloc((void**)&d_ptr_array, decomp_modulus_size * sizeof(uint64_t*)));

        checkCudaErrors(cudaMemcpy(d_ptr_array, d_key_ptr_array, decomp_modulus_size * sizeof(uint64_t*), cudaMemcpyHostToDevice));


        for (int i = 0; i < rns_modulus_size; ++i)
        {
            size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);
            uint64_t ratio_0 = key_modulus[key_index].const_ratio().data()[0];
            uint64_t ratio_1 = key_modulus[key_index].const_ratio().data()[1];
            uint64_t modulus = key_modulus[key_index].value();

            // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
            size_t lazy_reduction_summand_bound = size_t(SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX);
            size_t lazy_reduction_counter = lazy_reduction_summand_bound;

            // Allocate memory for a lazy accumulator (128-bit coefficients)
            checkCudaErrors(cudaMemset(d_t_poly_lazy, 0, coeff_count * 2 * key_component_count * sizeof(uint64_t)));

            // Multiply with keys and perform lazy reduction on product's coefficients

#if NTT_VERSION == 3
            switch_key_helper3_batch<<<(coeff_count * decomp_modulus_size + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_target, // d_t_target中数据要反复使用
                coeff_count, decomp_modulus_size, modulus, ratio_1, d_t_operand, d_key_modulu_value, key_modulus[key_index].value());

            ntt_v3_key_switch(context_, key_parms_id, d_t_operand, decomp_modulus_size,ntt_steam[0], key_index);

#else
            switch_key_helper3_batch<<<(coeff_count * decomp_modulus_size + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_target, // d_t_target中数据要反复使用
                coeff_count, decomp_modulus_size, modulus, ratio_1, d_t_operand, d_key_modulu_value, key_modulus[key_index].value());


            cudaDeviceSynchronize();
            ntt_v1(context_, key_parms_id, d_t_operand, decomp_modulus_size, key_index, true);
            cudaDeviceSynchronize();
#endif

            lazy_reduction_counter_kernel4<<<(key_coeff_count * key_component_count + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_operand,
                d_ptr_array,
                d_t_poly_lazy,
                key_coeff_count,
                decomp_modulus_size,
                modulus, 
                ratio_0, 
                ratio_1,
                lazy_reduction_counter,
                lazy_reduction_summand_bound,
                key_component_count,
                key_modulus_size, 
                key_index);
            
            for (int j = 0; j < decomp_modulus_size; ++j)
            {               
                if (!--lazy_reduction_counter)
                {
                    lazy_reduction_counter = lazy_reduction_summand_bound;
                }                
            }
            lazy_reduction_counter_kernel<<<(coeff_count * key_component_count + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_poly_lazy, d_t_poly_prod_iter + (i * coeff_count), coeff_count, modulus, ratio_0, ratio_1,
                key_component_count, rns_modulus_size, lazy_reduction_counter == lazy_reduction_summand_bound);
        }
        // Accumulated products are now stored in t_poly_prod

        // Perform modulus switching with scaling
        uint64_t *d_encrypted = encrypted.d_data();
        for (int i = 0; i < key_component_count; ++i)
        {
            // Lazy reduction; this needs to be then reduced mod qi
            uint64_t *d_t_last =
                d_t_poly_prod_iter + coeff_count * rns_modulus_size * i + coeff_count * decomp_modulus_size;

            uint64_t *d_inv_root_powers = key_context_data.d_inv_root_powers() + coeff_count * (key_modulus_size - 1);
            // cudaStream_t ntt = 0;
            mu1 = k_uint128_t::exp2(key_modulus[key_modulus_size - 1].bit_count() * 2);
            temp_mu = (mu1 / key_modulus[key_modulus_size - 1].value()).low;
            inverseNTT(
                d_t_last, coeff_count, ntt_steam[0], key_modulus[key_modulus_size - 1].value(), temp_mu,
                key_modulus[key_modulus_size - 1].bit_count(), d_inv_root_powers);

            // Add (p-1)/2 to change from flooring to rounding.
            uint64_t qk = key_modulus[key_modulus_size - 1].value();
            uint64_t qk_half = qk >> 1;
            barrett_reduce_64_helper<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_last, key_modulus[key_modulus_size - 1].value(),
                key_modulus[key_modulus_size - 1].const_ratio().data()[1], qk_half, d_t_last, coeff_count);


            switch_key_helper1_batch<<<(coeff_count * decomp_modulus_size + 255) / 256, 256, 0, ntt_steam[0]>>>(d_t_last, coeff_count, decomp_modulus_size,
                                                                                         d_key_modulu_value, d_key_ratio_1, qk, d_t_poly_lazy);
            
#if NTT_VERSION == 3
            ntt_v3(context_, key_parms_id, d_t_poly_lazy, decomp_modulus_size, 0, ntt_steam[0]);
            for (int j = 0; j < decomp_modulus_size; ++j)
            {
                switch_key_helper2<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[j % stream_num]>>>(
                    d_t_poly_lazy + j * coeff_count, d_t_poly_prod_iter + coeff_count * (rns_modulus_size * i + j), coeff_count,
                    modswitch_factors[j].operand, // scalar temp_scalar.operand, temp_scalar.quotient
                    modswitch_factors[j].quotient, key_modulus[j].value(),
                    d_encrypted + coeff_count * (decomp_modulus_size * i + j));
                

            }
#else
            cudaDeviceSynchronize();
            // ntt_v1(context_, key_parms_id, d_t_poly_lazy, decomp_modulus_size, 0);
            for (int j = 0; j < decomp_modulus_size; ++j)
            {
                ntt_v1_single(context_, key_parms_id, d_t_poly_lazy+ j * coeff_count, j, ntt_steam[j % stream_num]);

                switch_key_helper2<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[j % stream_num]>>>(
                    d_t_poly_lazy + j * coeff_count, d_t_poly_prod_iter + coeff_count * (rns_modulus_size * i + j), coeff_count,
                    modswitch_factors[j].operand, // scalar temp_scalar.operand, temp_scalar.quotient
                    modswitch_factors[j].quotient, key_modulus[j].value(),
                    d_encrypted + coeff_count * (decomp_modulus_size * i + j));
                

            }

#endif
        }
    
        deallocate_gpu<uint64_t>(&d_t_target_, coeff_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(&d_t_operand_, coeff_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(&d_t_poly_lazy_, coeff_count * 2 * key_component_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(
            &d_t_poly_prod_iter_, coeff_count * rns_modulus_size * key_component_count);
        deallocate_gpu<uint64_t *>(&d_ptr_array, decomp_modulus_size);

    }

    void Evaluator::switch_key_inplace_bgv(
        Ciphertext &encrypted, uint64_t *target_iter, const KSwitchKeys &kswitch_keys, size_t kswitch_keys_index,
        MemoryPoolHandle pool) const
    {
        auto parms_id = encrypted.parms_id();
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto key_parms_id = key_parms.parms_id();
        auto scheme = parms.scheme();

        // Verify parameters.
        if (!is_metadata_valid_for(encrypted, context_) || !is_buffer_valid(encrypted))
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }

        // Don't validate all of kswitch_keys but just check the parms_id.
        if (kswitch_keys.parms_id() != context_.key_parms_id())
        {
            throw invalid_argument("parameter mismatch");
        }

        if (kswitch_keys_index >= kswitch_keys.data().size())
        {
            throw out_of_range("kswitch_keys_index");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (scheme == scheme_type::bfv && encrypted.is_ntt_form())
        {
            throw invalid_argument("BFV encrypted cannot be in NTT form");
        }
        if (scheme == scheme_type::ckks && !encrypted.is_ntt_form())
        {
            throw invalid_argument("CKKS encrypted must be in NTT form");
        }
        if (scheme == scheme_type::bgv && !encrypted.is_ntt_form())
        {
            throw invalid_argument("BGV encrypted must be in NTT form");
        }

        // Extract encryption parameters.
        size_t coeff_count = parms.poly_modulus_degree();
        size_t decomp_modulus_size = parms.coeff_modulus().size();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();
        size_t key_coeff_count = key_parms.poly_modulus_degree();
        size_t rns_modulus_size = decomp_modulus_size + 1;
        auto key_ntt_table_plain = key_context_data.small_ntt_tables();
        auto modswitch_factors = key_context_data.rns_tool()->inv_q_last_mod_q();
        uint64_t *d_root_matrix = key_context_data.d_root_matrix();
        auto &first_context_data = *context_.first_context_data();
        // uint64_t *d_root_matrix = first_context_data.d_root_matrix();


        auto key_ntt_tables = iter(key_context_data.small_ntt_tables());

        uint64_t *key_modulu_value = key_parms.d_coeff_modulus_value();
        uint64_t *key_modulu_ratio0 = key_parms.d_coeff_modulus_ratio_0();
        uint64_t *key_modulu_ratio1 = key_parms.d_coeff_modulus_ratio_1();
        int *key_bit_count = key_context_data.d_bit_count();

        uint64_t *modswitch_factor_operand = key_context_data.rns_tool()->d_inv_q_last_mod_q_operand();
        uint64_t *modswitch_factor_quotient = key_context_data.rns_tool()->d_inv_q_last_mod_q_quotient();

        // Size check
        if (!product_fits_in(coeff_count, rns_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        // Prepare input
        auto &key_vector = kswitch_keys.data()[kswitch_keys_index];
        size_t key_component_count = key_vector[0].data().size();

        // Check only the used component in KSwitchKeys.
        for (auto &each_key : key_vector)
        {
            if (!is_metadata_valid_for(each_key, context_) || !is_buffer_valid(each_key))
            {
                throw invalid_argument("kswitch_keys is not valid for encryption parameters");
            }
        }

        allocate_gpu<uint64_t>(&d_t_target_, coeff_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(&d_t_operand_, coeff_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(&d_t_poly_lazy_, coeff_count * 2 * key_component_count * decomp_modulus_size);
        allocate_gpu<uint64_t>(
            &d_t_poly_prod_iter_, coeff_count * rns_modulus_size * key_component_count);


        uint64_t *d_target_iter = target_iter; // 输入数据
        uint64_t *d_t_target = d_t_target_; // copy了一份target_iter
        uint64_t *d_t_operand = d_t_operand_;
        uint64_t *d_t_poly_lazy = d_t_poly_lazy_; // d_t_ntt复用这一块内存
        uint64_t *d_t_poly_prod_iter = d_t_poly_prod_iter_; // ntt时复用这一块内存

        checkCudaErrors(cudaMemcpy(
            d_t_target, target_iter, coeff_count * decomp_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

        uint64_t *d_inv_root_powers = key_context_data.d_inv_root_powers();
        cudaStream_t ntt = 0;

        // 待检查后使用
        // context_.ensure_ntt_size(coeff_count*decomp_modulus_size);
        // uint64_t *ntt_temp = context_.ntt_temp();

        uint64_t temp_mu;
        k_uint128_t mu1;

        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();
        for (int i = 0; i < decomp_modulus_size; ++i)
        {
            mu1 = k_uint128_t::exp2(key_modulus[i].bit_count() * 2);
            temp_mu = (mu1 / key_modulus[i].value()).low;
            inverseNTT(
                d_t_target + i * coeff_count, coeff_count, ntt_steam[i % stream_num], key_modulus[i].value(), temp_mu,
                key_modulus[i].bit_count(), d_inv_root_powers + i * coeff_count);
        }



        uint64_t** d_key_ptr_array = new uint64_t*[decomp_modulus_size];
        for (int j = 0; j < decomp_modulus_size; ++j)
        {
            d_key_ptr_array[j] = key_vector[j].data().d_data() ;
        }
        uint64_t** d_ptr_array = nullptr;
        allocate_gpu<uint64_t *>(&d_ptr_array, decomp_modulus_size);
        checkCudaErrors(cudaMemcpy(d_ptr_array, d_key_ptr_array, decomp_modulus_size * sizeof(uint64_t*), cudaMemcpyHostToDevice));

        for (int i = 0; i < rns_modulus_size; ++i)
        {
            size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);
            uint64_t ratio_0 = key_modulus[key_index].const_ratio().data()[0];
            uint64_t ratio_1 = key_modulus[key_index].const_ratio().data()[1];
            uint64_t modulus = key_modulus[key_index].value();

            // Product of two numbers is up to 60 + 60 = 120 bits, so we can sum up to 256 of them without reduction.
            size_t lazy_reduction_summand_bound = size_t(SEAL_MULTIPLY_ACCUMULATE_USER_MOD_MAX);
            size_t lazy_reduction_counter = lazy_reduction_summand_bound;

            // Allocate memory for a lazy accumulator (128-bit coefficients)
            checkCudaErrors(cudaMemset(d_t_poly_lazy, 0, coeff_count * 2 * key_component_count * sizeof(uint64_t)));

            // Multiply with keys and perform lazy reduction on product's coefficients

#if NTT_VERSION == 3
            switch_key_helper3_batch<<<(coeff_count * decomp_modulus_size + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_target, // d_t_target中数据要反复使用
                coeff_count, decomp_modulus_size, modulus, ratio_1, d_t_operand, key_modulu_value, key_modulus[key_index].value());

            ntt_v3_key_switch(context_, key_parms_id, d_t_operand, decomp_modulus_size,ntt_steam[0], key_index);

#else
            switch_key_helper3_batch<<<(coeff_count * decomp_modulus_size + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_target, // d_t_target中数据要反复使用
                coeff_count, decomp_modulus_size, modulus, ratio_1, d_t_operand, key_modulu_value, key_modulus[key_index].value());


            cudaDeviceSynchronize();
            ntt_v1(context_, key_parms_id, d_t_operand, decomp_modulus_size, key_index, true);
            cudaDeviceSynchronize();
#endif

            lazy_reduction_counter_kernel4<<<(key_coeff_count * key_component_count + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_operand,
                d_ptr_array,
                d_t_poly_lazy,
                key_coeff_count,
                decomp_modulus_size,
                modulus, 
                ratio_0, 
                ratio_1,
                lazy_reduction_counter,
                lazy_reduction_summand_bound,
                key_component_count,
                key_modulus_size, 
                key_index);
            
            for (int j = 0; j < decomp_modulus_size; ++j)
            {               
                if (!--lazy_reduction_counter)
                {
                    lazy_reduction_counter = lazy_reduction_summand_bound;
                }                
            }
            lazy_reduction_counter_kernel<<<(coeff_count * key_component_count + 255) / 256, 256, 0, ntt_steam[0]>>>(
                d_t_poly_lazy, d_t_poly_prod_iter + (i * coeff_count), coeff_count, modulus, ratio_0, ratio_1,
                key_component_count, rns_modulus_size, lazy_reduction_counter == lazy_reduction_summand_bound);
        }

        // Accumulated products are now stored in t_poly_prod

        // Perform modulus switching with scaling
        uint64_t *d_encrypted = encrypted.d_data();

        for (int i = 0; i < key_component_count; ++i)
        {
            // Lazy reduction; this needs to be then reduced mod qi
            uint64_t *d_t_last =
                d_t_poly_prod_iter + coeff_count * rns_modulus_size * i + coeff_count * decomp_modulus_size;

            uint64_t *d_inv_root_powers = key_context_data.d_inv_root_powers() + coeff_count * (key_modulus_size - 1);
            cudaStream_t ntt = 0;
            k_uint128_t mu1 = k_uint128_t::exp2(key_modulus[key_modulus_size - 1].bit_count() * 2);
            uint64_t temp_mu = (mu1 / key_modulus[key_modulus_size - 1].value()).low;
            inverseNTT(
                d_t_last, coeff_count, ntt, key_modulus[key_modulus_size - 1].value(), temp_mu,
                key_modulus[key_modulus_size - 1].bit_count(), d_inv_root_powers);

            const Modulus &plain_modulus = parms.plain_modulus();

            // Add (p-1)/2 to change from flooring to rounding.
            uint64_t qk = key_modulus[key_modulus_size - 1].value();
            uint64_t qk_inv_qp = context_.key_context_data()->rns_tool()->inv_q_last_mod_t();

            bgv_switch_key_helper1<<<(coeff_count * decomp_modulus_size + 255) / 256, 256>>>(
                d_t_last, d_t_poly_lazy, coeff_count, decomp_modulus_size, plain_modulus.value(), plain_modulus.const_ratio().data()[1],
                qk_inv_qp, qk, key_modulu_value, key_modulu_ratio1);
#if NTT_VERSION == 3
            ntt_v3(context_, key_parms_id, d_t_poly_lazy, decomp_modulus_size);
#else
            ntt_v1(context_, key_parms_id, d_t_poly_lazy, decomp_modulus_size, 0);
#endif
            bgv_switch_key_helper2<<<(coeff_count * decomp_modulus_size + 255) / 256, 256>>>(
                d_t_poly_lazy, 
                d_t_poly_prod_iter + i * coeff_count * rns_modulus_size, 
                d_encrypted + coeff_count * decomp_modulus_size * i, 
                coeff_count, decomp_modulus_size, rns_modulus_size,
                key_modulu_value, key_modulu_ratio0, key_modulu_ratio1, 
                modswitch_factor_operand, modswitch_factor_quotient);
        }

        deallocate_gpu<uint64_t>(&d_t_target_, coeff_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(&d_t_operand_, coeff_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(&d_t_poly_lazy_, coeff_count * 2 * key_component_count * decomp_modulus_size);
        deallocate_gpu<uint64_t>(
            &d_t_poly_prod_iter_, coeff_count * rns_modulus_size * key_component_count);
        deallocate_gpu<uint64_t *>(&d_ptr_array, decomp_modulus_size);


    }

} // namespace seal
