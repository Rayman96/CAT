// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/util/pointer.h"
#include "seal/util/uintarith.cuh"
#include "seal/util/uintcore.h"
#include <cstdint>

namespace seal
{
    namespace util
    {
                inline void increment_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count, std::uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!modulus)
            {
                throw std::invalid_argument("modulus");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (is_greater_than_or_equal_uint(operand, modulus, uint64_count))
            {
                throw std::invalid_argument("operand");
            }
            if (modulus == result)
            {
                throw std::invalid_argument("result cannot point to the same value as modulus");
            }
#endif
            unsigned char carry = increment_uint(operand, uint64_count, result);
            if (carry || is_greater_than_or_equal_uint(result, modulus, uint64_count))
            {
                sub_uint(result, modulus, uint64_count, result);
            }
        }

        inline void decrement_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count, std::uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!modulus)
            {
                throw std::invalid_argument("modulus");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (is_greater_than_or_equal_uint(operand, modulus, uint64_count))
            {
                throw std::invalid_argument("operand");
            }
            if (modulus == result)
            {
                throw std::invalid_argument("result cannot point to the same value as modulus");
            }
#endif
            if (decrement_uint(operand, uint64_count, result))
            {
                add_uint(result, modulus, uint64_count, result);
            }
        }

        inline void negate_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count, std::uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!modulus)
            {
                throw std::invalid_argument("modulus");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (is_greater_than_or_equal_uint(operand, modulus, uint64_count))
            {
                throw std::invalid_argument("operand");
            }
#endif
            if (is_zero_uint(operand, uint64_count))
            {
                // Negation of zero is zero.
                set_zero_uint(uint64_count, result);
            }
            else
            {
                // Otherwise, we know operand > 0 and < modulus so subtract modulus - operand.
                sub_uint(modulus, operand, uint64_count, result);
            }
        }

        inline void div2_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count, std::uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!modulus)
            {
                throw std::invalid_argument("modulus");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (!is_bit_set_uint(modulus, uint64_count, 0))
            {
                throw std::invalid_argument("modulus");
            }
            if (is_greater_than_or_equal_uint(operand, modulus, uint64_count))
            {
                throw std::invalid_argument("operand");
            }
#endif
            if (*operand & 1)
            {
                unsigned char carry = add_uint(operand, modulus, uint64_count, result);
                right_shift_uint(result, 1, uint64_count, result);
                if (carry)
                {
                    set_bit_uint(result, uint64_count, static_cast<int>(uint64_count) * bits_per_uint64 - 1);
                }
            }
            else
            {
                right_shift_uint(operand, 1, uint64_count, result);
            }
        }

        void add_uint_uint_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *modulus,
            std::size_t uint64_count, std::uint64_t *result);

        void add_uint_uint_mod_cuda(
            const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *modulus,
            std::size_t uint64_count, std::uint64_t *result);

        inline void sub_uint_uint_mod(
            const std::uint64_t *operand1, const std::uint64_t *operand2, const std::uint64_t *modulus,
            std::size_t uint64_count, std::uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw std::invalid_argument("operand2");
            }
            if (!modulus)
            {
                throw std::invalid_argument("modulus");
            }
            if (!uint64_count)
            {
                throw std::invalid_argument("uint64_count");
            }
            if (!result)
            {
                throw std::invalid_argument("result");
            }
            if (is_greater_than_or_equal_uint(operand1, modulus, uint64_count))
            {
                throw std::invalid_argument("operand1");
            }
            if (is_greater_than_or_equal_uint(operand2, modulus, uint64_count))
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus == result)
            {
                throw std::invalid_argument("result cannot point to the same value as modulus");
            }
#endif
            if (sub_uint(operand1, operand2, uint64_count, result))
            {
                add_uint(result, modulus, uint64_count, result);
            }
        }

        bool try_invert_uint_mod(
            const std::uint64_t *operand, const std::uint64_t *modulus, std::size_t uint64_count, std::uint64_t *result,
            MemoryPool &pool);
        
         __device__ __forceinline__ unsigned char add_uint_kernel(
            const uint64_t operand1, const uint64_t operand2, uint64_t *result)
        {
            *result = operand1 + operand2;
            return static_cast<unsigned char>(*result < operand1);
        }

        __device__ __forceinline__ unsigned char add_uint_kernel(
            uint64_t operand1, uint64_t operand2, unsigned char carry, unsigned long long *result)
        {
            operand1 += operand2;
            *result = operand1 + carry;
            return (operand1 < operand2) || (~operand1 < carry);
            
        }

        __device__ __forceinline__ bool is_greater_than_or_equal_uint_kernel(
            const uint64_t *operand1, const uint64_t *operand2, size_t uint64_count)
        {
            int result = 0;
            operand1 += uint64_count - 1;
            operand2 += uint64_count - 1;

            for (; (result == 0) && uint64_count--; operand1--, operand2--)
            {
                result = (*operand1 > *operand2) - (*operand1 < *operand2);
            }
            return result>=0;
        }

        __device__ __forceinline__ unsigned char sub_uint64_kernel(
            const uint64_t operand1, const uint64_t operand2, uint64_t *result)
        {
            *result = operand1 - operand2;
            return static_cast<unsigned char>(operand2 > operand1);
        }
        __device__ __forceinline__ unsigned char sub_uint64_kernel(
            const uint64_t operand1, const uint64_t operand2, unsigned long long *result)
        {
            *result = operand1 - operand2;
            return static_cast<unsigned char>(operand2 > operand1);
        }

        __device__ __forceinline__ unsigned char sub_uint64_kernel(
            const uint64_t operand1, const uint64_t operand2, unsigned char borrow, unsigned long long *result)
        {
             auto diff = operand1 - operand2;
            *result = diff - (borrow != 0);
            return (diff > operand1) || (diff < borrow);
        }

        __device__ __forceinline__ unsigned char sub_uint_kernel(
            const uint64_t *operand1, const uint64_t *operand2, std::size_t uint64_count, uint64_t *result)
        {
            unsigned char borrow = sub_uint64_kernel(*operand1++, *operand2++, result++);

            // Do the rest
            for (; --uint64_count; operand1++, operand2++, result++)
            {
                unsigned long long temp_result;
                borrow = sub_uint64_kernel(*operand1, *operand2, borrow, &temp_result);
                *result = temp_result;
            }
            return borrow;
        }

        __global__ __forceinline__ void add_uint_uint_mod_kernel(
            const uint64_t *operand1, const uint64_t *operand2, const uint64_t *modulus, size_t uint64_count,
            uint64_t *result)        
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
    } // namespace util
} // namespace seal
