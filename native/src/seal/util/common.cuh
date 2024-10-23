#pragma once
#include "helper.cuh"
#include "ntt_60bit.cuh"
#include "ntt_helper.cuh"

namespace seal
{
    namespace util
    {
        template <typename T, typename = std::enable_if_t<is_uint32_v<T> || is_uint64_v<T>>>
        __device__ inline constexpr T reverse_bits_kernel(T operand) noexcept
        {
            T temp_result;
            if (sizeof(T) == sizeof(std::uint32_t))
            {
                operand = (((operand & T(0xaaaaaaaa)) >> 1) | ((operand & T(0x55555555)) << 1));
                operand = (((operand & T(0xcccccccc)) >> 2) | ((operand & T(0x33333333)) << 2));
                operand = (((operand & T(0xf0f0f0f0)) >> 4) | ((operand & T(0x0f0f0f0f)) << 4));
                operand = (((operand & T(0xff00ff00)) >> 8) | ((operand & T(0x00ff00ff)) << 8));
                return static_cast<T>(operand >> 16) | static_cast<T>(operand << 16);
            }
            else if (sizeof(T) == sizeof(std::uint64_t))
            {
                return static_cast<T>(reverse_bits_kernel(static_cast<std::uint32_t>(operand >> 32))) |
                       (static_cast<T>(reverse_bits_kernel(static_cast<std::uint32_t>(operand & T(0xFFFFFFFF))))
                        << 32);
            }
        }

        template <typename T, typename = std::enable_if_t<is_uint32_v<T> || is_uint64_v<T>>>
        __device__ inline T reverse_bits_kernel(T operand, int bit_count)
        {
            int bits_per_byte = 8;
            return (bit_count == 0)
                       ? T(0)
                       : reverse_bits_kernel(operand) >> (sizeof(T) * static_cast<std::size_t>(bits_per_byte) -
                                                           static_cast<std::size_t>(bit_count));
        }
    }
}