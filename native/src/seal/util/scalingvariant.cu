// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/encryptor.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.cuh"

using namespace std;

namespace seal
{
    namespace util
    {
 
        void add_plain_without_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination)
        {
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            const size_t plain_coeff_count = plain.coeff_count();
            const size_t coeff_modulus_size = coeff_modulus.size();
#ifdef SEAL_DEBUG
            if (plain_coeff_count > parms.poly_modulus_degree())
            {
                throw std::invalid_argument("invalid plaintext");
            }
            if (destination.poly_modulus_degree() != parms.poly_modulus_degree())
            {
                throw std::invalid_argument("destination is not valid for encryption parameters");
            }
#endif
            SEAL_ITERATE(iter(destination, coeff_modulus), coeff_modulus_size, [&](auto I) {
                std::transform(
                    plain.data(), plain.data() + plain_coeff_count, get<0>(I), get<0>(I),
                    [&](uint64_t m, uint64_t c) -> uint64_t {
                        m = barrett_reduce_64(m, get<1>(I));
                        return add_uint_mod(c, m, get<1>(I));
                    });
            });
        }

        void sub_plain_without_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination)
        {
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();

            const size_t plain_coeff_count = plain.coeff_count();
            const size_t coeff_modulus_size = coeff_modulus.size();
#ifdef SEAL_DEBUG
            if (plain_coeff_count > parms.poly_modulus_degree())
            {
                throw std::invalid_argument("invalid plaintext");
            }
            if (destination.poly_modulus_degree() != parms.poly_modulus_degree())
            {
                throw std::invalid_argument("destination is not valid for encryption parameters");
            }
#endif
            SEAL_ITERATE(iter(destination, coeff_modulus), coeff_modulus_size, [&](auto I) {
                std::transform(
                    plain.data(), plain.data() + plain_coeff_count, get<0>(I), get<0>(I),
                    [&](uint64_t m, uint64_t c) -> uint64_t {
                        m = barrett_reduce_64(m, get<1>(I));
                        return sub_uint_mod(c, m, get<1>(I));
                    });
            });
        }

        void multiply_add_plain_with_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination)
        {
            auto &parms = context_data.parms();
            size_t plain_coeff_count = plain.coeff_count();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            auto plain_modulus = context_data.parms().plain_modulus();
            auto coeff_div_plain_modulus = context_data.coeff_div_plain_modulus();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            uint64_t q_mod_t = context_data.coeff_modulus_mod_plain_modulus();
#ifdef SEAL_DEBUG
            if (plain_coeff_count > parms.poly_modulus_degree())
            {
                throw std::invalid_argument("invalid plaintext");
            }
            if (destination.poly_modulus_degree() != parms.poly_modulus_degree())
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
#endif
            // Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
            // and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
            // floor((q * m + floor((t+1) / 2)) / t).
            SEAL_ITERATE(iter(plain.data(), size_t(0)), plain_coeff_count, [&](auto I) {
                // Compute numerator = (q mod t) * m[i] + (t+1)/2
                unsigned long long prod[2]{ 0, 0 };
                uint64_t numerator[2]{ 0, 0 };
                multiply_uint64(get<0>(I), q_mod_t, prod);
                unsigned char carry = add_uint64(*prod, plain_upper_half_threshold, numerator);
                numerator[1] = static_cast<uint64_t>(prod[1]) + static_cast<uint64_t>(carry);

                // Compute fix[0] = floor(numerator / t)
                uint64_t fix[2] = { 0, 0 };
                divide_uint128_inplace(numerator, plain_modulus.value(), fix);

                // Add to ciphertext: floor(q / t) * m + increment
                size_t coeff_index = get<1>(I);
                SEAL_ITERATE(
                    iter(destination, coeff_modulus, coeff_div_plain_modulus), coeff_modulus_size, [&](auto J) {
                        uint64_t scaled_rounded_coeff = multiply_add_uint_mod(get<0>(I), get<2>(J), fix[0], get<1>(J));
                        get<0>(J)[coeff_index] = add_uint_mod(get<0>(J)[coeff_index], scaled_rounded_coeff, get<1>(J));
                    });
            });
        }

        __global__ void multiply_add_plain_with_scaling_variant_kernel(
            uint64_t *plain, uint64_t *destination, size_t coeff_count, size_t encrypted_coeff_count,
            size_t coeff_modulus_size, uint64_t q_mod_t,
            uint64_t plain_upper_half_threshold, uint64_t *modulu_value, uint64_t *ratio, uint64_t *coeff_div_plain_modulus_operand,
            uint64_t *coeff_div_plain_modulus_quoient, uint64_t plain_modulu_value)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size)
            {
                size_t coeff_idx = idx % coeff_count;
                int modulus_idx = idx / coeff_count;
                unsigned long long prod[2]{ 0, 0 };
                uint64_t numerator[2]{ 0, 0 };
                multiply_uint64_kernel(plain[coeff_idx], q_mod_t, prod);
                unsigned char carry = add_uint64(*prod, plain_upper_half_threshold, numerator);
                numerator[1] = static_cast<uint64_t>(prod[1]) + static_cast<uint64_t>(carry);
                // Compute fix[0] = floor(numerator / t)
                uint64_t fix[2] = { 0, 0 };

                divide_uint128_inplace_kernel(numerator, plain_modulu_value, fix);

                uint64_t scaled_rounded_coeff = multiply_add_uint_mod_kernel(
                    plain[coeff_idx], coeff_div_plain_modulus_operand[modulus_idx],
                    coeff_div_plain_modulus_quoient[modulus_idx], fix[0], modulu_value[modulus_idx], ratio[modulus_idx], modulus_idx);

                destination[modulus_idx * encrypted_coeff_count + coeff_idx] =
                    add_uint_mod_kernel(destination[modulus_idx * encrypted_coeff_count + coeff_idx], scaled_rounded_coeff, modulu_value[modulus_idx]);

                idx += blockDim.x * gridDim.x;

            }
        }

        __global__ void multiply_sub_plain_with_scaling_variant_kernel(
            uint64_t *plain, uint64_t *destination, size_t coeff_count, size_t encrypted_coeff_count,
            size_t coeff_modulus_size, uint64_t q_mod_t,
            uint64_t plain_upper_half_threshold, uint64_t *modulu_value, uint64_t *ratio, uint64_t *coeff_div_plain_modulus_operand,
            uint64_t *coeff_div_plain_modulus_quoient, uint64_t plain_modulu_value)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size)
            {
                size_t coeff_idx = idx % coeff_count;
                int modulus_idx = idx / coeff_count;
                unsigned long long prod[2]{ 0, 0 };
                uint64_t numerator[2]{ 0, 0 };
                multiply_uint64_kernel(plain[coeff_idx], q_mod_t, prod);
                unsigned char carry = add_uint64(*prod, plain_upper_half_threshold, numerator);
                numerator[1] = static_cast<uint64_t>(prod[1]) + static_cast<uint64_t>(carry);
                // Compute fix[0] = floor(numerator / t)
                uint64_t fix[2] = { 0, 0 };

                divide_uint128_inplace_kernel(numerator, plain_modulu_value, fix);

                uint64_t scaled_rounded_coeff = multiply_add_uint_mod_kernel(
                    plain[coeff_idx], coeff_div_plain_modulus_operand[modulus_idx],
                    coeff_div_plain_modulus_quoient[modulus_idx], fix[0], modulu_value[modulus_idx], ratio[modulus_idx], modulus_idx);

                destination[modulus_idx * encrypted_coeff_count + coeff_idx] =
                    sub_uint_mod_kernel(destination[modulus_idx * encrypted_coeff_count + coeff_idx], scaled_rounded_coeff, modulu_value[modulus_idx]);

                idx += blockDim.x * gridDim.x;

            }
        }

        void multiply_add_plain_with_scaling_variant_cuda(
            const Plaintext &plain, const SEALContext::ContextData &context_data, uint64_t *destination)
        {
            auto &parms = context_data.parms();
            size_t plain_coeff_count = plain.coeff_count();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            uint64_t *d_plain = plain.d_data();

            auto plain_modulus = context_data.parms().plain_modulus();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            uint64_t q_mod_t = context_data.coeff_modulus_mod_plain_modulus();
            uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
            uint64_t *d_ratio = parms.d_coeff_modulus_ratio_1();

            uint64_t *d_coeff_div_plain_modulus_operand = context_data.d_coeff_div_plain_modulus_operand();
            uint64_t *d_coeff_div_plain_modulus_quoient = context_data.d_coeff_div_plain_modulus_quoitent();

            multiply_add_plain_with_scaling_variant_kernel<<<(plain_coeff_count * coeff_modulus_size + 255) / 256, 256>>>(
                d_plain, 
                destination, 
                plain_coeff_count, 
                coeff_count,
                coeff_modulus_size, 
                q_mod_t, 
                plain_upper_half_threshold,
                d_coeff_modulus_value, 
                d_ratio,
                d_coeff_div_plain_modulus_operand, 
                d_coeff_div_plain_modulus_quoient, 
                plain_modulus.value());
        }

        void multiply_sub_plain_with_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination)
        {
            auto &parms = context_data.parms();
            size_t plain_coeff_count = plain.coeff_count();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            auto plain_modulus = context_data.parms().plain_modulus();
            auto coeff_div_plain_modulus = context_data.coeff_div_plain_modulus();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            uint64_t q_mod_t = context_data.coeff_modulus_mod_plain_modulus();
#ifdef SEAL_DEBUG
            if (plain_coeff_count > parms.poly_modulus_degree())
            {
                throw std::invalid_argument("invalid plaintext");
            }
            if (destination.poly_modulus_degree() != parms.poly_modulus_degree())
            {
                throw invalid_argument("destination is not valid for encryption parameters");
            }
#endif
            // Coefficients of plain m multiplied by coeff_modulus q, divided by plain_modulus t,
            // and rounded to the nearest integer (rounded up in case of a tie). Equivalent to
            // floor((q * m + floor((t+1) / 2)) / t).
            SEAL_ITERATE(iter(plain.data(), size_t(0)), plain_coeff_count, [&](auto I) {
                // Compute numerator = (q mod t) * m[i] + (t+1)/2
                unsigned long long prod[2]{ 0, 0 };
                uint64_t numerator[2]{ 0, 0 };
                multiply_uint64(get<0>(I), q_mod_t, prod);
                unsigned char carry = add_uint64(*prod, plain_upper_half_threshold, numerator);
                numerator[1] = static_cast<uint64_t>(prod[1]) + static_cast<uint64_t>(carry);

                // Compute fix[0] = floor(numerator / t)
                uint64_t fix[2] = { 0, 0 };
                divide_uint128_inplace(numerator, plain_modulus.value(), fix);

                // Add to ciphertext: floor(q / t) * m + increment
                size_t coeff_index = get<1>(I);
                SEAL_ITERATE(
                    iter(destination, coeff_modulus, coeff_div_plain_modulus), coeff_modulus_size, [&](auto J) {
                        uint64_t scaled_rounded_coeff = multiply_add_uint_mod(get<0>(I), get<2>(J), fix[0], get<1>(J));
                        get<0>(J)[coeff_index] = sub_uint_mod(get<0>(J)[coeff_index], scaled_rounded_coeff, get<1>(J));
                    });
            });
        }

        
        void multiply_sub_plain_with_scaling_variant_cuda(
            const Plaintext &plain, const SEALContext::ContextData &context_data, uint64_t *destination)
        {
            auto &parms = context_data.parms();
            size_t plain_coeff_count = plain.coeff_count();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            uint64_t *d_plain = plain.d_data();

            auto plain_modulus = context_data.parms().plain_modulus();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            uint64_t q_mod_t = context_data.coeff_modulus_mod_plain_modulus();
            uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
            uint64_t *d_ratio = parms.d_coeff_modulus_ratio_1();

            uint64_t *d_coeff_div_plain_modulus_operand = context_data.d_coeff_div_plain_modulus_operand();
            uint64_t *d_coeff_div_plain_modulus_quoient = context_data.d_coeff_div_plain_modulus_quoitent();

            multiply_sub_plain_with_scaling_variant_kernel<<<(plain_coeff_count * coeff_modulus_size + 255) / 256, 256>>>(
                d_plain, 
                destination, 
                plain_coeff_count, 
                coeff_count,
                coeff_modulus_size, 
                q_mod_t, 
                plain_upper_half_threshold,
                d_coeff_modulus_value, 
                d_ratio,
                d_coeff_div_plain_modulus_operand, 
                d_coeff_div_plain_modulus_quoient, 
                plain_modulus.value());
        }

    } // namespace util
} // namespace seal
