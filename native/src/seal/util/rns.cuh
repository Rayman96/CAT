// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/iterator.h"
#include "seal/util/ntt.h"
#include "seal/util/pointer.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/uintarith.cuh"
#include "seal/util/uintarithmod.cuh"
#include "seal/util/uintarithsmallmod.cuh"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace seal
{
    namespace util
    {
        uint64_t power_m(uint64_t x, uint64_t n, const Modulus &modulus);
        
        // __device__ inline uint64_t multiply_add_uint_mod_kernel(
        //     uint64_t operand1, uint64_t operand2, uint64_t operand3, const uint64_t modulus, uint64_t ratio0,
        //     uint64_t ratio1)
        // {
        //     // Lazy reduction
        //     unsigned long long temp[2];
        //     multiply_uint64_kernel(operand1, operand2, temp);
        //     temp[1] += add_uint64_kernel(temp[0], operand3, temp);
        //     uint64_t ratio[2];
        //     ratio[0] = ratio0;
        //     ratio[1] = ratio1;
        //     return barrett_reduce_128_kernel2(temp, modulus, ratio);
        // }

        __device__ inline uint64_t dot_product_mod_kernel(
            std::uint64_t *operand1, std::uint64_t *operand2, std::size_t count, uint64_t modulus, uint64_t *ratio){
                static_assert(SEAL_MULTIPLY_ACCUMULATE_MOD_MAX >= 16, "SEAL_MULTIPLY_ACCUMULATE_MOD_MAX");
                unsigned long long accumulator[2]{ 0, 0 };
                switch (count)
                {
                case 0:
                    return 0;
                case 1:
                    multiply_accumulate_uint64_kernel<1>(operand1, operand2, accumulator);
                    break;
                case 2:
                    multiply_accumulate_uint64_kernel<2>(operand1, operand2, accumulator);
                    break;
                case 3:
                    multiply_accumulate_uint64_kernel<3>(operand1, operand2, accumulator);
                    break;
                case 4:
                    multiply_accumulate_uint64_kernel<4>(operand1, operand2, accumulator);
                    break;
                case 5:
                    multiply_accumulate_uint64_kernel<5>(operand1, operand2, accumulator);
                    break;
                case 6:
                    multiply_accumulate_uint64_kernel<6>(operand1, operand2, accumulator);
                    break;
                case 7:
                    multiply_accumulate_uint64_kernel<7>(operand1, operand2, accumulator);
                    break;
                case 8:
                    multiply_accumulate_uint64_kernel<8>(operand1, operand2, accumulator);
                    break;
                case 9:
                    multiply_accumulate_uint64_kernel<9>(operand1, operand2, accumulator);
                    break;
                case 10:
                    multiply_accumulate_uint64_kernel<10>(operand1, operand2, accumulator);
                    break;
                case 11:
                    multiply_accumulate_uint64_kernel<11>(operand1, operand2, accumulator);
                    break;
                case 12:
                    multiply_accumulate_uint64_kernel<12>(operand1, operand2, accumulator);
                    break;
                case 13:
                    multiply_accumulate_uint64_kernel<13>(operand1, operand2, accumulator);
                    break;
                case 14:
                    multiply_accumulate_uint64_kernel<14>(operand1, operand2, accumulator);
                    break;
                case 15:
                    multiply_accumulate_uint64_kernel<15>(operand1, operand2, accumulator);
                    break;
                case 16:
                largest_case:
                    multiply_accumulate_uint64_kernel<16>(operand1, operand2, accumulator);
                    break;
                default:
                    accumulator[0] = dot_product_mod_kernel(operand1 + 16, operand2 + 16, count - 16, modulus, ratio);
                    goto largest_case;
                };
                uint64_t result = barrett_reduce_128_kernel2(accumulator, modulus, ratio);
                return result;
        }

        __global__ void modulo_poly_coeffs_kernel(
            uint64_t *last_input, uint64_t coeff_count, uint64_t base_q_value, uint64_t base_q_ratio,
            uint64_t *temp_result);

        __global__ void modulo_poly_coeffs_kernel(
            uint64_t *last_input, uint64_t coeff_count, size_t modulu_size, uint64_t *base_q_value, uint64_t *base_q_ratio,
            uint64_t *temp_result);


       __global__ void multiply_poly_scalar_coeffmod_kernel_kernel(uint64_t *input, uint64_t *result,
            size_t coeff_count, size_t base_size,
            uint64_t *modulus_value, uint64_t *modulus_ratio,
            uint64_t scalar);

       __global__ void multiply_poly_scalar_coeffmod_kernel_one_modulu(uint64_t *input, uint64_t *result,
            size_t coeff_count,
            uint64_t modulus_value, uint64_t modulus_ratio,
            uint64_t scalar);

        class RNSBase
        {
        public:
            RNSBase(const std::vector<Modulus> &rnsbase, MemoryPoolHandle pool);

            RNSBase(RNSBase &&source) = default;

            RNSBase(const RNSBase &copy, MemoryPoolHandle pool);

            RNSBase(const RNSBase &copy) : RNSBase(copy, copy.pool_)
            {}

            RNSBase &operator=(const RNSBase &assign) = delete;

            SEAL_NODISCARD inline const Modulus &operator[](std::size_t index) const
            {
                if (index >= size_)
                {
                    throw std::out_of_range("index is out of range");
                }
                return base_[index];
            }

            SEAL_NODISCARD inline std::size_t size() const noexcept
            {
                return size_;
            }

            SEAL_NODISCARD bool contains(const Modulus &value) const noexcept;

            SEAL_NODISCARD bool is_subbase_of(const RNSBase &superbase) const noexcept;

            SEAL_NODISCARD inline bool is_superbase_of(const RNSBase &subbase) const noexcept
            {
                return subbase.is_subbase_of(*this);
            }

            SEAL_NODISCARD inline bool is_proper_subbase_of(const RNSBase &superbase) const noexcept
            {
                return (size_ < superbase.size_) && is_subbase_of(superbase);
            }

            SEAL_NODISCARD inline bool is_proper_superbase_of(const RNSBase &subbase) const noexcept
            {
                return (size_ > subbase.size_) && !is_subbase_of(subbase);
            }

            SEAL_NODISCARD RNSBase extend(const Modulus &value) const;

            SEAL_NODISCARD RNSBase extend(const RNSBase &other) const;

            SEAL_NODISCARD RNSBase drop() const;

            SEAL_NODISCARD RNSBase drop(const Modulus &value) const;

            void decompose(std::uint64_t *value, MemoryPoolHandle pool) const;

            void decompose_array(std::uint64_t *value, std::size_t count, MemoryPoolHandle pool) const;
            void decompose_array_cuda(std::uint64_t *value, std::size_t count) const;

            void compose(std::uint64_t *value, MemoryPoolHandle pool) const;

            void compose_array(std::uint64_t *value, std::size_t count, MemoryPoolHandle pool) const;
            void compose_array_cuda(std::uint64_t *value, std::size_t count, MemoryPoolHandle pool) const;

            SEAL_NODISCARD inline const Modulus *base() const noexcept
            {
                return base_.get();
            }

            SEAL_NODISCARD inline const std::uint64_t *base_prod() const noexcept
            {
                return base_prod_.get();
            }

            SEAL_NODISCARD inline const std::uint64_t *punctured_prod_array() const noexcept
            {
                return punctured_prod_array_.get();
            }

            SEAL_NODISCARD inline const MultiplyUIntModOperand *inv_punctured_prod_mod_base_array() const noexcept
            {
                return inv_punctured_prod_mod_base_array_.get();
            }

            SEAL_NODISCARD inline uint64_t *d_inv_punctured_prod_mod_base_array_quotient() const noexcept
            {
                return d_inv_punctured_prod_mod_base_array_quotient_;
            }

            SEAL_NODISCARD inline uint64_t *d_inv_punctured_prod_mod_base_array_operand() const noexcept
            {
                return d_inv_punctured_prod_mod_base_array_operand_;
            }

            SEAL_NODISCARD inline uint64_t *d_base_prod() const noexcept
            {
                return d_base_prod_;
            }

            SEAL_NODISCARD inline uint64_t *d_base() const noexcept
            {
                return d_base_;
            }

            SEAL_NODISCARD inline uint64_t *d_ratio0() const noexcept
            {
                return d_base_ratio0_;
            }

            SEAL_NODISCARD inline uint64_t *d_ratio1() const noexcept
            {
                return d_base_ratio1_;
            }

            SEAL_NODISCARD inline uint64_t *d_punctured_prod() const noexcept
            {
                return d_punctured_prod_;
            }

        private:
            RNSBase(MemoryPoolHandle pool) : pool_(std::move(pool)), size_(0)
            {
                if (!pool_)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }
            }

            bool initialize();

            MemoryPoolHandle pool_;

            std::size_t size_;

            Pointer<Modulus> base_;

            Pointer<std::uint64_t> base_prod_;

            Pointer<std::uint64_t> punctured_prod_array_;

            Pointer<MultiplyUIntModOperand> inv_punctured_prod_mod_base_array_;

            void set_GPU_params();

            // GPU要用的参数
            uint64_t *d_inv_punctured_prod_mod_base_array_quotient_;
            uint64_t *d_inv_punctured_prod_mod_base_array_operand_;
            uint64_t *d_base_prod_;
            uint64_t *d_base_ = nullptr;
            uint64_t *d_base_ratio0_ = nullptr;
            uint64_t *d_base_ratio1_ = nullptr;
            uint64_t *d_punctured_prod_ = nullptr;
        };

        class BaseConverter
        {
        public:
            BaseConverter(const RNSBase &ibase, const RNSBase &obase, MemoryPoolHandle pool)
                : pool_(std::move(pool)), ibase_(ibase, pool_), obase_(obase, pool_)
            {
                if (!pool_)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }

                initialize();
            }

            SEAL_NODISCARD inline std::size_t ibase_size() const noexcept
            {
                return ibase_.size();
            }

            SEAL_NODISCARD inline std::size_t obase_size() const noexcept
            {
                return obase_.size();
            }

            SEAL_NODISCARD inline const RNSBase &ibase() const noexcept
            {
                return ibase_;
            }

            SEAL_NODISCARD inline const RNSBase &obase() const noexcept
            {
                return obase_;
            }

            void fast_convert(ConstCoeffIter in, CoeffIter out, MemoryPoolHandle pool) const;

            void fast_convert_array(ConstRNSIter in, RNSIter out, MemoryPoolHandle pool) const;

            void fast_convert_array_cuda(uint64_t *d_in, uint64_t *d_out, size_t count) const;


            // The exact base convertion function, only supports obase size of 1.
            void exact_convert_array(ConstRNSIter in, CoeffIter out, MemoryPoolHandle) const;

            void exact_convert_array_cuda(uint64_t *in, uint64_t *out, size_t count) const;

            void ensure_size(uint64_t **input, size_t current_size, size_t &size) const;


        private:
            BaseConverter(const BaseConverter &copy) = delete;

            BaseConverter(BaseConverter &&source) = delete;

            BaseConverter &operator=(const BaseConverter &assign) = delete;

            BaseConverter &operator=(BaseConverter &&assign) = delete;

            void initialize();

            MemoryPoolHandle pool_;

            RNSBase ibase_;

            RNSBase obase_;

            Pointer<Pointer<std::uint64_t>> base_change_matrix_;

            uint64_t *d_base_change_matrix_=nullptr;

            mutable uint64_t *d_temp_convert_ = nullptr;

            mutable size_t d_temp_convert_size_ = 0;

        };

        class RNSTool
        {
        public:
            /**
            @throws std::invalid_argument if poly_modulus_degree is out of range, coeff_modulus is not valid, or pool is
            invalid.
            @throws std::logic_error if coeff_modulus and extended bases do not support NTT or are not coprime.
            */
            RNSTool(
                std::size_t poly_modulus_degree, const RNSBase &coeff_modulus, const Modulus &plain_modulus,
                MemoryPoolHandle pool);

            /**
            @param[in] input Must be in RNS form, i.e. coefficient must be less than the associated modulus.
            */
            void divide_and_round_q_last_inplace(RNSIter input, MemoryPoolHandle pool) const;
            void divide_and_rount_q_last_inplace_cuda(uint64_t *input) const;

            void divide_and_round_q_last_ntt_inplace(
                RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const;

            /**
            Shenoy-Kumaresan conversion from Bsk to q
            */
            void fastbconv_sk(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;
            void fastbconv_sk_cuda(uint64_t *d_in, uint64_t *d_destination) const;

            /**
            Montgomery reduction mod q; changes base from Bsk U {m_tilde} to Bsk
            */
            void sm_mrq(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;
            void sm_mrq_cuda(uint64_t *input, uint64_t *destination) const;

            /**
            Divide by q and fast floor from q U Bsk to Bsk
            */
            void fast_floor(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;
            void fast_floor_cuda(uint64_t *d_in, uint64_t *d_destination) const;

            /**
            Fast base conversion from q to Bsk U {m_tilde}
            */
            void fastbconv_m_tilde(ConstRNSIter input, RNSIter destination, MemoryPoolHandle pool) const;
            void fastbconv_m_tilde_cuda(uint64_t *input, uint64_t *destination) const;

            /**
            Compute round(t/q * |input|_q) mod t exactly
            */
            void decrypt_scale_and_round(ConstRNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const;
            void decrypt_scale_and_round_cuda(uint64_t *input, uint64_t *destination) const;


            /**
            Remove the last q for bgv ciphertext
            */
            void mod_t_and_divide_q_last_ntt_inplace(
                RNSIter input, ConstNTTTablesIter rns_ntt_tables, MemoryPoolHandle pool) const;

            void mod_t_and_divide_q_last_ntt_inplace_cuda(
                uint64_t *input, uint64_t *matrix_n1, uint64_t *matrix_n2, uint64_t *matrix_n12, uint64_t *modulu, uint64_t *ratio0, uint64_t *ratio1,
                uint64_t *roots, int *bits, std::pair<int, int> split_coeff,uint64_t *d_inv_root_powers,ConstNTTTablesIter rns_ntt_tables) const;
            
             void mod_t_and_divide_q_last_ntt_inplace_cuda_v1(
            uint64_t *input, uint64_t *d_root_powers, uint64_t *d_inv_root_powers,ConstNTTTablesIter rns_ntt_tables, cudaStream_t *streams, int stream_num) const;

            void divide_and_round_q_last_ntt_inplace_cuda_test(
                uint64_t *d_input, uint64_t *matrix_n1, uint64_t *matrix_n2, uint64_t *matrix_n12, uint64_t *modulu, uint64_t *ratio0, uint64_t *ratio1,
                uint64_t *roots, int *bits, std::pair<int, int> split_coeff,uint64_t *d_inv_root_powers, ConstNTTTablesIter rns_ntt_tables) const;

            void divide_and_round_q_last_ntt_inplace_cuda_v1(
                uint64_t *d_input, uint64_t *d_root_powers, uint64_t *d_inv_root_powers, ConstNTTTablesIter rns_ntt_tables, cudaStream_t *streams, int stream_num) const;
            

            /**
            Compute mod t
            */
            void decrypt_modt(RNSIter phase, CoeffIter destination, MemoryPoolHandle pool) const;
            void decrypt_modt_cuda(uint64_t *phase, uint64_t *destination, size_t count) const;

            void ensure_size(uint64_t **input, size_t current_size, size_t &size) const;


            SEAL_NODISCARD inline auto inv_q_last_mod_q() const noexcept
            {
                return inv_q_last_mod_q_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk_ntt_tables() const noexcept
            {
                return base_Bsk_ntt_tables_.get();
            }

            SEAL_NODISCARD inline uint64_t *d_base_Bsk_root_powers() const noexcept
            {
                return d_base_Bsk_root_powers_;
            }

            SEAL_NODISCARD inline uint64_t *bae_Bsk_root_power() const noexcept
            {
                return d_base_Bsk_root_powers_;
            }

            SEAL_NODISCARD inline uint64_t *bae_Bsk_bit_count() const noexcept
            {
                return d_baes_Bsk_bit_count_;
            }
                

            SEAL_NODISCARD inline auto base_q() const noexcept
            {
                return base_q_.get();
            }

            SEAL_NODISCARD inline auto base_B() const noexcept
            {
                return base_B_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk() const noexcept
            {
                return base_Bsk_.get();
            }

            SEAL_NODISCARD inline auto base_Bsk_m_tilde() const noexcept
            {
                return base_Bsk_m_tilde_.get();
            }

            SEAL_NODISCARD inline auto base_t_gamma() const noexcept
            {
                return base_t_gamma_.get();
            }

            SEAL_NODISCARD inline auto &m_tilde() const noexcept
            {
                return m_tilde_;
            }

            SEAL_NODISCARD inline auto &m_sk() const noexcept
            {
                return m_sk_;
            }

            SEAL_NODISCARD inline auto &t() const noexcept
            {
                return t_;
            }

            SEAL_NODISCARD inline auto &gamma() const noexcept
            {
                return gamma_;
            }

            SEAL_NODISCARD inline auto &inv_q_last_mod_t() const noexcept
            {
                return inv_q_last_mod_t_;
            }

            SEAL_NODISCARD inline uint64_t *d_inv_q_last_mod_q_operand() const noexcept
            {
                return d_inv_q_last_mod_q_operand_;
            }

            SEAL_NODISCARD inline uint64_t *d_inv_q_last_mod_q_quotient() const noexcept
            {
                return d_inv_q_last_mod_q_quotient_;
            }

            SEAL_NODISCARD inline const uint64_t &q_last_mod_t() const noexcept
            {
                return q_last_mod_t_;
            }

        private:
            RNSTool(const RNSTool &copy) = delete;

            RNSTool(RNSTool &&source) = delete;

            RNSTool &operator=(const RNSTool &assign) = delete;

            RNSTool &operator=(RNSTool &&assign) = delete;

            /**
            Generates the pre-computations for the given parameters.
            */
            void initialize(std::size_t poly_modulus_degree, const RNSBase &q, const Modulus &t);

            MemoryPoolHandle pool_;

            std::size_t coeff_count_ = 0;

            Pointer<RNSBase> base_q_;

            Pointer<RNSBase> base_B_;

            Pointer<RNSBase> base_Bsk_;

            Pointer<RNSBase> base_Bsk_m_tilde_;

            Pointer<RNSBase> base_t_gamma_;

            // Base converter: q --> B_sk
            Pointer<BaseConverter> base_q_to_Bsk_conv_;

            // Base converter: q --> {m_tilde}
            Pointer<BaseConverter> base_q_to_m_tilde_conv_;

            // Base converter: B --> q
            Pointer<BaseConverter> base_B_to_q_conv_;

            // Base converter: B --> {m_sk}
            Pointer<BaseConverter> base_B_to_m_sk_conv_;

            // Base converter: q --> {t, gamma}
            Pointer<BaseConverter> base_q_to_t_gamma_conv_;

            // Base converter: q --> t
            Pointer<BaseConverter> base_q_to_t_conv_;

            // prod(q)^(-1) mod Bsk
            Pointer<MultiplyUIntModOperand> inv_prod_q_mod_Bsk_;

            // prod(q)^(-1) mod m_tilde
            MultiplyUIntModOperand neg_inv_prod_q_mod_m_tilde_;

            // prod(B)^(-1) mod m_sk
            MultiplyUIntModOperand inv_prod_B_mod_m_sk_;

            // gamma^(-1) mod t
            MultiplyUIntModOperand inv_gamma_mod_t_;

            // prod(B) mod q
            Pointer<std::uint64_t> prod_B_mod_q_;

            // m_tilde^(-1) mod Bsk
            Pointer<MultiplyUIntModOperand> inv_m_tilde_mod_Bsk_;

            // prod(q) mod Bsk
            Pointer<std::uint64_t> prod_q_mod_Bsk_;

            // -prod(q)^(-1) mod {t, gamma}
            Pointer<MultiplyUIntModOperand> neg_inv_q_mod_t_gamma_;

            // prod({t, gamma}) mod q
            Pointer<MultiplyUIntModOperand> prod_t_gamma_mod_q_;

            // q[last]^(-1) mod q[i] for i = 0..last-1
            Pointer<MultiplyUIntModOperand> inv_q_last_mod_q_;

            // NTTTables for Bsk
            Pointer<NTTTables> base_Bsk_ntt_tables_;

            uint64_t *d_base_Bsk_root_powers_ = nullptr;

            uint64_t *d_baes_Bsk_bit_count_;

            Modulus m_tilde_;

            Modulus m_sk_;

            Modulus t_;

            Modulus gamma_;

            std::uint64_t inv_q_last_mod_t_ = 1;

            std::uint64_t q_last_mod_t_ = 1;

            uint64_t *d_prod_B_mod_Bsk_ = nullptr;

            uint64_t *d_prod_B_mod_q_ = nullptr;

            uint64_t *d_inv_m_tilde_mod_Bsk_operand_ = nullptr;
            
            uint64_t *d_inv_m_tilde_mod_Bsk_quotient_ = nullptr;

            uint64_t *d_inv_prod_q_mod_Bsk_operand_ = nullptr;

            uint64_t *d_inv_prod_q_mod_Bsk_quotient_ = nullptr;

            uint64_t *d_inv_q_last_mod_q_operand_ = nullptr;

            uint64_t *d_inv_q_last_mod_q_quotient_ = nullptr;

            mutable uint64_t *d_temp_ = nullptr;

            mutable uint64_t *d_input_m_tilde_ = nullptr;

            mutable size_t d_temp_size_ = 0;

            mutable size_t d_input_m_tilde_size_ = 0;


        };
    } // namespace util
} // namespace seal
