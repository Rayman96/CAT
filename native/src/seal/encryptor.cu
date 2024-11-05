// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/encryptor.h"
#include "seal/modulus.h"
#include "seal/randomtostd.h"
#include "seal/util/common.h"
#include "seal/util/iterator.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/helper.cuh"
#include "seal/util/ntt_helper.cuh"
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;
using namespace seal::util;

namespace seal
{
    Encryptor::Encryptor(const SEALContext &context, const PublicKey &public_key) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_public_key(public_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

    Encryptor::Encryptor(const SEALContext &context, const SecretKey &secret_key) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_secret_key(secret_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

    Encryptor::Encryptor(const SEALContext &context, const PublicKey &public_key, const SecretKey &secret_key)
        : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_public_key(public_key);
        set_secret_key(secret_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

// bgv 和 bfv 用
    void Encryptor::encrypt_zero_internal(
        parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        
        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = public_key_.data().size();
        bool is_ntt_form = false;

        if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            is_ntt_form = true;
        }
        else if (parms.scheme() != scheme_type::bfv)
        {
            throw invalid_argument("unsupported scheme");
        }

        // Resize destination and save results
        destination.resize_pure_gpu(context_, parms_id, 2);

        // If asymmetric key encryption
        uint64_t *d_destination = destination.d_data();
        if (is_asymmetric)
        {
            // printf("encrypte goes is_asymmetric\n");
            auto prev_context_data_ptr = context_data.prev_context_data();
            if (prev_context_data_ptr)
            {
                // Requires modulus switching
                auto &prev_context_data = *prev_context_data_ptr;
                auto &prev_parms_id = prev_context_data.parms_id();
                auto rns_tool = prev_context_data.rns_tool();
                auto &context_data_prev = *context_.get_context_data(prev_parms_id);
                auto &parms_prev = context_data_prev.parms();
                auto &coeff_modulus_prev = parms_prev.coeff_modulus();
                uint64_t *prev_d_inv_root_powers = context_data_prev.d_inv_root_powers();

                size_t new_coeff_modulus_size = coeff_modulus_prev.size(); 

                std::pair<int, int> split_result = prev_context_data.split_degree();
                uint64_t *prev_modulu_value = parms_prev.d_coeff_modulus_value();
                uint64_t *prev_ratio0 = parms_prev.d_coeff_modulus_ratio_0();
                uint64_t *prev_ratio1 = parms_prev.d_coeff_modulus_ratio_1();
                int *d_bit_count = prev_context_data.d_bit_count();
                uint64_t *d_roots = prev_context_data.d_roots();
                uint64_t *d_root_matrix_n1 = prev_context_data.d_root_matrix_n1();
                uint64_t *d_root_matrix_n2 = prev_context_data.d_root_matrix_n2();
                uint64_t *d_root_matrix_n12 = prev_context_data.d_root_matrix_n12();
                uint64_t *d_root_powers = prev_context_data.d_root_powers();

                // Zero encryption without modulus switching
                uint64_t *d_middle_destination = nullptr;

                printf("encrypted_size: %d, coeff_count: %d, new_coeff_modulus_size: %d, coeff_modulus_size:%d\n", encrypted_size, coeff_count, new_coeff_modulus_size, coeff_modulus_size);
                allocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * new_coeff_modulus_size);
                util::encrypt_zero_asymmetric_ckks_test(public_key_, context_, prev_parms_id, is_ntt_form, d_middle_destination);
                SEAL_ITERATE(iter(destination, size_t(0)), encrypted_size, [&](auto I) {
                    // bfv switch-to-next
                    if (parms.scheme() == scheme_type::bfv)
                    {
                        // rns_tool->divide_and_round_q_last_inplace(get<0>(I), pool);
                        // set_poly(get<0>(I), coeff_count, coeff_modulus_size, get<1>(I));
                        rns_tool->divide_and_rount_q_last_inplace_cuda(d_middle_destination + get<1>(I) * coeff_count * new_coeff_modulus_size);
                        checkCudaErrors(cudaMemcpy(destination.d_data() + get<1>(I) * coeff_count * coeff_modulus_size, 
                                    d_middle_destination + get<1>(I) * coeff_count * new_coeff_modulus_size, 
                                    coeff_count * coeff_modulus_size * sizeof(uint64_t), 
                                    cudaMemcpyDeviceToDevice));

                    }
                    // bgv switch-to-next
                    else if (parms.scheme() == scheme_type::bgv)
                    {

                        printf("goes into bgv encrytption ntt test\n");
                        const int threads_per_block = 256;
                        const int blocks_per_grid = (coeff_count * new_coeff_modulus_size + threads_per_block - 1) / threads_per_block;

                        rns_tool->mod_t_and_divide_q_last_ntt_inplace_cuda(
                            d_destination + get<1>(I) * coeff_count * new_coeff_modulus_size,
                            d_root_matrix_n1, d_root_matrix_n2, d_root_matrix_n12,
                            prev_modulu_value, prev_ratio0, prev_ratio1, d_roots, d_bit_count,split_result,
                            prev_d_inv_root_powers,
                            prev_context_data.small_ntt_tables());       

                        set_poly_kernel<<<blocks_per_grid, threads_per_block>>>(d_destination + get<1>(I) * coeff_count * new_coeff_modulus_size, 
                        d_middle_destination, coeff_count, new_coeff_modulus_size);

                        checkCudaErrors(cudaMemcpy(destination.d_data() + get<1>(I) * coeff_count * new_coeff_modulus_size, 
                                                d_middle_destination, 
                                                coeff_count * coeff_modulus_size * sizeof(uint64_t), 
                                                cudaMemcpyDeviceToDevice));

                    }
                });
                deallocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * new_coeff_modulus_size);

            }
            else
            {
                printf("encrypte goes is_asymmetric no prev_context_data_ptr\n");
                // Does not require modulus switching
                // util::encrypt_zero_asymmetric(public_key_, context_, parms_id, is_ntt_form, destination);

                util::encrypt_zero_asymmetric_ckks_test(public_key_, context_, parms_id, is_ntt_form, d_destination);
                checkCudaErrors(cudaMemcpy(destination.data(), d_destination, coeff_count * encrypted_size * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            }
            destination.parms_id() = parms_id;
            destination.is_ntt_form() = is_ntt_form;
            destination.scale() = 1.0;
            destination.correction_factor() = 1;
        }
        else
        {
            printf("encrypte goes not_asymmetric\n");
            // Does not require modulus switching
            util::encrypt_zero_symmetric(secret_key_, context_, parms_id, is_ntt_form, save_seed, destination);
        }


    }
    
    void Encryptor::encrypt_zero_internal_bgv(
        parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t coeff_count = parms.poly_modulus_degree();
        bool is_ntt_form = false;

        if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            is_ntt_form = true;
        }
        else if (parms.scheme() != scheme_type::bfv)
        {
            throw invalid_argument("unsupported scheme");
        }

        // Resize destination and save results
        destination.resize(context_, parms_id, 2);

        // If asymmetric key encryption
        if (is_asymmetric)
        {
            auto prev_context_data_ptr = context_data.prev_context_data();
            if (prev_context_data_ptr)
            {
                // Requires modulus switching
                auto &prev_context_data = *prev_context_data_ptr;
                auto &prev_parms_id = prev_context_data.parms_id();
                auto rns_tool = prev_context_data.rns_tool();

                // Zero encryption without modulus switching
                Ciphertext temp(pool);
                util::encrypt_zero_asymmetric(public_key_, context_, prev_parms_id, is_ntt_form, temp);

                // Modulus switching
                SEAL_ITERATE(iter(temp, destination), temp.size(), [&](auto I) {
                    if (parms.scheme() == scheme_type::ckks)
                    {
                        rns_tool->divide_and_round_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    // bfv switch-to-next
                    else if (parms.scheme() == scheme_type::bfv)
                    {
                        rns_tool->divide_and_round_q_last_inplace(get<0>(I), pool);
                    }
                    // bgv switch-to-next
                    else if (parms.scheme() == scheme_type::bgv)
                    {
                        rns_tool->mod_t_and_divide_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    set_poly(get<0>(I), coeff_count, coeff_modulus_size, get<1>(I));
                });

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = temp.scale();
                destination.correction_factor() = temp.correction_factor();
            }
            else
            {
                // Does not require modulus switching
                util::encrypt_zero_asymmetric(public_key_, context_, parms_id, is_ntt_form, destination);
            }
        }
        else
        {
            // Does not require modulus switching
            util::encrypt_zero_symmetric(secret_key_, context_, parms_id, is_ntt_form, save_seed, destination);
        }
    }


    void Encryptor::encrypt_zero_internal_ckks_asymmetric(
        parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination, const Plaintext &plain,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        std::uint64_t *d_coeff_modulus = parms.d_coeff_modulus_value();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t encrypted_size = public_key_.data().size();

        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();
        bool is_ntt_form = false;

        if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            is_ntt_form = true;
        }
        else if (parms.scheme() != scheme_type::bfv)
        {
            throw invalid_argument("unsupported scheme");
        }

        // Resize destination and save results
        destination.resize_pure_gpu(context_, parms_id, 2);

        // If asymmetric key encryption
        uint64_t *d_plain = plain.d_data();

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (coeff_count * coeff_modulus_size + threadsPerBlock - 1) / threadsPerBlock;
        if (is_asymmetric)
        {
            auto prev_context_data_ptr = context_data.prev_context_data();
            if (prev_context_data_ptr)
            {
                // printf("encrypte goes is_asymmetric\n");
                // Requires modulus switching
                auto &prev_context_data = *prev_context_data_ptr;
                auto &prev_parms_id = prev_context_data.parms_id();
                auto rns_tool = prev_context_data.rns_tool();
                auto &context_data_prev = *context_.get_context_data(prev_parms_id);
                auto &parms_prev = context_data_prev.parms();
                auto &coeff_modulus_prev = parms_prev.coeff_modulus();
                size_t new_coeff_modulus_size = coeff_modulus_prev.size(); 
                uint64_t *prev_d_inv_root_powers = context_data_prev.d_inv_root_powers();


                std::pair<int, int> split_result = context_data_prev.split_degree();
                uint64_t *prev_modulu_value = parms_prev.d_coeff_modulus_value();
                uint64_t *prev_ratio0 = parms_prev.d_coeff_modulus_ratio_0();
                uint64_t *prev_ratio1 = parms_prev.d_coeff_modulus_ratio_1();
                int *d_bit_count = context_data_prev.d_bit_count();
                uint64_t *d_roots = context_data_prev.d_roots();
                uint64_t *d_root_matrix_n1 = context_data_prev.d_root_matrix_n1();
                uint64_t *d_root_matrix_n2 = context_data_prev.d_root_matrix_n2();
                uint64_t *d_root_matrix_n12 = context_data_prev.d_root_matrix_n12();
                uint64_t *d_root_powers = context_data_prev.d_root_powers();

                // Zero encryption without modulus switching
                uint64_t *d_middle_destination = nullptr;
                allocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * new_coeff_modulus_size);
                util::encrypt_zero_asymmetric_ckks_test(public_key_, context_, prev_parms_id, is_ntt_form, d_middle_destination);

                SEAL_ITERATE(iter(destination, size_t(0)), encrypted_size, [&](auto I) {
                    if (parms.scheme() == scheme_type::ckks)
                    {
#if NTT_VERSION == 3
                        rns_tool->divide_and_round_q_last_ntt_inplace_cuda_test(
                            d_middle_destination + get<1>(I) * coeff_count * new_coeff_modulus_size,
                            d_root_matrix_n1, d_root_matrix_n2, d_root_matrix_n12,
                            prev_modulu_value, prev_ratio0, prev_ratio1, d_roots, d_bit_count,split_result,
                            prev_d_inv_root_powers,
                            prev_context_data.small_ntt_tables());

#else 
                        rns_tool->divide_and_round_q_last_ntt_inplace_cuda_v1(
                            d_middle_destination + get<1>(I) * coeff_count * new_coeff_modulus_size,
                            d_root_powers,
                            prev_d_inv_root_powers,
                            prev_context_data.small_ntt_tables(),
                            ntt_steam, stream_num);

#endif
// 因为modulus不同，所以需要转换，无法直接直接使用destination.d_data()作为地址，而且需要循环拷贝

                    checkCudaErrors(cudaMemcpy(destination.d_data() + get<1>(I) * coeff_count * coeff_modulus_size, 
                                d_middle_destination + get<1>(I) * coeff_count * new_coeff_modulus_size, 
                                coeff_count * coeff_modulus_size * sizeof(uint64_t), 
                                cudaMemcpyDeviceToDevice));

                    }
                });

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = 1.0;
                destination.correction_factor() = 1;

                deallocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * new_coeff_modulus_size);
            }
            else
            {
                uint64_t *d_middle_destination = nullptr;
                allocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * coeff_modulus_size);
        
                // Does not require modulus switching
                util::encrypt_zero_asymmetric_ckks_test(public_key_, context_, parms_id, is_ntt_form, d_middle_destination);
                
                add_poly_coeffmod_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                        d_middle_destination, d_plain, coeff_count, coeff_modulus_size, d_coeff_modulus, d_middle_destination);

                checkCudaErrors(cudaMemcpy(destination.d_data(), d_middle_destination, coeff_count * encrypted_size * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
                destination.scale() = plain.scale();
                deallocate_gpu<uint64_t>(&d_middle_destination, coeff_count * encrypted_size * coeff_modulus_size);

            }
        }
        else
        {
// TODO: sysmetric encryption
            printf("encrypt goes sysmetric\n");
            // Does not require modulus switching
            util::encrypt_zero_symmetric(secret_key_, context_, parms_id, is_ntt_form, save_seed, destination);
        }
    

    }


    void Encryptor::encrypt_zero_internal_origin(
        parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t coeff_count = parms.poly_modulus_degree();
        bool is_ntt_form = false;

        if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            is_ntt_form = true;
        }
        else if (parms.scheme() != scheme_type::bfv)
        {
            throw invalid_argument("unsupported scheme");
        }

        // Resize destination and save results
        destination.resize(context_, parms_id, 2);

        // If asymmetric key encryption
        if (is_asymmetric)
        {
            auto prev_context_data_ptr = context_data.prev_context_data();
            if (prev_context_data_ptr)
            {
                // Requires modulus switching
                auto &prev_context_data = *prev_context_data_ptr;
                auto &prev_parms_id = prev_context_data.parms_id();
                auto rns_tool = prev_context_data.rns_tool();

                // Zero encryption without modulus switching
                Ciphertext temp(pool);
                // checkCudaErrors(cudaMemcpy((void **)public_key_.data().data(), 
                //                             public_key_.data().d_data(), 
                //                             2 * context_.key_context_data()->parms().coeff_modulus().size() * context_.key_context_data()->parms().poly_modulus_degree() * sizeof(uint64_t), 
                //                             cudaMemcpyDeviceToHost));

                PublicKey public_key_copy = public_key_;
                public_key_copy.to_cpu();

                util::encrypt_zero_asymmetric(public_key_copy, context_, prev_parms_id, is_ntt_form, temp);
                // util::encrypt_zero_asymmetric_ckks_test(public_key_copy, context_, prev_parms_id, is_ntt_form, temp);
                

                // Modulus switching
                SEAL_ITERATE(iter(temp, destination), temp.size(), [&](auto I) {
                    if (parms.scheme() == scheme_type::ckks)
                    {
                        rns_tool->divide_and_round_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    // bfv switch-to-next
                    else if (parms.scheme() == scheme_type::bfv)
                    {
                        rns_tool->divide_and_round_q_last_inplace(get<0>(I), pool);

                    }
                    // bgv switch-to-next
                    else if (parms.scheme() == scheme_type::bgv)
                    {
                        rns_tool->mod_t_and_divide_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    set_poly(get<0>(I), coeff_count, coeff_modulus_size, get<1>(I));
                });

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = temp.scale();
                destination.correction_factor() = temp.correction_factor();
            }
            else
            {
                // Does not require modulus switching
                util::encrypt_zero_asymmetric(public_key_, context_, parms_id, is_ntt_form, destination);
            }
        }
        else
        {
            // Does not require modulus switching
            util::encrypt_zero_symmetric(secret_key_, context_, parms_id, is_ntt_form, save_seed, destination);
        }
    }

namespace{
    __global__ void bgv_encrypt_helper2(uint64_t *input,
                                        uint64_t *output, 
                                        const unsigned long long  plain_coeff_count, 
                                        const unsigned long long coeff_modulus_size, 
                                        const unsigned long long coeff_count,
                                        uint64_t *plain_upper_half_increment, 
                                        uint64_t plain_upper_half_threshold){
        uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < 10)
        {
            // if (index < 3){
                printf("plain_coeff_count %lu, coeff_count %lu, coeff_modulus_size %lu, idx:%lu\n", plain_coeff_count, coeff_count, coeff_modulus_size, index);
                // printf("index %d, output_index %d, shift_modulu %d\n", index, output_index, shift_modulu);
            // }

            uint64_t modulu_index = index / coeff_count;
            uint64_t shift_modulu = (coeff_modulus_size - 1 - modulu_index);
            uint64_t output_index =  shift_modulu * coeff_count + index % coeff_count;
            printf("index: %lu, shift_modulu:%lu, output_index %llu\n", index, shift_modulu, output_index);

            if(index % coeff_count >= plain_coeff_count){
                printf("index % coeff bigger, plain_coeff_count %lu, coeff_count %lu\n", plain_coeff_count, coeff_count);
                output[output_index] = 0;
            } else {
                printf("one way\n");
                uint64_t plain_value = input[0];
                printf("plain_value %llu, plain_upper_half_threshold:%lu\n", plain_value, plain_upper_half_threshold);
                if (plain_value >= plain_upper_half_threshold){
                    printf("plain_value %llu, plain_upper_half_increment[shift_modulu] %llu\n", plain_value, plain_upper_half_increment[shift_modulu]);
                    // output[output_index] = plain_value + plain_upper_half_increment[shift_modulu];
                }else{
                    printf("other way\n");
                    // output[output_index] = plain_value;
                }
            }

            printf("output %llu\n", output[output_index]);

            
            index += blockDim.x * gridDim.x;
        }
    }
}

// check
    void Encryptor::encrypt_internal(
        const Plaintext &plain, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Minimal verification that the keys are set
        if (is_asymmetric)
        {
            if (!is_metadata_valid_for(public_key_, context_))
            {
                throw logic_error("public key is not set");
            }
        }
        else
        {
            if (!is_metadata_valid_for(secret_key_, context_))
            {
                throw logic_error("secret key is not set");
            }
        }

        // Verify that plain is valid
        // if (!is_valid_for(plain, context_))
        // {
        //     throw invalid_argument("plain is not valid for encryption parameters");
        // }

        auto scheme = context_.key_context_data()->parms().scheme();
        if (scheme == scheme_type::bfv)
        {
            if (plain.is_ntt_form())
            {
                throw invalid_argument("plain cannot be in NTT form");
            }
            auto &context_data = *context_.first_context_data();
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            // encrypt_zero_internal(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);
            // print_helper<<<1,3>>>(destination.d_data(), 3);

            // Ciphertext destination_copy;
            encrypt_zero_internal_origin(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);
            // destination.to_cpu();
            // print_helper<<<1,3>>>(destination_copy.d_data(), 3);

            // for (int i = 0; i < 3; i++) {
            //     printf("destination %llu\n", *(destination.data() + i));
            // }

            // destination.resize_pure_gpu(context_, context_.first_parms_id(), 2);

            // checkCudaErrors(cudaMemcpy(destination.d_data(), destination.data(), 2 * coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            // cudaDeviceSynchronize();
            // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
            // Result gets added into the c_0 term of ciphertext (c_0,c_1).
            multiply_add_plain_with_scaling_variant(plain, *context_.first_context_data(), *iter(destination));

            destination.to_gpu();
            // multiply_add_plain_with_scaling_variant_cuda(plain, *context_.first_context_data(), destination.d_data());
            // print_helper<<<1,3>>>(destination.d_data(), 3);

        }
        else if (scheme == scheme_type::ckks)
        {

            if (!plain.is_ntt_form())
            {
                throw invalid_argument("plain must be in NTT form");
            }

            auto context_data_ptr = context_.get_context_data(plain.parms_id());
            if (!context_data_ptr)
            {
                throw invalid_argument("plain is not valid for encryption parameters");
            }
            
            auto &parms = context_.get_context_data(plain.parms_id())->parms();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
          
            encrypt_zero_internal_ckks_asymmetric(plain.parms_id(), is_asymmetric, save_seed, destination, plain, pool);

            // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
            int threads_per_block = 256;
            int blocks_per_grid = (coeff_count * coeff_modulus_size + threads_per_block - 1) / threads_per_block;

            add_poly_coeffmod_kernel<<<blocks_per_grid, threads_per_block>>>(
                                destination.d_data(), plain.d_data(), coeff_count, coeff_modulus_size, d_coeff_modulus_value, destination.d_data());

            destination.scale() = plain.scale();
            
        }
        else if (scheme == scheme_type::bgv)
        {
            if (plain.is_ntt_form())
            {
                throw invalid_argument("plain cannot be in NTT form");
            }
            // 暂时先用这个
            encrypt_zero_internal_origin(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);

            // encrypt_zero_internal(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);
            // encrypt_zero_internal_bgv(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);

            auto &context_data = *context_.first_context_data();
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            size_t plain_coeff_count = plain.coeff_count();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            auto plain_upper_half_increment = context_data.plain_upper_half_increment();
            auto ntt_tables = iter(context_data.small_ntt_tables());

            // c_{0} = pk_{0}*u + p*e_{0} + M
            Plaintext plain_copy = plain;
            // Resize to fit the entire NTT transformed (ciphertext size) polynomial
            // Note that the new coefficients are automatically set to 0
            plain_copy.resize(coeff_count * coeff_modulus_size);
            RNSIter plain_iter(plain_copy.data(), coeff_count);
            if (!context_data.qualifiers().using_fast_plain_lift)
            {
                // Allocate temporary space for an entire RNS polynomial
                // Slight semantic misuse of RNSIter here, but this works well
                SEAL_ALLOCATE_ZERO_GET_RNS_ITER(temp, coeff_modulus_size, coeff_count, pool);

                SEAL_ITERATE(iter(plain_copy.data(), temp), plain_coeff_count, [&](auto I) {
                    auto plain_value = get<0>(I);
                    if (plain_value >= plain_upper_half_threshold)
                    {
                        add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
                    }
                    else
                    {
                        *get<1>(I) = plain_value;
                    }
                });

                context_data.rns_tool()->base_q()->decompose_array(temp, coeff_count, pool);

                // Copy data back to plain
                set_poly(temp, coeff_count, coeff_modulus_size, plain_copy.data());
            }
            else
            {
                // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the
                // coeff_modulus primes.

                // Create a "reversed" helper iterator that iterates in the reverse order both plain RNS components and
                // the plain_upper_half_increment values.
                auto helper_iter = reverse_iter(plain_iter, plain_upper_half_increment);
                advance(helper_iter, -safe_cast<ptrdiff_t>(coeff_modulus_size - 1));

                SEAL_ITERATE(helper_iter, coeff_modulus_size, [&](auto I) {
                    SEAL_ITERATE(iter(*plain_iter, get<0>(I)), plain_coeff_count, [&](auto J) {
                        get<1>(J) =
                            SEAL_COND_SELECT(get<0>(J) >= plain_upper_half_threshold, get<0>(J) + get<1>(I), get<0>(J));
                    });
                });
            }
            // Transform to NTT domain
            ntt_negacyclic_harvey(plain_iter, coeff_modulus_size, ntt_tables);

            // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
            RNSIter destination_iter = *iter(destination);
            add_poly_coeffmod(destination_iter, plain_iter, coeff_modulus_size, coeff_modulus, destination_iter);
            // destination.d_data_malloc(2*coeff_count*coeff_modulus_size);
            // cudaMemcpy(destination.d_data(), destination.data(), 2*coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
            destination.to_gpu();



        }
        else
        {
            throw invalid_argument("unsupported scheme");
        }
    }
} // namespace seal
