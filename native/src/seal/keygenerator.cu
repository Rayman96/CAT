// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.
#include "seal/keygenerator.h"
#include "seal/randomtostd.h"
#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/galois.h"
#include "seal/util/ntt.h"
#include "seal/util/polycore.h"
#include "seal/util/rlwe.h"
#include "seal/util/uintarithsmallmod.cuh"
#include "seal/util/uintcore.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#define USE_GPU false
using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        __global__ void sample_poly_ternary_kernel(
            curandState *state, unsigned long long seed, uint64_t coeff_count, uint64_t coeff_modulus_size, uint64_t *d_coeff_modulus,
            uint64_t *d_secret_key)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            // 生成随机数
            while (idx < coeff_count * coeff_modulus_size)
            {
                curand_init(seed, idx, 0, &state[idx]);
                uint64_t rand = curand(&state[idx]) % 3;
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(rand == 0));
                int64_t k = static_cast<int64_t>(idx / coeff_count);
                d_secret_key[idx] = rand + (flag & d_coeff_modulus[k]) - 1;

                idx += blockDim.x * gridDim.x;
            }
        }
    
        __global__ void multiply_poly_scalar_coeffmod_kernel(
            uint64_t *poly, size_t coeff_count, uint64_t operand, uint64_t quotient, const uint64_t modulus_value,
            uint64_t *result)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < coeff_count){
                unsigned long long tmp1, tmp2;
                multiply_uint64_hw64_kernel(poly[idx], quotient, &tmp1);
                tmp2 = operand * poly[idx] - tmp1 * modulus_value;
                result[idx] = tmp2 >= modulus_value ? tmp2 - modulus_value : tmp2;
                idx += blockDim.x * gridDim.x;
            }

        }
    } // namespace

    KeyGenerator::KeyGenerator(const SEALContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        // Secret key has not been generated
        sk_generated_ = false;
        // Generate the secret and public key
        generate_sk();
    }

    KeyGenerator::KeyGenerator(const SEALContext &context, const SecretKey &secret_key) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        if (!is_valid_for(secret_key, context_))
        {
            throw invalid_argument("secret key is not valid for encryption parameters");
        }

        // Set the secret key
        secret_key_ = secret_key;
        sk_generated_ = true;

        // Generate the public key
        generate_sk(sk_generated_);
    }

    void KeyGenerator::generate_sk(bool is_initialized)
    {
        // Extract encryption parameters.
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        if (!is_initialized)
        {
            // Initialize secret key.
            secret_key_ = SecretKey();
            sk_generated_ = false;
            secret_key_.data().resize(mul_safe(coeff_count, coeff_modulus_size));
            secret_key_.data().resize_gpu(mul_safe(coeff_count, coeff_modulus_size));
            // Generate secret key
            RNSIter secret_key(secret_key_.data().data(), coeff_count);
            sample_poly_ternary(parms.random_generator()->create(), parms, secret_key);

            // Transform the secret s into NTT representation.
            auto ntt_tables = context_data.small_ntt_tables();
            ntt_negacyclic_harvey(secret_key, coeff_modulus_size, ntt_tables);

            // Set the parms_id for secret key
            secret_key_.parms_id() = context_data.parms_id();
            checkCudaErrors(cudaMemcpy(secret_key_.data().d_data(), secret_key_.data().data(), mul_safe(coeff_count, coeff_modulus_size) * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

        // Set the secret_key_array to have size 1 (first power of secret)
        secret_key_array_ = allocate_poly(coeff_count, coeff_modulus_size, pool_);        
        set_poly(secret_key_.data().data(), coeff_count, coeff_modulus_size, secret_key_array_.get());
        secret_key_array_size_ = 1;

        // Secret key has been generated
        sk_generated_ = true;
    }

    void KeyGenerator::compute_secret_key_array(const SEALContext::ContextData &context_data, size_t max_power)
    {
#ifdef SEAL_DEBUG
        if (max_power < 1)
        {
            throw invalid_argument("max_power must be at least 1");
        }
        if (!secret_key_array_size_ || !secret_key_array_)
        {
            throw logic_error("secret_key_array_ is uninitialized");
        }
#endif

        // Extract encryption parameters.
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size, max_power))
        {
            throw logic_error("invalid parameters");
        }

        ReaderLock reader_lock(secret_key_array_locker_.acquire_read());

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);

        if (old_size == new_size)
        {
            return;
        }

        reader_lock.unlock();

        // Need to extend the array
        // Compute powers of secret key until max_power
        auto secret_key_array(allocate_poly_array(new_size, coeff_count, coeff_modulus_size, pool_));
        set_poly_array(secret_key_array_.get(), old_size, coeff_count, coeff_modulus_size, secret_key_array.get());
        RNSIter secret_key(secret_key_array.get(), coeff_count);

        PolyIter secret_key_power(secret_key_array.get(), coeff_count, coeff_modulus_size);
        secret_key_power += (old_size - 1);
        auto next_secret_key_power = secret_key_power + 1;

        size_t dest_size = coeff_modulus_size * coeff_count;
        size_t new_old_dist = new_size - old_size;
        // malloc
        uint64_t *d_secret_key = nullptr;
        uint64_t *d_secret_key_power = nullptr;
        uint64_t *d_next_secret_key_power = nullptr;
 
        allocate_gpu<uint64_t>(&d_secret_key, new_old_dist * coeff_modulus_size * coeff_count);
        allocate_gpu<uint64_t>(&d_secret_key_power, new_old_dist * coeff_modulus_size * coeff_count);
        allocate_gpu<uint64_t>(&d_next_secret_key_power, new_old_dist * coeff_modulus_size * coeff_count);
        
        checkCudaErrors(cudaMemcpy(
            d_secret_key, secret_key_array.get(), new_old_dist * coeff_modulus_size * coeff_count * sizeof(uint64_t),
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
            d_secret_key_power, secret_key_array.get() + (old_size - 1) * dest_size,
            new_old_dist * coeff_modulus_size * coeff_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // kernel
        for (size_t i = 0; i < new_old_dist * coeff_modulus_size; i++)
        {
            dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_secret_key_power + i * coeff_count, d_secret_key + i * coeff_count, coeff_count,
                coeff_modulus[i].value(), coeff_modulus[i].const_ratio()[0], coeff_modulus[i].const_ratio()[1],
                d_next_secret_key_power + i * coeff_count);
        }
        // copy back
        checkCudaErrors(cudaMemcpy(
            secret_key_array.get() + old_size * dest_size, d_next_secret_key_power,
            new_old_dist * coeff_modulus_size * coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // free
        deallocate_gpu<uint64_t>(&d_secret_key, new_old_dist * coeff_modulus_size * coeff_count);
        deallocate_gpu<uint64_t>(&d_secret_key_power, new_old_dist * coeff_modulus_size * coeff_count);
        deallocate_gpu<uint64_t>(&d_next_secret_key_power, new_old_dist * coeff_modulus_size * coeff_count);

        // == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

        // Take writer lock to update array
        WriterLock writer_lock(secret_key_array_locker_.acquire_write());

        // Do we still need to update size?
        old_size = secret_key_array_size_;
        new_size = max(max_power, secret_key_array_size_);

        if (old_size == new_size)
        {
            return;
        }

        // Acquire new array
        secret_key_array_size_ = new_size;
        secret_key_array_.acquire(secret_key_array);
    }

    void KeyGenerator::generate_one_kswitch_key(ConstRNSIter new_key, vector<PublicKey> &destination, bool save_seed)
    {
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }
        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        size_t decomp_mod_count = context_.first_context_data()->parms().coeff_modulus().size();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, decomp_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // malloc
        uint64_t *d_destination = nullptr;
        uint64_t *d_new_key = nullptr;
        uint64_t *d_temp = nullptr;

        allocate_gpu<uint64_t>(&d_new_key, decomp_mod_count * coeff_count);
        allocate_gpu<uint64_t>(&d_temp, coeff_count);

        cudaMemcpy(
            d_new_key, (*new_key).ptr(), decomp_mod_count * coeff_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // kernel
        destination.resize(decomp_mod_count);
        // printf("goes generate_one_kswitch_key\n");
        for (size_t i = 0; i < decomp_mod_count; i++)
        {

            uint64_t factor = barrett_reduce_64(key_modulus.back().value(), key_modulus[i]);
            MultiplyUIntModOperand temp_scalar;
            temp_scalar.set(barrett_reduce_64(factor, key_modulus[i]), key_modulus[i]);

            
            encrypt_zero_symmetric_cuda(
                 context_, secret_key_.data().d_data(), key_context_data.parms_id(), true, save_seed, destination[i].data());
            
            d_destination = destination[i].data().d_data() + i * coeff_count;

            multiply_poly_scalar_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_new_key + i * coeff_count, coeff_count, temp_scalar.operand, temp_scalar.quotient,
                key_modulus[i].value(), d_temp);

            add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_destination , d_temp, coeff_count, key_modulus[i].value(),
                d_destination );

        }
        deallocate_gpu<uint64_t>(&d_new_key, decomp_mod_count * coeff_count);
        deallocate_gpu<uint64_t>(&d_temp, coeff_count);

    }

    void KeyGenerator::generate_one_kswitch_key_cuda(uint64_t *new_key, vector<PublicKey> &destination, bool save_seed)
    {
        if (!context_.using_keyswitching())
        {
            throw logic_error("keyswitching is not supported by the context");
        }
        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        size_t decomp_mod_count = context_.first_context_data()->parms().coeff_modulus().size();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        auto &key_modulus = key_parms.coeff_modulus();
        size_t key_modulus_size = key_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, decomp_mod_count))
        {
            throw logic_error("invalid parameters");
        }

        // malloc
        uint64_t *d_destination = nullptr;
        uint64_t *d_temp = nullptr;
        allocate_gpu<uint64_t>(&d_temp, coeff_count);

        uint64_t *d_new_key = new_key;

        // kernel
        destination.resize(decomp_mod_count);
        for (size_t i = 0; i < decomp_mod_count; i++)
        {

            uint64_t factor = barrett_reduce_64(key_modulus.back().value(), key_modulus[i]);
            MultiplyUIntModOperand temp_scalar;
            temp_scalar.set(barrett_reduce_64(factor, key_modulus[i]), key_modulus[i]);

            
            encrypt_zero_symmetric_cuda(
                 context_, secret_key_.data().d_data(), key_context_data.parms_id(), true, save_seed, destination[i].data());
            
            d_destination = destination[i].data().d_data() + i * coeff_count;

            multiply_poly_scalar_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_new_key + i * coeff_count, coeff_count, temp_scalar.operand, temp_scalar.quotient,
                key_modulus[i].value(), d_temp);

            add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_destination , d_temp, coeff_count, key_modulus[i].value(),
                d_destination );

        }
        deallocate_gpu<uint64_t>(&d_temp, coeff_count);

    }

    PublicKey KeyGenerator::generate_pk(bool save_seed) const
    {
        if (!sk_generated_)
        {
            throw logic_error("cannot generate public key for unspecified secret key");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        PublicKey public_key;    
        encrypt_zero_symmetric_cuda(
                context_, secret_key_.data().d_data(), context_data.parms_id(), true, save_seed, public_key.data());
        // Set the parms_id for public key
        public_key.parms_id() = context_data.parms_id();

        return public_key;
    }

    RelinKeys KeyGenerator::create_relin_keys(size_t count, bool save_seed)
    {
        // Check to see if secret key and public key have been generated
        if (!sk_generated_)
        {
            throw logic_error("cannot generate relinearization keys for unspecified secret key");
        }
        if (!count || count > SEAL_CIPHERTEXT_SIZE_MAX - 2)
        {
            throw invalid_argument("invalid count");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = parms.coeff_modulus().size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size))
        {
            throw logic_error("invalid parameters");
        }

        // Make sure we have enough secret keys computed
        compute_secret_key_array(context_data, count + 1);

        // Create the RelinKeys object to return
        RelinKeys relin_keys;

        // Assume the secret key is already transformed into NTT form.
        ConstPolyIter secret_key(secret_key_array_.get(), coeff_count, coeff_modulus_size);
        generate_kswitch_keys(secret_key + 1, count, static_cast<KSwitchKeys &>(relin_keys), save_seed);

        // Set the parms_id
        relin_keys.parms_id() = context_data.parms_id();

        return relin_keys;
    }

    GaloisKeys KeyGenerator::create_galois_keys(const vector<uint32_t> &galois_elts, bool save_seed)
    {
        // Check to see if secret key and public key have been generated
        if (!sk_generated_)
        {
            throw logic_error("cannot generate Galois keys for unspecified secret key");
        }

        // Extract encryption parameters.
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto galois_tool = context_data.galois_tool();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }

        // Create the GaloisKeys object to return
        GaloisKeys galois_keys;

        // The max number of keys is equal to number of coefficients
        galois_keys.data().resize(coeff_count);

        uint64_t *d_rotate_sceret_key = nullptr;
        allocate_gpu<uint64_t>(&d_rotate_sceret_key, coeff_count * coeff_modulus_size);
        for (auto galois_elt : galois_elts)
        {
            // Verify coprime conditions.
            if (!(galois_elt & 1) || (galois_elt >= coeff_count << 1))
            {
                throw invalid_argument("Galois element is not valid");
            }

            // Do we already have the key?
            if (galois_keys.has_key(galois_elt))
            {
                continue;
            }

            // Rotate secret key for each coeff_modulus
            galois_tool->apply_galois_ntt_batch_cuda(secret_key_.data().d_data(), 
                                                    galois_elt,  
                                                    d_rotate_sceret_key,
                                                    coeff_modulus_size);

            // Initialize Galois key
            // This is the location in the galois_keys vector
            size_t index = GaloisKeys::get_index(galois_elt);
            // Create Galois keys.
            generate_one_kswitch_key_cuda(d_rotate_sceret_key, galois_keys.data()[index], save_seed);
        }

        // Set the parms_id
        galois_keys.parms_id_ = context_data.parms_id();
        deallocate_gpu<uint64_t>(&d_rotate_sceret_key, coeff_count * coeff_modulus_size);
        return galois_keys;
    }

    const SecretKey &KeyGenerator::secret_key() const
    {
        if (!sk_generated_)
        {
            throw logic_error("secret key has not been generated");
        }
        return secret_key_;
    }

    void KeyGenerator::generate_kswitch_keys(
        ConstPolyIter new_keys, size_t num_keys, KSwitchKeys &destination, bool save_seed)
    {
        size_t coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        auto &key_context_data = *context_.key_context_data();
        auto &key_parms = key_context_data.parms();
        size_t coeff_modulus_size = key_parms.coeff_modulus().size();

        // Size check
        if (!product_fits_in(coeff_count, coeff_modulus_size, num_keys))
        {
            throw logic_error("invalid parameters");
        }
#ifdef SEAL_DEBUG
        if (new_keys.poly_modulus_degree() != coeff_count)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
        if (new_keys.coeff_modulus_size() != coeff_modulus_size)
        {
            throw invalid_argument("iterator is incompatible with encryption parameters");
        }
#endif
        destination.data().resize(num_keys);
        SEAL_ITERATE(iter(new_keys, destination.data()), num_keys, [&](auto I) {
            this->generate_one_kswitch_key(get<0>(I), get<1>(I), save_seed);
        });
    }

} // namespace seal
