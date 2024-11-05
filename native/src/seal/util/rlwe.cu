// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/ciphertext.cuh"
#include "seal/randomgen.h"
#include "seal/randomtostd.h"
#include "seal/util/clipnormal.h"
#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/globals.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/polycore.h"
#include "seal/util/rlwe.h"
#include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

namespace seal
{
    namespace util
    {

        __device__ inline constexpr int hamming_weight_kernel(unsigned char value)
        {
            int t = static_cast<int>(value);
            t -= (t >> 1) & 0x55;
            t = (t & 0x33) + ((t >> 2) & 0x33);
            return (t + (t >> 4)) & 0x0F;
        }

        __device__ inline int32_t sample_one(uint32_t seed, uint32_t index)
        {
            curandState_t state;
            curand_init(seed, index, 0, &state);
            uint8_t random_byte;
            unsigned char x[6];
            for (size_t i = 0; i < 6; i++)
            {
                random_byte = curand(&state) * 256;
                x[i] = random_byte;
            }
            x[2] &= 0x1F;
            x[5] &= 0x1F;
            return hamming_weight_kernel(x[0]) + hamming_weight_kernel(x[1]) + hamming_weight_kernel(x[2]) -
                   hamming_weight_kernel(x[3]) - hamming_weight_kernel(x[4]) - hamming_weight_kernel(x[5]);
        }

        __global__ void sample_poly_ternary_kernel(
            curandState *state, unsigned long long seed, uint64_t coeff_count, uint64_t modulus_value, uint64_t *d_secret_key)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            // 生成随机数
            while (idx < coeff_count)
            {
                curand_init(seed, idx, 0, &state[idx]);
                uint64_t rand = curand(&state[idx]) % 3;
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(rand == 0));
                d_secret_key[idx] = rand + (flag & modulus_value) - 1;

                idx += blockDim.x * gridDim.x;
            }
        }

        __global__ void sample_poly_cbd_kernel(
            uint64_t *destination, uint32_t coeff_count, const uint64_t *coeff_modulus, uint32_t coeff_modulus_size)
        {
            uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t step = blockDim.x * gridDim.x;

            while (index < coeff_count)
            {
                uint32_t seed = index;
                uint32_t noise = sample_one(seed, index);
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
                for (size_t i = 0; i < coeff_modulus_size; ++i)
                {
                    destination[index + i * coeff_count] = static_cast<uint64_t>(noise) + (flag & coeff_modulus[i]);
                }

                index += step;
            }
        }

        void sample_poly_ternary(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            RandomToStandardAdapter engine(prng);
            uniform_int_distribution<uint64_t> dist(0, 2);

            SEAL_ITERATE(iter(destination), coeff_count, [&](auto &I) {
                uint64_t rand = dist(engine);
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(rand == 0));
                SEAL_ITERATE(
                    iter(StrideIter<uint64_t *>(&I, coeff_count), coeff_modulus), coeff_modulus_size,
                    [&](auto J) { *get<0>(J) = rand + (flag & get<1>(J).value()) - 1; });
            });
        }

        void sample_poly_normal(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            cout << "sample_poly_normal" << endl;
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            if (are_close(global_variables::noise_max_deviation, 0.0))
            {
                set_zero_poly(coeff_count, coeff_modulus_size, destination);
                return;
            }

            RandomToStandardAdapter engine(prng);
            ClippedNormalDistribution dist(
                0, global_variables::noise_standard_deviation, global_variables::noise_max_deviation);

            SEAL_ITERATE(iter(destination), coeff_count, [&](auto &I) {
                int64_t noise = static_cast<int64_t>(dist(engine));
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
                SEAL_ITERATE(
                    iter(StrideIter<uint64_t *>(&I, coeff_count), coeff_modulus), coeff_modulus_size,
                    [&](auto J) { *get<0>(J) = static_cast<uint64_t>(noise) + (flag & get<1>(J).value()); });
            });
        }

        void sample_poly_cbd(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            if (are_close(global_variables::noise_max_deviation, 0.0))
            {
                set_zero_poly(coeff_count, coeff_modulus_size, destination);
                return;
            }

            if (!are_close(global_variables::noise_standard_deviation, 3.2))
            {
                throw logic_error("centered binomial distribution only supports standard deviation 3.2; use rounded "
                                  "Gaussian instead");
            }

            auto cbd = [&]() {
                unsigned char x[6];
                prng->generate(6, reinterpret_cast<seal_byte *>(x));
                x[2] &= 0x1F;
                x[5] &= 0x1F;
                return hamming_weight(x[0]) + hamming_weight(x[1]) + hamming_weight(x[2]) - hamming_weight(x[3]) -
                       hamming_weight(x[4]) - hamming_weight(x[5]);
            };

            for (int j = 0; j < coeff_count; j++)
            {
                int32_t noise = cbd();
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
                for (size_t i = 0; i < coeff_modulus_size; ++i)
                {
                    *(destination + i * coeff_count + j) =
                        static_cast<uint64_t>(noise) + (flag & coeff_modulus[i].value());
                }
            }
        }

        void sample_poly_uniform(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            // Extract encryption parameters
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            size_t dest_byte_count = mul_safe(coeff_modulus_size, coeff_count, sizeof(uint64_t));

            constexpr uint64_t max_random = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);

            // Fill the destination buffer with fresh randomness
            prng->generate(dest_byte_count, reinterpret_cast<seal_byte *>(destination));

            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                auto &modulus = coeff_modulus[j];
                uint64_t max_multiple = max_random - barrett_reduce_64(max_random, modulus) - 1;
                // inplace transform
                transform(destination, destination + coeff_count, destination, [&](uint64_t rand) {
                    // This ensures uniform distribution
                    while (rand >= max_multiple)
                    {
                        prng->generate(sizeof(uint64_t), reinterpret_cast<seal_byte *>(&rand));
                    }
                    return barrett_reduce_64(rand, modulus);
                });
                destination += coeff_count;
            }
        }

        void sample_poly_uniform_seal_3_4(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            // Extract encryption parameters
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            RandomToStandardAdapter engine(prng);

            constexpr uint64_t max_random = static_cast<uint64_t>(0x7FFFFFFFFFFFFFFFULL);
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                auto &modulus = coeff_modulus[j];
                uint64_t max_multiple = max_random - barrett_reduce_64(max_random, modulus) - 1;
                for (size_t i = 0; i < coeff_count; i++)
                {
                    // This ensures uniform distribution
                    uint64_t rand;
                    do
                    {
                        rand = (static_cast<uint64_t>(engine()) << 31) | (static_cast<uint64_t>(engine()) >> 1);
                    } while (rand >= max_multiple);
                    destination[i + j * coeff_count] = barrett_reduce_64(rand, modulus);
                }
            }
        }

        void sample_poly_uniform_seal_3_5(
            shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms, uint64_t *destination)
        {
            // Extract encryption parameters
            auto coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            RandomToStandardAdapter engine(prng);

            constexpr uint64_t max_random = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                auto &modulus = coeff_modulus[j];
                uint64_t max_multiple = max_random - barrett_reduce_64(max_random, modulus) - 1;
                for (size_t i = 0; i < coeff_count; i++)
                {
                    // This ensures uniform distribution
                    uint64_t rand;
                    do
                    {
                        rand = (static_cast<uint64_t>(engine()) << 32) | static_cast<uint64_t>(engine());
                    } while (rand >= max_multiple);
                    destination[i + j * coeff_count] = barrett_reduce_64(rand, modulus);
                }
            }
        }

        __global__ void sub_modulu_kernel(
                uint64_t *input, uint64_t *output, uint64_t *d_modulu_value, uint64_t coeff_count, 
                uint64_t coeff_modulus_size){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size){
                if (input[idx] >= 2 * d_modulu_value[idx / coeff_count]){
                    output[idx] = input[idx] - 2 * d_modulu_value[idx / coeff_count];
                }
                if (output[idx] >= d_modulu_value[idx / coeff_count]){
                    output[idx] = input[idx] - d_modulu_value[idx / coeff_count];
                }
                idx += blockDim.x * gridDim.x;
            }
        }


        void encrypt_zero_asymmetric_ckks_test(
            const PublicKey &public_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
            uint64_t *d_destination)
        {
            MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            auto &plain_modulus = parms.plain_modulus();
            uint64_t *d_coeff_modulus = parms.d_coeff_modulus_value();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            uint64_t *d_modulu_value = parms.d_coeff_modulus_value();
            const int stream_num = context.num_streams();
            cudaStream_t *ntt_steam = context.stream_context();
            auto ntt_tables = context_data.small_ntt_tables();
            size_t encrypted_size = public_key.data().size();

            scheme_type type = parms.scheme();


            // Make destination have right size and parms_id
            // Ciphertext (c_0,c_1, ...)
            // c[j] = public_key[j] * u + e[j] in BFV/CKKS = public_key[j] * u + p * e[j] in BGV
            // where e[j] <-- chi, u <-- R_3

            uint64_t *d_random = nullptr;
            curandState *state = nullptr;

            allocate_gpu<uint64_t>(&d_random, coeff_count * coeff_modulus_size);
            allocate_gpu<curandState>(&state, coeff_count * coeff_modulus_size);

            uint64_t *d_public_key = public_key.data().d_data();

            // c[j] = u * public_key[j]
            std::random_device rd; 
            unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
            seed ^= static_cast<unsigned long long>(rd()) << 32; // rd()是一个随机数生成器
            for (int i = 0; i < coeff_modulus_size; i++)
            {

                sample_poly_ternary_kernel<<<(coeff_count + 255) / 256, 256>>>(
                    state, seed, coeff_count, coeff_modulus[i].value(), d_random + i * coeff_count);
            }

#if NTT_VERSION == 3
            ntt_v3(context, parms_id, d_random, coeff_modulus_size);
#else 
            ntt_v1(context, parms_id, d_random, coeff_modulus_size);
#endif
            
            sub_modulu_kernel<<<(coeff_count * coeff_modulus_size + 1023) / 1024, 1024>>>(
                d_random, d_random, d_modulu_value, coeff_count, coeff_modulus_size);

            const std::size_t threadsPerBlock = 256;
            const std::size_t blocksPerGrid = (coeff_count + threadsPerBlock - 1) / threadsPerBlock;
            for (size_t i = 0; i < coeff_modulus_size; i++)
            {                
                for (size_t j = 0; j < encrypted_size; j++)
                {

                    dyadic_product_coeffmod_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                        d_random + i * coeff_count, 
                        d_public_key + j * public_key.data().coeff_modulus_size() * coeff_count + i * coeff_count, 
                        coeff_count,
                        coeff_modulus[i].value(), 
                        coeff_modulus[i].const_ratio()[0], 
                        coeff_modulus[i].const_ratio()[1],
                        d_destination + j * coeff_modulus_size * coeff_count + i * coeff_count);
                }
            }

            // Generate e_j <-- chi
            // c[j] = public_key[j] * u + e[j] in BFV/CKKS, = public_key[j] * u + p * e[j] in BGV,
            for (size_t j = 0; j < encrypted_size; j++)
            {
                sample_poly_cbd_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_random, coeff_count, d_coeff_modulus, coeff_modulus_size);

                if (is_ntt_form)
                {
#if NTT_VERSION == 3
                    ntt_v3(context, parms_id, d_random, coeff_modulus_size);
#else 
                    ntt_v1(context, parms_id, d_random, coeff_modulus_size);
#endif
                sub_modulu_kernel<<<(coeff_count * coeff_modulus_size + 1023) / 1024, 1024>>>(
                    d_random, d_random, d_modulu_value, coeff_count, coeff_modulus_size);
                }

                eltwiseAddModKernel<<<(coeff_count * coeff_modulus_size + 1023) / 1024, 1024>>>(
                    d_destination + j * coeff_count * coeff_modulus_size, 
                    d_random, 
                    d_destination+ j * coeff_count * coeff_modulus_size, 
                    d_coeff_modulus, 
                    coeff_count, 
                    coeff_modulus_size);
            }

            deallocate_gpu<uint64_t>(&d_random, coeff_count * coeff_modulus_size);
            deallocate_gpu<curandState>(&state, coeff_count * coeff_modulus_size);
        }

        void encrypt_zero_asymmetric(
            const PublicKey &public_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
            Ciphertext &destination)
        {
// #ifdef SEAL_DEBUG
//             if (!is_valid_for(public_key, context))
//             {
//                 throw invalid_argument("public key is not valid for the encryption parameters");
//             }
// #endif
            // We use a fresh memory pool with `clear_on_destruction' enabled
            MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            auto &plain_modulus = parms.plain_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            auto ntt_tables = context_data.small_ntt_tables();
            size_t encrypted_size = public_key.data().size();
            scheme_type type = parms.scheme();

            // Make destination have right size and parms_id
            // Ciphertext (c_0,c_1, ...)
            destination.resize(context, parms_id, encrypted_size);
            // destination.resize_pure_gpu(context, parms_id, encrypted_size);
            destination.is_ntt_form() = is_ntt_form;
            destination.scale() = 1.0;
            destination.correction_factor() = 1;

            // c[j] = public_key[j] * u + e[j] in BFV/CKKS = public_key[j] * u + p * e[j] in BGV
            // where e[j] <-- chi, u <-- R_3

            // Create a PRNG; u and the noise/error share the same PRNG
            auto prng = parms.random_generator()->create();

            // Generate u <-- R_3
            auto u(allocate_poly(coeff_count, coeff_modulus_size, pool));
            sample_poly_ternary(prng, parms, u.get());

            // c[j] = u * public_key[j]
            for (size_t i = 0; i < coeff_modulus_size; i++)
            {
                ntt_negacyclic_harvey(u.get() + i * coeff_count, ntt_tables[i]);
                for (size_t j = 0; j < encrypted_size; j++)
                {
                    dyadic_product_coeffmod(
                        u.get() + i * coeff_count, 
                        public_key.data().data(j) + i * coeff_count, 
                        coeff_count,
                        coeff_modulus[i], destination.data(j) + i * coeff_count);

                    // Addition with e_0, e_1 is in non-NTT form
                    if (!is_ntt_form)
                    {
                        inverse_ntt_negacyclic_harvey(destination.data(j) + i * coeff_count, ntt_tables[i]);
                    }
                }
            }

            // Generate e_j <-- chi
            // c[j] = public_key[j] * u + e[j] in BFV/CKKS, = public_key[j] * u + p * e[j] in BGV,
            for (size_t j = 0; j < encrypted_size; j++)
            {
                SEAL_NOISE_SAMPLER(prng, parms, u.get());
                RNSIter gaussian_iter(u.get(), coeff_count);

                // In BGV, p * e is used
                if (type == scheme_type::bgv)
                {
                    if (is_ntt_form)
                    {
                        ntt_negacyclic_harvey_lazy(gaussian_iter, coeff_modulus_size, ntt_tables);
                    }
                    multiply_poly_scalar_coeffmod(
                        gaussian_iter, coeff_modulus_size, plain_modulus.value(), coeff_modulus, gaussian_iter);
                }
                else
                {
                    if (is_ntt_form)
                    {
                        ntt_negacyclic_harvey(gaussian_iter, coeff_modulus_size, ntt_tables);
                    }
                }
                RNSIter dst_iter(destination.data(j), coeff_count);
                add_poly_coeffmod(gaussian_iter, dst_iter, coeff_modulus_size, coeff_modulus, dst_iter);
            }

       }

        void encrypt_zero_symmetric(
            const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id, bool is_ntt_form,
            bool save_seed, Ciphertext &destination)
        {
#ifdef SEAL_DEBUG
            if (!is_valid_for(secret_key, context))
            {
                throw invalid_argument("secret key is not valid for the encryption parameters");
            }
#endif
            // We use a fresh memory pool with `clear_on_destruction' enabled.
            MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            auto &plain_modulus = parms.plain_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            auto ntt_tables = context_data.small_ntt_tables();
            size_t encrypted_size = 2;
            scheme_type type = parms.scheme();

            // If a polynomial is too small to store UniformRandomGeneratorInfo,
            // it is best to just disable save_seed. Note that the size needed is
            // the size of UniformRandomGeneratorInfo plus one (uint64_t) because
            // of an indicator word that indicates a seeded ciphertext.
            size_t poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);
            size_t prng_info_byte_count =
                static_cast<size_t>(UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none));
            size_t prng_info_uint64_count =
                divide_round_up(prng_info_byte_count, static_cast<size_t>(bytes_per_uint64));
            if (save_seed && poly_uint64_count < prng_info_uint64_count + 1)
            {
                save_seed = false;
            }

            destination.resize(context, parms_id, encrypted_size);
            destination.is_ntt_form() = is_ntt_form;
            destination.scale() = 1.0;
            destination.correction_factor() = 1;

            // Create an instance of a random number generator. We use this for sampling
            // a seed for a second PRNG used for sampling u (the seed can be public
            // information. This PRNG is also used for sampling the noise/error below.
            auto bootstrap_prng = parms.random_generator()->create();

            // Sample a public seed for generating uniform randomness
            prng_seed_type public_prng_seed;
            bootstrap_prng->generate(prng_seed_byte_count, reinterpret_cast<seal_byte *>(public_prng_seed.data()));

            // Set up a new default PRNG for expanding u from the seed sampled above
            auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create(public_prng_seed);

            // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
            // Generate ciphertext: (c[0], c[1]) = ([-(as+pe)]_q, a) in BGV
            uint64_t *c0 = destination.data();
            uint64_t *c1 = destination.data(1);

            // Sample a uniformly at random
            if (is_ntt_form || !save_seed)
            {
                // Sample the NTT form directly
                sample_poly_uniform(ciphertext_prng, parms, c1);
            }
            else if (save_seed)
            {
                // Sample non-NTT form and store the seed
                sample_poly_uniform(ciphertext_prng, parms, c1);
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    // Transform the c1 into NTT representation
                    ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
                }
            }

            // Sample e <-- chi
            auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
            SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());

            // Calculate -(as+ e) (mod q) and store in c[0] in BFV/CKKS
            // Calculate -(as+pe) (mod q) and store in c[0] in BGV
            for (size_t i = 0; i < coeff_modulus_size; i++)
            {
                dyadic_product_coeffmod(
                    secret_key.data().data() + i * coeff_count, c1 + i * coeff_count, coeff_count, coeff_modulus[i],
                    c0 + i * coeff_count);
                if (is_ntt_form)
                {
                    // Transform the noise e into NTT representation
                    ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);
                }
                else
                {
                    inverse_ntt_negacyclic_harvey(c0 + i * coeff_count, ntt_tables[i]);
                }

                if (type == scheme_type::bgv)
                {
                    // noise = pe instead of e in BGV
                    multiply_poly_scalar_coeffmod(
                        noise.get() + i * coeff_count, coeff_count, plain_modulus.value(), coeff_modulus[i],
                        noise.get() + i * coeff_count);
                }

                // c0 = as + noise
                add_poly_coeffmod(
                    noise.get() + i * coeff_count, c0 + i * coeff_count, coeff_count, coeff_modulus[i],
                    c0 + i * coeff_count);
                // (as + noise, a) -> (-(as + noise), a),
                negate_poly_coeffmod(c0 + i * coeff_count, coeff_count, coeff_modulus[i], c0 + i * coeff_count);
            }

            if (!is_ntt_form && !save_seed)
            {
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    // Transform the c1 into non-NTT representation
                    inverse_ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
                }
            }

            if (save_seed)
            {
                UniformRandomGeneratorInfo prng_info = ciphertext_prng->info();

                // Write prng_info to destination.data(1) after an indicator word
                c1[0] = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
                prng_info.save(reinterpret_cast<seal_byte *>(c1 + 1), prng_info_byte_count, compr_mode_type::none);
            }
        }
        
        void encrypt_zero_symmetric_cuda(const SEALContext &context, uint64_t *secret_key,  parms_id_type parms_id, bool is_ntt_form,
                    bool save_seed, Ciphertext &destination){
// #ifdef SEAL_DEBUG
//             if (!is_valid_for(secret_key, context))
//             {
//                 throw invalid_argument("secret key is not valid for the encryption parameters");
//             }
// #endif
            // We use a fresh memory pool with `clear_on_destruction' enabled.
            MemoryPoolHandle pool = MemoryManager::GetPool(mm_prof_opt::mm_force_new, true);

            auto &context_data = *context.get_context_data(parms_id);
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            auto &plain_modulus = parms.plain_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            auto ntt_tables = context_data.small_ntt_tables();
            size_t encrypted_size = 2;
            scheme_type type = parms.scheme();


            uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();

            const int stream_num = context.num_streams();
            cudaStream_t *ntt_steam = context.stream_context();
            // If a polynomial is too small to store UniformRandomGeneratorInfo,
            // it is best to just disable save_seed. Note that the size needed is
            // the size of UniformRandomGeneratorInfo plus one (uint64_t) because
            // of an indicator word that indicates a seeded ciphertext.
            size_t poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);
            size_t prng_info_byte_count =
                static_cast<size_t>(UniformRandomGeneratorInfo::SaveSize(compr_mode_type::none));
            size_t prng_info_uint64_count =
                divide_round_up(prng_info_byte_count, static_cast<size_t>(bytes_per_uint64));
            if (save_seed && poly_uint64_count < prng_info_uint64_count + 1)
            {
                save_seed = false;
            }

            destination.resize(context, parms_id, encrypted_size);
            destination.resize_pure_gpu(context, parms_id, encrypted_size);
            destination.is_ntt_form() = is_ntt_form;
            destination.scale() = 1.0;
            destination.correction_factor() = 1;

            // Create an instance of a random number generator. We use this for sampling
            // a seed for a second PRNG used for sampling u (the seed can be public
            // information. This PRNG is also used for sampling the noise/error below.
            auto bootstrap_prng = parms.random_generator()->create();

            // Sample a public seed for generating uniform randomness
            prng_seed_type public_prng_seed;
            bootstrap_prng->generate(prng_seed_byte_count, reinterpret_cast<seal_byte *>(public_prng_seed.data()));

            // Set up a new default PRNG for expanding u from the seed sampled above
            auto ciphertext_prng = UniformRandomGeneratorFactory::DefaultFactory()->create(public_prng_seed);

            // Generate ciphertext: (c[0], c[1]) = ([-(as+ e)]_q, a) in BFV/CKKS
            // Generate ciphertext: (c[0], c[1]) = ([-(as+pe)]_q, a) in BGV
            uint64_t *c0 = destination.data();
            uint64_t *c1 = destination.data(1);

            // Sample a uniformly at random
            if (is_ntt_form || !save_seed)
            {
                // Sample the NTT form directly
                sample_poly_uniform(ciphertext_prng, parms, c1);
            }
            else if (save_seed)
            {
                // Sample non-NTT form and store the seed
                sample_poly_uniform(ciphertext_prng, parms, c1);
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    // Transform the c1 into NTT representation
                    ntt_negacyclic_harvey(c1 + i * coeff_count, ntt_tables[i]);
                }
            }

            // Sample e <-- chi
            auto noise(allocate_poly(coeff_count, coeff_modulus_size, pool));
            SEAL_NOISE_SAMPLER(bootstrap_prng, parms, noise.get());


            uint64_t *d_noise = nullptr;
            allocate_gpu<uint64_t>(&d_noise, coeff_count * coeff_modulus_size);
            checkCudaErrors(cudaMemcpy(d_noise, noise.get(), coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice));


            checkCudaErrors(cudaMemcpy(destination.d_data() + coeff_count * coeff_modulus_size, destination.data() + coeff_count * coeff_modulus_size ,  coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

            // Calculate -(as+ e) (mod q) and store in c[0] in BFV/CKKS
            // Calculate -(as+pe) (mod q) and store in c[0] in BGV
            for (size_t i = 0; i < coeff_modulus_size; i++)
            {
                
                dyadic_product_coeffmod_kernel<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[i % stream_num]>>>(
                    secret_key + i * coeff_count, 
                    destination.d_data() + coeff_count * coeff_modulus_size + i * coeff_count, 
                    coeff_count, 
                    coeff_modulus[i].value(),
                    coeff_modulus[i].const_ratio()[0], 
                    coeff_modulus[i].const_ratio()[1],
                    destination.d_data() + i * coeff_count);

                if (is_ntt_form)
                {
                    // Transform the noise e into NTT representation
                    ntt_v1_single(context, parms_id, d_noise + i * coeff_count, i, ntt_steam[i % stream_num]);

                }
                else
                {
                    intt_v1_single(context, parms_id, destination.d_data() + i * coeff_count, i, ntt_steam[i % stream_num]);
                }



                if (type == scheme_type::bgv)
                {
                    // noise = pe instead of e in BGV
                    multiply_poly_scalar_coeffmod_kernel_one_modulu<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[i % stream_num]>>>(
                        d_noise, d_noise, coeff_count, coeff_modulus[i].value(), coeff_modulus[i].const_ratio().data()[0], plain_modulus.value()) ;
                }

                // c0 = as + noise
                add_poly_coeffmod_kernel<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[i % stream_num]>>>(
                    d_noise + i * coeff_count, 
                    destination.d_data() + i * coeff_count, 
                    coeff_count, coeff_modulus[i].value(),
                    destination.d_data() + i * coeff_count);

                // (as + noise, a) -> (-(as + noise), a),
                negate_poly_coeffmod_cuda<<<(coeff_count + 255) / 256, 256, 0, ntt_steam[i % stream_num]>>>(
                    destination.d_data() + i * coeff_count, 
                    parms.d_coeff_modulus_value() + i,
                    destination.d_data() + i * coeff_count,
                    coeff_count, 1,
                    coeff_count
                    );
            }


            if (!is_ntt_form && !save_seed)
            {
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    // Transform the c1 into non-NTT representation
                    intt_v1_single(context, parms_id, destination.d_data() + coeff_count * coeff_modulus_size + i * coeff_count, i, ntt_steam[i % stream_num]);
                }
            }

            // cudaMemcpy(destination.data() , 
            //     destination.d_data(), coeff_count * coeff_modulus_size * 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            if (save_seed)
            {
                UniformRandomGeneratorInfo prng_info = ciphertext_prng->info();

                // Write prng_info to destination.data(1) after an indicator word
                c1[0] = static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL);
                prng_info.save(reinterpret_cast<seal_byte *>(c1 + 1), prng_info_byte_count, compr_mode_type::none);
            }
            deallocate_gpu<uint64_t>(&d_noise, coeff_count * coeff_modulus_size);
        }

    } // namespace util
} // namespace seal
