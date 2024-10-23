
#include "seal/decryptor.h"
#include "seal/valcheck.h"
#include "seal/util/common.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/polycore.h"
#include "seal/util/scalingvariant.h"
#include "seal/util/uintarith.cuh"
#include "seal/util/uintcore.h"
#include "seal/util/rlwe.h"
#include <algorithm>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdexcept>
using namespace std;
using namespace seal::util;

namespace seal
{

    namespace
    {
        __global__ void compute_secret_key_array_kernel(
            uint64_t *operand1, uint64_t *operand2, uint64_t coeff_count, uint64_t coeff_modulus_size,
            uint64_t *coeff_modulus, uint64_t *coeff_modulus_r0, uint64_t *coeff_modulus_r1, uint64_t *result)
        {
            // printf("======= in compute_secret_key_array_kernel ==\n");
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while(idx < coeff_count * coeff_modulus_size){

                int k = (idx / coeff_count) % coeff_modulus_size;

                const uint64_t modulus_value = coeff_modulus[k];
                const uint64_t const_ratio_0 = coeff_modulus_r0[k];
                const uint64_t const_ratio_1 = coeff_modulus_r1[k];

                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
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

        __global__ void decrypt_ntt_size2_kernel(
            uint64_t *operand1, uint64_t *operand2, size_t coeff_count, size_t coeff_modulus_size, uint64_t *modulus,
            uint64_t *modulus_ratio_0, uint64_t *modulus_ratio_1, uint64_t *result)
        {
            // operand1: encrypted_data, operand2: secret_key_array, result: destination
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            while (idx < coeff_count * coeff_modulus_size)
            {
                const size_t idx_product = idx + coeff_count * coeff_modulus_size;
                const uint64_t modulus_value = modulus[(idx / coeff_count) % coeff_modulus_size];
                const uint64_t const_ratio_0 = modulus_ratio_0[(idx / coeff_count) % coeff_modulus_size];
                const uint64_t const_ratio_1 = modulus_ratio_1[(idx / coeff_count) % coeff_modulus_size];

                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64_kernel2(operand1[idx_product], operand2[idx], z);
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

                // --------------
                uint64_t sum = result[idx] + operand1[idx];
                result[idx] = sum >= modulus_value ? sum - modulus_value : sum;

                idx += blockDim.x * gridDim.x;
            }
        }
    } // namespace

    void Decryptor::bgv_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        auto &plain_modulus = parms.plain_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t c_vec_size = mul_safe(coeff_count, coeff_modulus_size);
        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();
        auto ntt_tables = context_data.small_ntt_tables();

        uint64_t *d_tmp_dest_modq = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_tmp_dest_modq, c_vec_size * sizeof(uint64_t)));
        dot_product_ct_sk_array_cuda(encrypted, d_tmp_dest_modq, pool_);

        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

        cudaStream_t ntt = 0;
        uint64_t temp_mu;
        for (int i = 0; i < coeff_modulus_size; i++)
        {
            k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[i].bit_count() * 2);
            temp_mu = (mu1 / coeff_modulus[i].value()).low;
            inverseNTT(
                d_tmp_dest_modq + coeff_count * i, coeff_count, ntt, coeff_modulus[i].value(), temp_mu,
                coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);
        }
        context_data.rns_tool()->decrypt_modt_cuda(d_tmp_dest_modq, destination.d_data(), coeff_count);

        if (encrypted.correction_factor() != 1)
        {
            uint64_t fix = 1;
            if (!try_invert_uint_mod(encrypted.correction_factor(), plain_modulus, fix))
            {
                throw logic_error("invalid correction factor");
            }

            multiply_poly_scalar_coeffmod_kernel_one_modulu<<<(coeff_count + 255) / 256, 256>>>(destination.d_data(), 
                                                                                                destination.d_data(), 
                                                                                                coeff_count,
                                                                                                plain_modulus.value(), 
                                                                                                plain_modulus.const_ratio().data()[1], fix);
        }
        checkCudaErrors(cudaMemcpy(destination.data(), destination.d_data(), coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // How many non-zero coefficients do we really have in the result?
        size_t plain_coeff_count = get_significant_uint64_count_uint(destination.data(), coeff_count);

        // Resize destination to appropriate size
        destination.resize(max(plain_coeff_count, size_t(1)));

        // destination.to_cpu();
    }

    void Decryptor::compute_secret_key_array(std::size_t max_power, uint64_t *d_secret_key_array)
    {
        // WARNING: This function must be called with the original context_data
        auto &context_data = *context_.key_context_data();
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // ReaderLock reader_lock(secret_key_array_locker_.acquire_read());

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);

        if (old_size == new_size)
        {
            return;
        }

        // reader_lock.unlock();

        // Need to extend the array
        // Compute powers of secret key until max_power
        size_t rns_size = mul_safe(coeff_count, coeff_modulus_size);
        size_t old_poly_size = mul_safe(old_size, coeff_count, coeff_modulus_size);
        size_t new_poly_size = mul_safe(new_size, coeff_count, coeff_modulus_size);

        // kernel
        for (size_t i = 0; i < (new_size - old_size) * coeff_modulus_size; ++i)
        {
            i = i % coeff_modulus_size;
            dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                d_secret_key_array + (old_size - 1) * rns_size + i * coeff_count, d_secret_key_array + i * coeff_count,
                coeff_count, coeff_modulus[i].value(), coeff_modulus[i].const_ratio()[0],
                coeff_modulus[i].const_ratio()[1], d_secret_key_array + old_size * rns_size + i * coeff_count);
        }

        // WriterLock writer_lock(secret_key_array_locker_.acquire_write());
        old_size = secret_key_array_size_;
        new_size = max(max_power, secret_key_array_size_);

        if (old_size == new_size)
        {
            return;
        }

        // Acquire new array
        secret_key_array_size_ = new_size;

    }

    // Compute c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q.
    // Store result in destination in RNS form.
    void Decryptor::dot_product_ct_sk_array(const Ciphertext &encrypted, RNSIter destination, MemoryPoolHandle pool)
    {
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t key_coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        size_t key_coeff_modulus_size = context_.key_context_data()->parms().coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        size_t max_power = encrypted_size - 1;
        auto is_ntt_form = encrypted.is_ntt_form();

        auto ntt_tables = context_data.small_ntt_tables();
        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();

        // === start for ntt ===============================================================
        // malloc
        cudaStream_t ntt = 0;

        uint64_t mu[2 * coeff_modulus_size];
        for (int i = 0; i < coeff_modulus_size; i++)
        {
            k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[i].bit_count() * 2);
            mu[i] = (mu1 / coeff_modulus[i].value()).low;
        }
        // === end for ntt ===============================================================

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);
        size_t old_poly_size = mul_safe(old_size, coeff_count, key_coeff_modulus_size);
        size_t new_poly_size = mul_safe(new_size, coeff_count, key_coeff_modulus_size);

        uint64_t *d_secret_key_array = nullptr;
        uint64_t *d_destination = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_secret_key_array, new_poly_size * sizeof(uint64_t)));
        checkCudaErrors(cudaMalloc((void **)&d_destination, coeff_modulus_size * coeff_count * sizeof(uint64_t)));
        // cudaMemcpy(
        //     d_secret_key_array, secret_key_array_.get(), old_poly_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // 这个函数没有释放内存，接着往下做
        // compute_secret_key_array(encrypted_size - 1, d_secret_key_array);
        compute_secret_key_array(encrypted_size - 1);
        checkCudaErrors(cudaMemcpy(
            d_secret_key_array, secret_key_array_.get(), new_poly_size * sizeof(uint64_t), cudaMemcpyHostToDevice));


        if (encrypted_size == 2)
        {
            printf("in if (encrypted_size == 2) \n");

            uint64_t *d_c0 = encrypted.d_data();
            uint64_t *d_c1 = encrypted.d_data() + coeff_modulus_size * coeff_count;


            if (is_ntt_form)
            {
                // printf("in encrypted_size == 2 --- if (is_ntt_form) \n");

                for (int i = 0; i < coeff_modulus_size; ++i)
                {
                    dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_c1 + i * coeff_count, d_secret_key_array + i * coeff_count, coeff_count,
                        coeff_modulus[i].value(), coeff_modulus[i].const_ratio()[0], coeff_modulus[i].const_ratio()[1],
                        d_destination + i * coeff_count);
                    add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_destination + i * coeff_count, d_c0 + i * coeff_count, coeff_count, coeff_modulus[i].value(),
                        d_destination + i * coeff_count);
                }
            }
            else
            {
                // printf("in encrypted_size == 2 --- if(is_ntt_form) ELSE \n");

                // 先将 c1 拷贝到 destination
                set_uint_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_c1, coeff_modulus_size * coeff_count, d_destination);
                uint64_t temp_mu;

#if NTT_VERSION == 3
                ntt_v3(context_, encrypted.parms_id(), d_destination, coeff_modulus_size);
#else 
                ntt_v1(context_, encrypted.parms_id(), d_destination, coeff_modulus_size);
#endif



                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[i].bit_count() * 2);
                    temp_mu = (mu1 / coeff_modulus[i].value()).low;                    

                    dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_destination + i * coeff_count, d_secret_key_array + i * coeff_count, coeff_count,
                        coeff_modulus[i].value(), coeff_modulus[i].const_ratio()[0], coeff_modulus[i].const_ratio()[1],
                        d_destination + i * coeff_count);

                    inverseNTT(
                        d_destination + coeff_count * i, coeff_count, ntt, coeff_modulus[i].value(), temp_mu,
                        coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);

                    add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_destination + i * coeff_count, d_c0 + i * coeff_count, coeff_count, coeff_modulus[i].value(),
                        d_destination + i * coeff_count);
                }
            }

            // free
            // cudaFree(d_c0);
            // cudaFree(d_c1);
        }
        else
        {
            // printf("else encrypted_size != 2  \n");
            // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
            // Now do the dot product of encrypted_copy and the secret key array using NTT.
            // The secret key powers are already NTT transformed.
            SEAL_ALLOCATE_GET_POLY_ITER(encrypted_copy, encrypted_size - 1, coeff_count, coeff_modulus_size, pool);
            set_poly_array(encrypted.data(1), encrypted_size - 1, coeff_count, coeff_modulus_size, encrypted_copy);

            // malloc
            uint64_t *d_encrypted_copy = nullptr;
            uint64_t *d_encrypted_0 = nullptr;
            checkCudaErrors(cudaMalloc((void **)&d_encrypted_copy, (encrypted_size - 1) * coeff_count * coeff_modulus_size * sizeof(uint64_t)));
            checkCudaErrors(cudaMalloc((void **)&d_encrypted_0, coeff_count * coeff_modulus_size * sizeof(uint64_t)));
            checkCudaErrors(cudaMemcpy(
                d_encrypted_copy, encrypted.data(1),
                (encrypted_size - 1) * coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(
                d_encrypted_0, encrypted.data(0), coeff_count * coeff_modulus_size * sizeof(uint64_t),
                cudaMemcpyHostToDevice));

            // Transform c_1, c_2, ... to NTT form unless they already are
            if (!is_ntt_form)
            {
                // ntt_negacyclic_harvey_lazy(encrypted_copy, encrypted_size - 1, ntt_tables);
                for (size_t i = 0; i < (encrypted_size - 1) ; i++)
                {
#if NTT_VERSION == 3
                    ntt_v3(context_, encrypted.parms_id(), d_encrypted_copy + coeff_count * i * coeff_modulus_size, coeff_modulus_size);
#else
                    ntt_v1(context_, encrypted.parms_id(), d_encrypted_copy + coeff_count * i * coeff_modulus_size, coeff_modulus_size);
#endif
                }
            }

            // Compute dyadic product with secret power array
            // auto secret_key_array = PolyIter(secret_key_array_.get(), coeff_count, key_coeff_modulus_size);
            // SEAL_ITERATE(iter(encrypted_copy, secret_key_array), encrypted_size - 1, [&](auto I) {
            //     dyadic_product_coeffmod(get<0>(I), get<1>(I), coeff_modulus_size, coeff_modulus, get<0>(I));
            // });
            for (size_t i = 0; i < (encrypted_size - 1); ++i)
            {
                for (size_t j = 0; j < coeff_modulus_size; ++j)
                {
                    dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_encrypted_copy + i * coeff_modulus_size * coeff_count + j * coeff_count,
                        d_secret_key_array + i * key_coeff_modulus_size * coeff_count + j * coeff_count, coeff_count,
                        coeff_modulus[j].value(), coeff_modulus[j].const_ratio()[0], coeff_modulus[j].const_ratio()[1],
                        d_encrypted_copy + i * coeff_modulus_size * coeff_count + j * coeff_count);
                }
            }

            // Aggregate all polynomials together to complete the dot product
            // set_zero_poly(coeff_count, coeff_modulus_size, destination);
            set_zero_poly_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(coeff_count, coeff_modulus_size, d_destination);

            // SEAL_ITERATE(encrypted_copy, encrypted_size - 1, [&](auto I) {
            //     add_poly_coeffmod(destination, I, coeff_modulus_size, coeff_modulus, destination);
            // });
            for (size_t i = 0; i < (encrypted_size - 1) * coeff_modulus_size; i++)
            {
                size_t j = i % coeff_modulus_size;
                add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_destination + j * coeff_count, d_encrypted_copy + i * coeff_count, coeff_count,
                    coeff_modulus[j].value(), d_destination + j * coeff_count);
            }

            if (!is_ntt_form)
            {
                // If the input was not in NTT form, need to transform back
                // inverse_ntt_negacyclic_harvey(destination, coeff_modulus_size, ntt_tables);
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    inverseNTT(
                        d_destination + coeff_count * i, coeff_count, ntt, coeff_modulus[i].value(), mu[i],
                        coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);
                }
            }

            // Finally add c_0 to the result; note that destination should be in the same (NTT) form as encrypted
            // add_poly_coeffmod(destination, *iter(encrypted), coeff_modulus_size, coeff_modulus, destination);
            for (size_t i = 0; i < (encrypted_size - 1) * coeff_modulus_size; ++i)
            {
                size_t j = i % coeff_modulus_size;
                add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_destination + j * coeff_count, d_encrypted_0 + i * coeff_count, coeff_count,
                    coeff_modulus[j].value(), d_destination + j * coeff_count);
            }

            // free
            // cudaFree(d_encrypted_copy);
            // cudaFree(d_encrypted_0);
        }

        // copy back
        checkCudaErrors(cudaMemcpy(
            (*destination).ptr(), d_destination, coeff_modulus_size * coeff_count * sizeof(uint64_t),
            cudaMemcpyDeviceToHost));
    }

    // Compute c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q.
    // Store result in destination in RNS form.
    void Decryptor::dot_product_ct_sk_array_cuda(
        const Ciphertext &encrypted, uint64_t *d_destination, MemoryPoolHandle pool)
    {
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t key_coeff_count = context_.key_context_data()->parms().poly_modulus_degree();
        // size_t key_coeff_modulus_size = 2;
        size_t key_coeff_modulus_size = context_.key_context_data()->parms().coeff_modulus().size();
        size_t encrypted_size = encrypted.size();
        size_t max_power = encrypted_size - 1;
        auto is_ntt_form = encrypted.is_ntt_form();
        uint64_t *d_encrypted_data = encrypted.d_data();
        uint64_t *d_modulus = parms.d_coeff_modulus_value();

        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();

        auto ntt_tables = context_data.small_ntt_tables();

        // printf("check root matrix goes here\n");
        // auto &first_context_data = *context_.first_context_data();
        // uint64_t *d_root_matrix = first_context_data.d_root_matrix();        

        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();

        // === start for ntt ===============================================================
        // malloc
        cudaStream_t ntt = 0;

        uint64_t mu[2 * coeff_modulus_size];
        for (int i = 0; i < coeff_modulus_size; i++)
        {
            k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[i].bit_count() * 2);
            mu[i] = (mu1 / coeff_modulus[i].value()).low;
        }
        // === end for ntt ===============================================================

        size_t old_size = secret_key_array_size_;
        size_t new_size = max(max_power, old_size);
        // printf("decrypt new_size: %llu\n", new_size);
        size_t old_poly_size = mul_safe(old_size, coeff_count, key_coeff_modulus_size);
        size_t new_poly_size = mul_safe(new_size, coeff_count, key_coeff_modulus_size);

        uint64_t *d_secret_key_array = d_secret_key_;

        // 这个函数没有释放内存，接着往下做
        compute_secret_key_array(encrypted_size - 1, d_secret_key_array);

        // 参数检查
        // printf("key coeff modulus size %lu\n", key_coeff_modulus_size);
        // printf("coeff modulus size %lu\n", coeff_modulus_size);
        if (encrypted_size == 2)
        {
            // printf("in if (encrypted_size == 2) \n");
            // malloc
            uint64_t *d_c0 = d_encrypted_data;
            uint64_t *d_c1 = d_encrypted_data + coeff_count * coeff_modulus_size;


            if (is_ntt_form)
            {
                decrypt_ntt_size2_kernel<<<(coeff_count * coeff_modulus_size + 255) / 256, 256>>>(
                    d_c0, d_secret_key_array, coeff_count, coeff_modulus_size, d_modulus, d_coeff_modulus_ratio_0,
                    d_coeff_modulus_ratio_1, d_destination);
            }
            else
            {
                // printf("in encrypted_size == 2 --- if(is_ntt_form) ELSE \n");

                // 先将 c1 拷贝到 destination
                set_uint_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_c1, coeff_modulus_size * coeff_count, d_destination);
                uint64_t temp_mu;
#if NTT_VERSION == 3
                ntt_v3(context_, encrypted.parms_id(), d_destination, coeff_modulus_size);
#else 
                ntt_v1(context_, encrypted.parms_id(), d_destination, coeff_modulus_size);
#endif

                for (size_t i = 0; i < coeff_modulus_size; i++)
                {

                    uint64_t ratio_0 = coeff_modulus[i].const_ratio().data()[0];
                    uint64_t ratio_1 = coeff_modulus[i].const_ratio().data()[1];
                    uint64_t modulus = coeff_modulus[i].value();
                    int bit_count = ntt_tables[i].coeff_count_power();
// 修改待测试
                    dyadic_product_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_destination + i * coeff_count, d_secret_key_array + i * coeff_count, coeff_count,
                        coeff_modulus[i].value(), coeff_modulus[i].const_ratio()[0], coeff_modulus[i].const_ratio()[1],
                        d_destination + i * coeff_count);

                    inverseNTT(
                        d_destination + coeff_count * i, coeff_count, ntt, coeff_modulus[i].value(), temp_mu,
                        coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);

                    add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                        d_destination + i * coeff_count, d_c0 + i * coeff_count, coeff_count, coeff_modulus[i].value(),
                        d_destination + i * coeff_count);
                }
            }
        }
        else
        {
            context_.ensure_ntt_size(coeff_count);
            uint64_t *ntt_temp = context_.ntt_temp();
            // printf("else encrypted_size != 2, it is: %d  \n", encrypted_size);

            // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
            // Now do the dot product of encrypted_copy and the secret key array using NTT.
            // The secret key powers are already NTT transformed.

            uint64_t *d_encrypted_copy = d_encrypted_data + coeff_count * coeff_modulus_size;
            uint64_t *d_encrypted_0 = d_encrypted_data;

            // Transform c_1, c_2, ... to NTT form unless they already are
            if (!is_ntt_form)
            {
                for(size_t i = 0; i < (encrypted_size - 1); i++){
#if NTT_VERSION == 3            
                    ntt_v3(context_, encrypted.parms_id(), d_encrypted_copy + i * coeff_count * coeff_modulus_size, coeff_modulus_size);
#else 
                    ntt_v1(context_, encrypted.parms_id(), d_encrypted_copy + i * coeff_count * coeff_modulus_size, coeff_modulus_size);
#endif
                }


            }

            // Compute dyadic product with secret power array
            dyadic_product_coeffmod_kernel_two_modulu<<<
                (coeff_count * coeff_modulus_size * (encrypted_size - 1) + 255) / 256, 256>>>(
                d_encrypted_copy, d_secret_key_array, (encrypted_size - 1), coeff_count, coeff_modulus_size,
                key_coeff_modulus_size, d_modulus, d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1, d_encrypted_copy);

            // Aggregate all polynomials together to complete the dot product
            set_zero_poly_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(coeff_count, coeff_modulus_size, d_destination);

            for (size_t i = 0; i < (encrypted_size - 1) * coeff_modulus_size; i++)
            {
                size_t j = i % coeff_modulus_size;
                add_poly_coeffmod_kernel<<<(coeff_count + 1023) / 1024, 1024>>>(
                    d_destination + j * coeff_count, d_encrypted_copy + i * coeff_count, coeff_count,
                    coeff_modulus[j].value(), d_destination + j * coeff_count);
            }

            if (!is_ntt_form)
            {
                // If the input was not in NTT form, need to transform back
                for (size_t i = 0; i < coeff_modulus_size; i++)
                {
                    inverseNTT(
                        d_destination + coeff_count * i, coeff_count, ntt, coeff_modulus[i].value(), mu[i],
                        coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);
                }
            }

            // Finally add c_0 to the result; note that destination should be in the same (NTT) form as encrypted
            add_poly_coeffmod_kernel<<<(coeff_count * coeff_modulus_size + 255) / 256, 256>>>(
                d_destination, d_encrypted_0, 1, coeff_count, coeff_modulus_size, d_modulus, d_destination);

            // free
            // cudaFree(d_encrypted_copy);
            // cudaFree(d_encrypted_0);
        }
    }

    void Decryptor::ckks_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (!encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted must be in NTT form");
        }

        // We already know that the parameters are valid
        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t key_coeff_modulus_size = context_.key_context_data()->parms().coeff_modulus().size();
        size_t rns_poly_uint64_count = mul_safe(coeff_count, coeff_modulus_size);

        auto is_ntt_form = encrypted.is_ntt_form();
        auto ntt_tables = context_data.small_ntt_tables();

        // compute_secret_key_array
        // WARNING: This function must be called with the original context_data

        // Decryption consists in finding
        // c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q_1 * q_2 * q_3
        // as long as ||m + v|| < q_1 * q_2 * q_3.
        // This is equal to m + v where ||v|| is small enough.

        // Since we overwrite destination, we zeroize destination parameters
        // This is necessary, otherwise resize will throw an exception.
        destination.parms_id() = parms_id_zero;

        // Resize destination to appropriate size

        destination.resize(rns_poly_uint64_count);
        destination.resize_gpu(rns_poly_uint64_count);

        // Do the dot product of encrypted and the secret key array using NTT.

        dot_product_ct_sk_array_cuda(encrypted, destination.d_data(), pool);


        // Set destination parameters as in encrypted
        destination.parms_id() = encrypted.parms_id();
        destination.scale() = encrypted.scale();
    }

    void Decryptor::bfv_decrypt(const Ciphertext &encrypted, Plaintext &destination, MemoryPoolHandle pool)
    {
        if (encrypted.is_ntt_form())
        {
            throw invalid_argument("encrypted cannot be in NTT form");
        }

        // printf("bfv decrypt goes here\n");

        auto &context_data = *context_.get_context_data(encrypted.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Firstly find c_0 + c_1 *s + ... + c_{count-1} * s^{count-1} mod q
        // This is equal to Delta m + v where ||v|| < Delta/2.
        // Add Delta / 2 and now we have something which is Delta * (m + epsilon) where epsilon < 1
        // Therefore, we can (integer) divide by Delta and the answer will round down to m.

        // Make a temp destination for all the arithmetic mod qi before calling FastBConverse
        SEAL_ALLOCATE_ZERO_GET_RNS_ITER(tmp_dest_modq, coeff_count, coeff_modulus_size, pool);

        uint64_t *d_temp_dest_modq = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_temp_dest_modq, coeff_count * coeff_modulus_size * sizeof(uint64_t)));

        // put < (c_1 , c_2, ... , c_{count-1}) , (s,s^2,...,s^{count-1}) > mod q in destination
        // Now do the dot product of encrypted_copy and the secret key array using NTT.
        // The secret key powers are already NTT transformed.
        // dot_product_ct_sk_array(encrypted, tmp_dest_modq, pool_);
        dot_product_ct_sk_array_cuda(encrypted, d_temp_dest_modq, pool_);

        checkCudaErrors(cudaMemcpy(tmp_dest_modq, d_temp_dest_modq, coeff_count * coeff_modulus_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // Allocate a full size destination to write to
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count);

// 下边这个函数后续迁移，目标保证decrypt前不用搬数据到cpu
        // Divide scaling variant using BEHZ FullRNS techniques
        context_data.rns_tool()->decrypt_scale_and_round(tmp_dest_modq, destination.data(), pool);

        // How many non-zero coefficients do we really have in the result?
        size_t plain_coeff_count = get_significant_uint64_count_uint(destination.data(), coeff_count);

        // Resize destination to appropriate size
        destination.resize(max(plain_coeff_count, size_t(1)));
    }
} // namespace seal
