#include "seal/ckks.h"
#include "seal/util/rlwe.h"
// #include "seal/util/uintarithmod.cuh"
#include "seal/util/helper.cuh"
#include <complex>
#include <cufft.h>
#include <random>
#include <stdexcept>

using namespace std;
using namespace seal::util;

namespace seal
{
    CKKSEncoder::CKKSEncoder(const SEALContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        auto &context_data = *context_.first_context_data();
        if (context_data.parms().scheme() != scheme_type::ckks)
        {
            throw invalid_argument("unsupported scheme");
        }

        size_t coeff_count = context_data.parms().poly_modulus_degree();
        slots_ = coeff_count >> 1;
        int logn = get_power_of_two(coeff_count); // n = 2^logn

        matrix_reps_index_map_ = allocate<size_t>(coeff_count, pool_);

        // Copy from the matrix to the value vectors
        uint64_t gen = 3;
        uint64_t pos = 1;
        uint64_t m = static_cast<uint64_t>(coeff_count) << 1;
        for (size_t i = 0; i < slots_; i++)
        {
            // Position in normal bit order
            uint64_t index1 = (pos - 1) >> 1;
            uint64_t index2 = (m - pos - 1) >> 1;

            // Set the bit-reversed locations
            matrix_reps_index_map_[i] = safe_cast<size_t>(reverse_bits(index1, logn));
            matrix_reps_index_map_[slots_ | i] = safe_cast<size_t>(reverse_bits(index2, logn));

            // Next primitive root
            pos *= gen;
            pos &= (m - 1);
        }

        // We need 1~(n-1)-th powers of the primitive 2n-th root, m = 2n
        root_powers_ = allocate<complex<double>>(coeff_count, pool_);
        inv_root_powers_ = allocate<complex<double>>(coeff_count, pool_);
        // Powers of the primitive 2n-th root have 4-fold symmetry
        if (m >= 8)
        {
            complex_roots_ = make_shared<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m), pool_));
            for (size_t i = 1; i < coeff_count; i++)
            {
                root_powers_[i] = complex_roots_->get_root(reverse_bits(i, logn));
                inv_root_powers_[i] = conj(complex_roots_->get_root(reverse_bits(i - 1, logn) + 1));
            }
        }
        else if (m == 4)
        {
            root_powers_[1] = { 0, 1 };
            inv_root_powers_[1] = { 0, -1 };
        }

        complex_arith_ = ComplexArith();
        fft_handler_ = FFTHandler(complex_arith_);
    }

    __global__ void encode_negative_kernel(
        int64_t value, uint64_t *destination, uint64_t *coeff_modulus, uint64_t *coeff_modulus_ratio,
        size_t coeff_count, size_t coeff_modulus_size)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < coeff_count * coeff_modulus_size)
        {
            uint64_t tmp = static_cast<uint64_t>(value);
            tmp += coeff_modulus[index / coeff_count];
            tmp = barrett_reduce_64_kernel(tmp, coeff_modulus[index / coeff_count], coeff_modulus_ratio[index / coeff_count]);
            destination[index] = tmp;

            index += blockDim.x * gridDim.x;
        }
    }
    __global__ void encode_positive_kernel(
        int64_t value, uint64_t *destination, uint64_t *coeff_modulus, uint64_t *coeff_modulus_ratio,
        size_t coeff_count, size_t coeff_modulus_size)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < coeff_count * coeff_modulus_size)
        {
            uint64_t tmp = static_cast<uint64_t>(value);
            tmp += coeff_modulus[index / coeff_count];
            tmp = barrett_reduce_64_kernel(tmp, coeff_modulus[index / coeff_count], coeff_modulus_ratio[index / coeff_count]);
            destination[index] = tmp;

            index += blockDim.x * gridDim.x;
        }
    }

    __global__ void encode_negate_uint_kernel(
        int64_t value, uint64_t *destination, uint64_t *coeff_modulus, uint64_t *coeff_modulus_ratio,
        size_t coeff_count, size_t coeff_modulus_size)
    {
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        while (j < coeff_modulus_size)
        {
            uint64_t tmp = static_cast<uint64_t>(value);
            tmp = barrett_reduce_64_kernel(tmp, coeff_modulus[j], coeff_modulus_ratio[j]);
            tmp = (coeff_modulus[j] - tmp) & (-(tmp != 0));
            for (size_t i = 0; i < coeff_count; i++)
            {
                destination[j * coeff_count + i] = tmp;
            }

            j += blockDim.x * gridDim.x;
        }
    }

    __global__ void encode_positive_128_kernel(
        uint64_t *value, uint64_t *destination, uint64_t *coeff_modulus, uint64_t *coeff_modulus_ratio_0,
        uint64_t *coeff_modulus_ratio_1, const unsigned long long modulus_size, const unsigned long long coeff_count)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < coeff_count * modulus_size)
        {
            uint64_t *tmp = static_cast<uint64_t *>(static_cast<void *>(value));

            uint64_t coeff_modulus_ratio[2] ={coeff_modulus_ratio_0[index / coeff_count],
                                              coeff_modulus_ratio_1[index / coeff_count]};  

            uint64_t result = barrett_reduce_128_kernel(tmp, coeff_modulus[index / coeff_count], coeff_modulus_ratio);
            destination[index] = result;

            index += blockDim.x * gridDim.x;

        }
    }

    __global__ void encode_negate_uint_128_kernel(
        uint64_t *value, uint64_t *destination, uint64_t *coeff_modulus, uint64_t *coeff_modulus_ratio_0,
        uint64_t *coeff_modulus_ratio_1, const size_t coeff_count, const size_t coeff_modulus_size)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < coeff_count * coeff_modulus_size)
        {
            uint64_t *tmp = static_cast<uint64_t *>(static_cast<void *>(value));

            uint64_t coeff_modulus_ratio[2] ={coeff_modulus_ratio_0[index / coeff_count],
                                              coeff_modulus_ratio_1[index / coeff_count]};  


            uint64_t result = barrett_reduce_128_kernel(tmp, coeff_modulus[index / coeff_count], coeff_modulus_ratio);
            result = (coeff_modulus[index / coeff_count] - result) & (-(result != 0));

            destination[index] = result;
            
            index += blockDim.x * gridDim.x;

        }
    }

    __global__ inline void add_plain_copy_to_complex(
        double *res, const uint64_t *plain_copy2, const uint64_t *decryption_modulus,  const unsigned long long coeff_count,
         const unsigned long long coeff_modulus_size, uint64_t *upper_half_threshold, double inv_scale)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        double two_pow_64 = pow(2.0, 64);
        while (tid < coeff_count)
        {
            double res_accum = 0.0;
            size_t plain_start = tid * coeff_modulus_size;
            if (is_greater_than_or_equal_uint_kernel(
                    plain_copy2 + plain_start, upper_half_threshold, coeff_modulus_size))
            {
                double scaled_two_pow_64 = inv_scale;
                for (std::size_t j = 0; j < coeff_modulus_size; j++, scaled_two_pow_64 *= two_pow_64)
                {
                    if (plain_copy2[plain_start + j] > decryption_modulus[j])
                    {
                        auto diff = plain_copy2[plain_start + j] - decryption_modulus[j];
                        res_accum += diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
                    }
                    else
                    {
                        auto diff = decryption_modulus[j] - plain_copy2[plain_start + j];
                        res_accum -= diff ? static_cast<double>(diff) * scaled_two_pow_64 : 0.0;
                    }
                }
            }
            else
            {
                double scaled_two_pow_64 = inv_scale;
                for (std::size_t j = 0; j < coeff_modulus_size; j++, scaled_two_pow_64 *= two_pow_64)
                {
                    auto curr_coeff = plain_copy2[plain_start + j];
                    res_accum += curr_coeff ? static_cast<double>(curr_coeff) * scaled_two_pow_64 : 0.0;
                }
            }

            res[tid] = res_accum;
            // res[tid].imag(0.0); // imaginary part is always 0

            tid += blockDim.x * gridDim.x;
        }
    }

    __global__ inline void encode_internal_size_kernel(
        double *conj_values, uint64_t *destination, uint64_t *modulus, uint64_t *coeff_modulus_ratio,
         const unsigned long long coeff_count,  const unsigned long long coeff_modulus_size,  const unsigned long long n)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < n * coeff_modulus_size)
        {
            size_t des_index = index % coeff_count + (index / coeff_count) * coeff_count;
            double value = conj_values[index % coeff_count];
            bool is_negative = (value < 0);

            uint64_t coeffu = static_cast<uint64_t>(abs(value));

            uint64_t tmp = barrett_reduce_64_kernel(
                coeffu, modulus[index / coeff_count], coeff_modulus_ratio[index / coeff_count]);
            if (is_negative)
            {
                std::int64_t non_zero = static_cast<std::int64_t>(tmp != 0);
                destination[des_index] =
                    (modulus[index / coeff_count] - tmp) & static_cast<std::uint64_t>(-non_zero);
            }
            else
            {
                destination[des_index] = tmp;
            }
            index += blockDim.x * gridDim.x;
        }
    }

    __global__ inline void encode_internal_size_128_kernel(
        double *conj_values, uint64_t *destination, uint64_t *modulus, uint64_t *coeff_modulus_ratio0,
        uint64_t *coeff_modulus_ratio1, const unsigned long long coeff_count, const unsigned long long coeff_modulus_size, const unsigned long long n)
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < n * coeff_modulus_size)
        {
            size_t des_index = index % coeff_count + (index / coeff_count) * coeff_count;
            double value = conj_values[index % coeff_count];
            bool is_negative = (value < 0);

            value = abs(value);
            double two_pow_64 = pow(2.0, 64);

            std::uint64_t coeffu[2]{ static_cast<std::uint64_t>(
                                         value -
                                         two_pow_64 * static_cast<double>(static_cast<int>(value / two_pow_64))),
                                     static_cast<std::uint64_t>(value / two_pow_64) };

            std::uint64_t ratio[2]{ coeff_modulus_ratio0[index / coeff_count],
                                    coeff_modulus_ratio1[index / coeff_count] };

            uint64_t *coeff_ptr = coeffu;
            uint64_t tmp = barrett_reduce_128_kernel2(coeff_ptr, modulus[index / coeff_count], ratio);
            if (is_negative)
            {
                std::int64_t non_zero = static_cast<std::int64_t>(tmp != 0);
                destination[des_index] =
                    (modulus[index / coeff_count] - tmp) & static_cast<std::uint64_t>(-non_zero);
            }
            else
            {
                destination[des_index] = tmp;
            }
            index += blockDim.x * gridDim.x;
        }
    }

    __global__ void encode_internal_size_bt128_helper1(double *conj_values, uint64_t *d_coeffu, const unsigned long long n){
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < n)
        {
            double value = conj_values[index];
            bool is_negative = (value < 0);
            value = abs(value);
            __shared__ double two_pow_64;
            two_pow_64 = pow(2.0, 64);
            d_coeffu[index] = 0;
            while(value >= 1){
                d_coeffu[index] = static_cast<std::uint64_t>(fmod(value, two_pow_64));
                value /= two_pow_64;
            }

            index += blockDim.x * gridDim.x;
        }
    }

    __global__ void encode_internal_size_bt128_helper2(double *conj_values, uint64_t *d_coeffu, uint64_t *modulus, uint64_t *coeff_modulus_ratio0,
        uint64_t *coeff_modulus_ratio1, const unsigned long long coeff_count, const unsigned long long coeff_modulus_size, const unsigned long long n){
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        while (index < n)
        {
            double value = conj_values[index];
            bool is_negative = (value < 0);
            value = abs(value);
            __shared__ double two_pow_64;
            two_pow_64 = pow(2.0, 64);

            while(value >= 1){
                d_coeffu[index] = static_cast<std::uint64_t>(fmod(value, two_pow_64));
                value /= two_pow_64;
            }

            index += blockDim.x * gridDim.x;
        }
    }


    void CKKSEncoder::encode_internal(
        double value, parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Quick sanity check
        if (!product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        // Check that scale is positive and not too large
        if (scale <= 0 || (static_cast<int>(log2(scale)) >= context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // Compute the scaled value
        value *= scale;

        int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
        if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw invalid_argument("encoded value is too large");
        }

        double two_pow_64 = pow(2.0, 64);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.

        destination.parms_id() = parms_id_zero;
        destination.resize_gpu(coeff_count * coeff_modulus_size);
        uint64_t *d_destination = destination.d_data();

        double coeffd = round(value);
        bool is_negative = signbit(coeffd);
        coeffd = fabs(coeffd);

        // Use faster decomposition methods when possible
        if (coeff_bit_count <= 64)
        {
            uint64_t coeffu = static_cast<uint64_t>(fabs(coeffd));

            const int threads_per_block = 256;
            const int blocks_per_grid = (coeff_modulus_size * coeff_count + threads_per_block - 1) / threads_per_block;
            if (is_negative)
            {
                encode_negate_uint_kernel<<<blocks_per_grid, threads_per_block>>>(
                    coeffu, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_1, coeff_count,
                    coeff_modulus_size);
            }
            else
            {
                encode_positive_kernel<<<blocks_per_grid, threads_per_block>>>(
                    coeffu, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_1, coeff_count,
                    coeff_modulus_size);
            }

        }
        else if (coeff_bit_count <= 128)
        {
            printf("wpf in coeff_bit_count <=128 \n");
            uint64_t coeffu[2]{ static_cast<uint64_t>(fmod(coeffd, two_pow_64)),
                                static_cast<uint64_t>(coeffd / two_pow_64) };

            const int threads_per_block = 256;
            const int blocks_per_grid = (coeff_modulus_size * coeff_count + threads_per_block - 1) / threads_per_block;
            if (is_negative)
            {
                encode_negate_uint_128_kernel<<<blocks_per_grid, threads_per_block>>>(
                    coeffu, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1,
                    coeff_count, coeff_modulus_size);
            }
            else
            {

                encode_positive_128_kernel<<<blocks_per_grid, threads_per_block>>>(
                    coeffu, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1, coeff_modulus_size,
                    coeff_count);
            }

        }
        else
        {
            printf("wpf in coeff_bit_count defult \n");
            // Slow case
            auto coeffu(allocate_uint(coeff_modulus_size, pool));

            // We are at this point guaranteed to fit in the allocated space
            set_zero_uint(coeff_modulus_size, coeffu.get());
            auto coeffu_ptr = coeffu.get();
            while (coeffd >= 1)
            {
                *coeffu_ptr++ = static_cast<uint64_t>(fmod(coeffd, two_pow_64));
                coeffd /= two_pow_64;
            }

            // Next decompose this coefficient
            context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);

            // Finally replace the sign if necessary
            if (is_negative)
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        negate_uint_mod(coeffu[j], coeff_modulus[j]));
                }
            }
            else
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(destination.data() + (j * coeff_count), coeff_count, coeffu[j]);
                }
            }

        }


        destination.parms_id() = parms_id;
        destination.scale() = scale;
    }

    void CKKSEncoder::encode_internal(int64_t value, parms_id_type parms_id, Plaintext &destination) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Quick sanity check
        if (!product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        int coeff_bit_count = get_significant_bit_count(static_cast<uint64_t>(llabs(value))) + 2;
        if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw invalid_argument("encoded value is too large");
        }

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;
        // destination.resize(coeff_count * coeff_modulus_size);
        destination.d_data_malloc(coeff_count * coeff_modulus_size);

        uint64_t *d_destination = destination.d_data();

        const int threads_per_block = 256;
        const int blocks_per_grid = (coeff_modulus_size * coeff_count + threads_per_block - 1) / threads_per_block;

        if (value < 0)
        {
            encode_negative_kernel<<<blocks_per_grid, threads_per_block>>>(
                value, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_1, coeff_count, coeff_modulus_size);
        }
        else
        {
            encode_positive_kernel<<<blocks_per_grid, threads_per_block>>>(
                value, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_1, coeff_count, coeff_modulus_size);
        }

        // cudaMemcpy(
        //     destination.data(), d_destination, coeff_count * coeff_modulus_size * sizeof(uint64_t),
        //     cudaMemcpyDeviceToHost);

        destination.parms_id() = parms_id;
        destination.scale() = 1.0;
    }

    template <typename T, typename>
    void CKKSEncoder::encode_internal(
        const T *values, std::size_t values_size, parms_id_type parms_id, double scale, Plaintext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw std::invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!values && values_size > 0)
        {
            throw std::invalid_argument("values cannot be null");
        }
        if (values_size > slots_)
        {
            throw std::invalid_argument("values_size is too large");
        }
        if (!pool)
        {
            throw std::invalid_argument("pool is uninitialized");
        }
        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        uint64_t *d_coeff_modulus_ratio_0 = parms.d_coeff_modulus_ratio_0();
        uint64_t *d_coeff_modulus_ratio_1 = parms.d_coeff_modulus_ratio_1();
        uint64_t *d_coeff_modulus_value = parms.d_coeff_modulus_value();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = parms.poly_modulus_degree();
        int *d_bit_count = context_data.d_bit_count();
        auto ntt_tables = context_data.small_ntt_tables();
        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();

        // Quick sanity check
        if (!util::product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw std::logic_error("invalid parameters");
        }

        // Check that scale is positive and not too large
        if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
        {
            throw std::invalid_argument("scale out of bounds");
        }

        // values_size is guaranteed to be no bigger than slots_
        std::size_t n = util::mul_safe(slots_, std::size_t(2));

        auto conj_values = util::allocate<std::complex<double>>(n, pool, 0);
        for (std::size_t i = 0; i < values_size; i++)
        {
            conj_values[matrix_reps_index_map_[i]] = values[i];
            // TODO: if values are real, the following values should be set to zero, and multiply results by 2.
            conj_values[matrix_reps_index_map_[i + slots_]] = std::conj(values[i]);
        }
        double fix = scale / static_cast<double>(n);
        fft_handler_.transform_from_rev(conj_values.get(), util::get_power_of_two(n), inv_root_powers_.get(), &fix);

        double max_coeff = 0;
        for (std::size_t i = 0; i < n; i++)
        {
            max_coeff = std::max<>(max_coeff, std::fabs(conj_values[i].real()));
        }
        // Verify that the values are not too large to fit in coeff_modulus
        // Note that we have an extra + 1 for the sign bit
        // Don't compute logarithmis of numbers less than 1
        int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
        if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw std::invalid_argument("encoded values are too large");
        }

        double two_pow_64 = std::pow(2.0, 64);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;
        // destination.d_data_malloc(coeff_count * coeff_modulus_size);
        destination.resize(coeff_count*coeff_modulus_size);

        

        uint64_t *d_destination = destination.d_data();

        double conj_value_real[n];
        for (size_t i = 0; i < n; i++)
        {
            conj_value_real[i] = std::round(conj_values[i].real());
        }

        const int threads_per_block = 256;
        const int blocks_per_grid = (n * coeff_modulus_size + threads_per_block - 1) / threads_per_block;

        double *d_conj_values = nullptr;
        // checkCudaErrors(cudaMalloc((void **)&d_conj_values, n * sizeof(double)));
        allocate_gpu<double>(&d_conj_values, n);

        checkCudaErrors(cudaMemcpy(d_conj_values, conj_value_real, n * sizeof(double), cudaMemcpyHostToDevice));
        size_t c_vec_size = mul_safe(coeff_count, coeff_modulus_size);

        // Use faster decomposition methods when possible
        if (max_coeff_bit_count <= 64)
        {
            encode_internal_size_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_conj_values, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_1, coeff_count, coeff_modulus_size,
                n);

        }
        else if (max_coeff_bit_count <= 128)
        {
            encode_internal_size_128_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_conj_values, d_destination, d_coeff_modulus_value, d_coeff_modulus_ratio_0, d_coeff_modulus_ratio_1,
                coeff_count, coeff_modulus_size, n);
        }
        else
        {
            auto coeffu(util::allocate_uint(coeff_modulus_size, pool));
            for (std::size_t i = 0; i < n; i++)
            {
                double coeffd = std::round(conj_values[i].real());
                bool is_negative = std::signbit(coeffd);
                coeffd = std::fabs(coeffd);

                // We are at this point guaranteed to fit in the allocated space
                util::set_zero_uint(coeff_modulus_size, coeffu.get());
                auto coeffu_ptr = coeffu.get();
                while (coeffd >= 1)
                {
                    *coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
                    coeffd /= two_pow_64;
                }

                // Next decompose this coefficient
                context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);

                // Finally replace the sign if necessary
                if (is_negative)
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        destination[i + (j * coeff_count)] = util::negate_uint_mod(coeffu[j], coeff_modulus[j]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        destination[i + (j * coeff_count)] = coeffu[j];
                    }
                }
            }


            for (std::size_t i = 0; i < coeff_modulus_size; i++)
            {
                util::ntt_negacyclic_harvey(destination.data(i * coeff_count), ntt_tables[i]);
            }

            checkCudaErrors(cudaMemcpy(
                d_destination, destination.data(), coeff_count * coeff_modulus_size * sizeof(uint64_t),
                cudaMemcpyHostToDevice));

        }
        deallocate_gpu<double>(&d_conj_values, n);

        if (max_coeff_bit_count <= 128){
#if NTT_VERSION == 3
            ntt_v3(context_, parms_id, d_destination, coeff_modulus_size);

#else 
            ntt_v1(context_, parms_id, d_destination, coeff_modulus_size);
#endif
        }
        destination.parms_id() = parms_id;
        destination.scale() = scale;
    }

    template <typename T, typename>
    void CKKSEncoder::decode_internal(const Plaintext &plain, T *destination, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        // if (!is_valid_for(plain, context_))
        // {
        //     throw std::invalid_argument("plain is not valid for encryption parameters");
        // }
        if (!plain.is_ntt_form())
        {
            throw std::invalid_argument("plain is not in NTT form");
        }
        if (!destination)
        {
            throw std::invalid_argument("destination cannot be null");
        }
        if (!pool)
        {
            throw std::invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_.get_context_data(plain.parms_id());
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        const int stream_num = context_.num_streams();
        cudaStream_t *ntt_steam = context_.stream_context();

        std::size_t coeff_modulus_size = parms.coeff_modulus().size();
        std::size_t coeff_count = parms.poly_modulus_degree();
        std::size_t rns_poly_uint64_count = util::mul_safe(coeff_count, coeff_modulus_size);

        auto ntt_tables = context_data.small_ntt_tables();
        uint64_t *d_inv_root_powers = context_data.d_inv_root_powers();

        // Check that scale is positive and not too large
        if (plain.scale() <= 0 ||
            (static_cast<int>(log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count()))
        {
            throw std::invalid_argument("scale out of bounds");
        }

        // auto decryption_modulus = context_data.total_coeff_modulus();
        uint64_t *d_decryption_modulus = context_data.d_total_coeff_modulus();

        // auto upper_half_threshold = context_data.upper_half_threshold();
        uint64_t *d_upper_half_threshold = context_data.d_upper_half_threshold();
        int logn = util::get_power_of_two(coeff_count);

        // Quick sanity check
        if ((logn < 0) || (coeff_count < SEAL_POLY_MOD_DEGREE_MIN) || (coeff_count > SEAL_POLY_MOD_DEGREE_MAX))
        {
            throw std::logic_error("invalid parameters");
        }

        double inv_scale = double(1.0) / plain.scale();

        uint64_t *d_plain_copy = nullptr;
        // checkCudaErrors(cudaMalloc((void **)&d_plain_copy, sizeof(uint64_t) * rns_poly_uint64_count));
        allocate_gpu<uint64_t>(&d_plain_copy, rns_poly_uint64_count);

        checkCudaErrors(cudaMemcpy(
            d_plain_copy, plain.d_data(), sizeof(uint64_t) * rns_poly_uint64_count, cudaMemcpyDeviceToDevice));

        // printf("d_res allocate \n");
        double *d_res = nullptr;
        // checkCudaErrors(cudaMalloc((void **)&d_res, sizeof(double) * coeff_count));
        allocate_gpu<double>(&d_res, coeff_count);

        // Transform each polynomial from NTT domain
        uint64_t temp_mu = 0;
        for (std::size_t i = 0; i < coeff_modulus_size; i++)
        {
            k_uint128_t mu1 = k_uint128_t::exp2(coeff_modulus[i].bit_count() * 2);
            temp_mu = (mu1 / coeff_modulus[i].value()).low;
            inverseNTT(
                d_plain_copy + coeff_count * i, coeff_count, ntt_steam[i % stream_num], coeff_modulus[i].value(), temp_mu,
                coeff_modulus[i].bit_count(), d_inv_root_powers + coeff_count * i);
        }

        // CRT-compose the polynomial
        context_data.rns_tool()->base_q()->compose_array_cuda(d_plain_copy, coeff_count, pool);

        auto res(util::allocate<double>(coeff_count, pool));
        int block_size = 256;
        int grid_size = (coeff_count + block_size - 1) / block_size;
        add_plain_copy_to_complex<<<grid_size, block_size>>>(
            d_res, d_plain_copy, d_decryption_modulus, coeff_count, coeff_modulus_size, d_upper_half_threshold,
            inv_scale);

        checkCudaErrors(cudaMemcpy(res.get(), d_res, sizeof(double) * coeff_count, cudaMemcpyDeviceToHost));
        deallocate_gpu<double>(&d_res, coeff_count);
        deallocate_gpu<uint64_t>(&d_plain_copy, rns_poly_uint64_count);


        // 以下操作在CPU上执行
        auto res_complex(util::allocate<std::complex<double>>(coeff_count, pool));
        for (int i = 0; i < coeff_count; i++)
        {
            res_complex[i] = complex<double>(res[i], 0.0);
        }

        fft_handler_.transform_to_rev(res_complex.get(), logn, root_powers_.get());

        for (std::size_t i = 0; i < slots_; i++)
        {
            destination[i] = from_complex<double>(res_complex[static_cast<std::size_t>(matrix_reps_index_map_[i])]);
        }

        


    }

    template void CKKSEncoder::decode_internal<double, void>(
        const Plaintext &plain, double *destination, MemoryPoolHandle pool) const;
    template void CKKSEncoder::decode_internal<std::complex<double>, void>(
        const Plaintext &plain, std::complex<double> *destination, MemoryPoolHandle pool) const;
    template void CKKSEncoder::encode_internal<double, void>(
        const double *values, std::size_t values_size, parms_id_type parms_id, double scale, Plaintext &destination,
        MemoryPoolHandle pool) const;
    template void CKKSEncoder::encode_internal<std::complex<double>, void>(
        const std::complex<double> *values, std::size_t values_size, parms_id_type parms_id, double scale,
        Plaintext &destination, MemoryPoolHandle pool) const;

} // namespace seal
