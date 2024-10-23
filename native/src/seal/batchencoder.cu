#include "seal/batchencoder.h"
#include "seal/valcheck.h"
#include "seal/util/common.h"
#include "seal/util/common.cuh"
#include <algorithm>
#include <cublas_v2.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "cuda_runtime.h"
// #include "device_launch_parameters.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    BatchEncoder::BatchEncoder(const SEALContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        auto &context_data = *context_.first_context_data();
        if (context_data.parms().scheme() != scheme_type::bfv && context_data.parms().scheme() != scheme_type::bgv)
        {
            throw invalid_argument("unsupported scheme");
        }
        if (!context_data.qualifiers().using_batching)
        {
            throw invalid_argument("encryption parameters are not valid for batching");
        }

        // Set the slot count
        slots_ = context_data.parms().poly_modulus_degree();

        // Reserve space for all of the primitive roots
        roots_of_unity_ = allocate_uint(slots_, pool_);

        // Fill the vector of roots of unity with all distinct odd powers of generator.
        // These are all the primitive (2*slots_)-th roots of unity in integers modulo
        // parms.plain_modulus().
        populate_roots_of_unity_vector(context_data);

        // Populate matrix representation index map
        populate_matrix_reps_index_map();

        checkCudaErrors(cudaMalloc((void **)&d_matrix_reps_index_map_, slots_ * sizeof(size_t)));
        checkCudaErrors(cudaMemcpy(
            d_matrix_reps_index_map_, matrix_reps_index_map_.get(), slots_ * sizeof(size_t), cudaMemcpyHostToDevice));


    }

    void BatchEncoder::populate_roots_of_unity_vector(const SEALContext::ContextData &context_data)
    {
        uint64_t root = context_data.plain_ntt_tables()->get_root();
        auto &modulus = context_data.parms().plain_modulus();

        uint64_t generator_sq = multiply_uint_mod(root, root, modulus);
        roots_of_unity_[0] = root;

        for (size_t i = 1; i < slots_; i++)
        {
            roots_of_unity_[i] = multiply_uint_mod(roots_of_unity_[i - 1], generator_sq, modulus);
        }
    }

    void BatchEncoder::populate_matrix_reps_index_map()
    {
        int logn = get_power_of_two(slots_);
        matrix_reps_index_map_ = allocate<size_t>(slots_, pool_);

        // Copy from the matrix to the value vectors
        size_t row_size = slots_ >> 1;
        size_t m = slots_ << 1;
        uint64_t gen = 3;
        uint64_t pos = 1;
        for (size_t i = 0; i < row_size; i++)
        {
            // Position in normal bit order
            uint64_t index1 = (pos - 1) >> 1;
            uint64_t index2 = (m - pos - 1) >> 1;

            // Set the bit-reversed locations
            matrix_reps_index_map_[i] = safe_cast<size_t>(util::reverse_bits(index1, logn));
            matrix_reps_index_map_[row_size | i] = safe_cast<size_t>(util::reverse_bits(index2, logn));

            // Next primitive root
            pos *= gen;
            pos &= (m - 1);
        }
    }

    void BatchEncoder::reverse_bits(uint64_t *input)
    {
#ifdef SEAL_DEBUG
        if (input == nullptr)
        {
            throw invalid_argument("input cannot be null");
        }
#endif
        size_t coeff_count = context_.first_context_data()->parms().poly_modulus_degree();
        int logn = get_power_of_two(coeff_count);
        for (size_t i = 0; i < coeff_count; i++)
        {
            uint64_t reversed_i = util::reverse_bits(i, logn);
            if (i < reversed_i)
            {
                swap(input[i], input[reversed_i]);
            }
        }
    }

    __global__ void encode_kernel(
        const uint64_t *values_matrix, uint64_t *destination, size_t *matrix_reps_index_map, int slots, size_t values_matrix_size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while(i < slots){
            int index = matrix_reps_index_map[i];
            if (i < values_matrix_size)
            {
                destination[index] = values_matrix[i];
            }
            else if (i < slots)
            {
                destination[index] = 0;
            }
            i += blockDim.x * gridDim.x;
        }

    }

    __global__ void encode_kernel(
        const int64_t *values_matrix, uint64_t *destination, size_t *matrix_reps_index_map, int slots, size_t values_matrix_size, uint64_t modulus)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while(i < slots){
            if (i < values_matrix_size)
            {
                int index = matrix_reps_index_map[i];
                destination[index] = (values_matrix[i] < 0) ? (modulus + static_cast<uint64_t>(values_matrix[i]))
                                                            : static_cast<uint64_t>(values_matrix[i]);
            } else if (i < slots)
            {
                int index = matrix_reps_index_map[i];
                destination[index] = 0;
            }
            i += blockDim.x * gridDim.x;
        }
        
    }

    void BatchEncoder::encode(const vector<uint64_t> &values_matrix, Plaintext &destination) const
    {
        auto &context_data = *context_.first_context_data();
        auto plain_modulu = context_data.parms().plain_modulus();
        uint64_t modulus = plain_modulu.value();

        // Validate input parameters
        size_t values_matrix_size = values_matrix.size();
        if (values_matrix_size > slots_)
        {
            throw invalid_argument("values_matrix size is too large");
        }
        // Set destination to full size
        destination.resize(slots_);
        destination.parms_id() = parms_id_zero;
        // First write the values to destination coefficients.
        // Read in top row, then bottom row.

        uint64_t *d_values = nullptr;
        allocate_gpu<uint64_t>(&d_values, values_matrix_size);
        checkCudaErrors(cudaMemcpy(d_values, values_matrix.data(), values_matrix_size * sizeof(uint64_t), cudaMemcpyHostToDevice));


        destination.d_data_malloc(slots_);


        encode_kernel<<<(slots_ + 255) / 256, 256>>>(d_values, destination.d_data(), d_matrix_reps_index_map_, slots_, values_matrix_size);

        // Transform destination using inverse of negacyclic NTT
        // Note: We already performed bit-reversal when reading in the matrix

        uint64_t *d_plain_inv_root_power = context_data.d_plain_inv_root_powers();
        k_uint128_t mu1 = k_uint128_t::exp2(plain_modulu.bit_count() * 2);
        uint64_t temp_mu = (mu1 / plain_modulu.value()).low;
        cudaStream_t ntt = 0;
        inverseNTT(
            destination.d_data(), 
            slots_, 
            ntt, 
            plain_modulu.value(), 
            temp_mu,
            plain_modulu.bit_count(), 
            d_plain_inv_root_power);


        checkCudaErrors(cudaMemcpy(destination.data(), destination.d_data(), slots_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    }

    void BatchEncoder::encode(const vector<int64_t> &values_matrix, Plaintext &destination) const
    {
        auto &context_data = *context_.first_context_data();
        auto plain_modulu = context_data.parms().plain_modulus();
        uint64_t modulus = plain_modulu.value();

        // Validate input parameters
        size_t values_matrix_size = values_matrix.size();
        if (values_matrix_size > slots_)
        {
            throw invalid_argument("values_matrix size is too large");
        }
#ifdef SEAL_DEBUG
        uint64_t plain_modulus_div_two = modulus >> 1;
        for (auto v : values_matrix)
        {
            // Validate the i-th input
            if (unsigned_gt(llabs(v), plain_modulus_div_two))
            {
                throw invalid_argument("input value is larger than plain_modulus");
            }
        }
#endif
        // Set destination to full size
        destination.resize(slots_);
        destination.parms_id() = parms_id_zero;
        // First write the values to destination coefficients.
        // Read in top row, then bottom row.

        int64_t *d_values = nullptr;
        checkCudaErrors(cudaMalloc((void **)&d_values, values_matrix_size * sizeof(int64_t)));
        checkCudaErrors(cudaMemcpy(d_values, values_matrix.data(), values_matrix_size * sizeof(int64_t), cudaMemcpyHostToDevice));

        encode_kernel<<<(slots_ + 255) / 256, 256>>>(d_values, destination.d_data(), d_matrix_reps_index_map_, slots_, values_matrix_size, modulus);

        // Transform destination using inverse of negacyclic NTT
        // Note: We already performed bit-reversal when reading in the matrix
        uint64_t *d_plain_inv_root_power = context_data.d_plain_inv_root_powers();
        k_uint128_t mu1 = k_uint128_t::exp2(plain_modulu.bit_count() * 2);
        uint64_t temp_mu = (mu1 / plain_modulu.value()).low;
        cudaStream_t ntt = 0;

        inverseNTT(
            destination.d_data() , 
            values_matrix_size, 
            ntt, 
            plain_modulu.value(), 
            temp_mu,
            plain_modulu.bit_count(), 
            d_plain_inv_root_power);

        checkCudaErrors(cudaMemcpy(destination.data(), destination.d_data(), slots_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    }

    void BatchEncoder::decode(const Plaintext &plain, vector<uint64_t> &destination, MemoryPoolHandle pool) const
    {
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (plain.is_ntt_form())
        {
            throw invalid_argument("plain cannot be in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_.first_context_data();

        // Set destination size
        destination.resize(slots_);

        // Never include the leading zero coefficient (if present)
        size_t plain_coeff_count = min(plain.coeff_count(), slots_);

        auto temp_dest(allocate_uint(slots_, pool));

        // Make a copy of poly
        set_uint(plain.data(), plain_coeff_count, temp_dest.get());
        set_zero_uint(slots_ - plain_coeff_count, temp_dest.get() + plain_coeff_count);

        // Transform destination using negacyclic NTT.
        ntt_negacyclic_harvey(temp_dest.get(), *context_data.plain_ntt_tables());

        // Read top row, then bottom row
        for (size_t i = 0; i < slots_; i++)
        {
            destination[i] = temp_dest[matrix_reps_index_map_[i]];
        }
    }

    void BatchEncoder::decode(const Plaintext &plain, vector<int64_t> &destination, MemoryPoolHandle pool) const
    {
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
        if (plain.is_ntt_form())
        {
            throw invalid_argument("plain cannot be in NTT form");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_.first_context_data();
        uint64_t modulus = context_data.parms().plain_modulus().value();

        // Set destination size
        destination.resize(slots_);

        // Never include the leading zero coefficient (if present)
        size_t plain_coeff_count = min(plain.coeff_count(), slots_);

        auto temp_dest(allocate_uint(slots_, pool));

        // Make a copy of poly
        set_uint(plain.data(), plain_coeff_count, temp_dest.get());
        set_zero_uint(slots_ - plain_coeff_count, temp_dest.get() + plain_coeff_count);

        // Transform destination using negacyclic NTT.
        ntt_negacyclic_harvey(temp_dest.get(), *context_data.plain_ntt_tables());

        // Read top row, then bottom row
        uint64_t plain_modulus_div_two = modulus >> 1;
        for (size_t i = 0; i < slots_; i++)
        {
            uint64_t curr_value = temp_dest[matrix_reps_index_map_[i]];
            destination[i] = (curr_value > plain_modulus_div_two)
                                 ? (static_cast<int64_t>(curr_value) - static_cast<int64_t>(modulus))
                                 : static_cast<int64_t>(curr_value);
        }
    }

} // namespace seal