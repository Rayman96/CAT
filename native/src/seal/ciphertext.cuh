// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.cuh"
#include "seal/util/helper.cuh"
#include "seal/dynarray.h"
#include "seal/memorymanager.h"
#include "seal/randomgen.h"
#include "seal/valcheck.h"
#include "seal/version.h"
#include "seal/util/common.h"
#include "seal/util/defines.h"
#include "seal/util/gpu_data.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace seal
{
    /**
    Class to store a ciphertext element. The data for a ciphertext consists
    of two or more polynomials, which are in IDEA SEAL_GPU stored in a CRT
    form with respect to the factors of the coefficient modulus. This data
    itself is not meant to be modified directly by the user, but is instead
    operated on by functions in the Evaluator class. The size of the backing
    array of a ciphertext depends on the encryption parameters and the size
    of the ciphertext (at least 2). If the size of the ciphertext is T,
    the poly_modulus_degree encryption parameter is N, and the number of
    primes in the coeff_modulus encryption parameter is K, then the
    ciphertext backing array requires precisely 8*N*K*T bytes of memory.
    A ciphertext also carries with it the parms_id of its associated
    encryption parameters, which is used to check the validity of the
    ciphertext for homomorphic operations and decryption.

    @par Memory Management
    The size of a ciphertext refers to the number of polynomials it contains,
    whereas its capacity refers to the number of polynomials that fit in the
    current memory allocation. In high-performance applications unnecessary
    re-allocations should be avoided by reserving enough memory for the
    ciphertext to begin with either by providing the desired capacity to the
    constructor as an extra argument, or by calling the reserve function at
    any time.

    @par Thread Safety
    In general, reading from ciphertext is thread-safe as long as no other
    thread is concurrently mutating it. This is due to the underlying data
    structure storing the ciphertext not being thread-safe.

    @see Plaintext for the class that stores plaintexts.
    */
    class Ciphertext
    {
    public:
        using ct_coeff_type = std::uint64_t;

        /**
        Constructs an empty ciphertext allocating no memory.

        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if pool is uninitialized
        */
        Ciphertext(MemoryPoolHandle pool = MemoryManager::GetPool()) : data_(std::move(pool))
        {}

        /**
        Constructs an empty ciphertext with capacity 2. In addition to the
        capacity, the allocation size is determined by the highest-level
        parameters associated to the given SEALContext.

        @param[in] context The SEALContext
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if pool is uninitialized
        */
        explicit Ciphertext(const SEALContext &context, MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
            // Allocate memory but don't resize
            reserve(context, 2);
        }

        /**
        Constructs an empty ciphertext with capacity 2. In addition to the
        capacity, the allocation size is determined by the encryption parameters
        with given parms_id.

        @param[in] context The SEALContext
        @param[in] parms_id The parms_id corresponding to the encryption
        parameters to be used
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        explicit Ciphertext(
            const SEALContext &context, parms_id_type parms_id, MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
            // Allocate memory but don't resize
            reserve(context, parms_id, 2);
        }

        /**
        Constructs an empty ciphertext with given capacity. In addition to
        the capacity, the allocation size is determined by the given
        encryption parameters.

        @param[in] context The SEALContext
        @param[in] parms_id The parms_id corresponding to the encryption
        parameters to be used
        @param[in] size_capacity The capacity
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if size_capacity is less than 2 or too large
        @throws std::invalid_argument if pool is uninitialized
        */
        explicit Ciphertext(
            const SEALContext &context, parms_id_type parms_id, std::size_t size_capacity,
            MemoryPoolHandle pool = MemoryManager::GetPool())
            : data_(std::move(pool))
        {
            // Allocate memory but don't resize
            reserve(context, parms_id, size_capacity);
        }

        /**
        Creates a new ciphertext by copying a given one.

        @param[in] copy The ciphertext to copy from
        */
        Ciphertext(const Ciphertext &copy) = default;

        /**
        Creates a new ciphertext by moving a given one.

        @param[in] source The ciphertext to move from
        */
        Ciphertext(Ciphertext &&source) = default;

        /**
        Creates a new ciphertext by copying a given one.

        @param[in] copy The ciphertext to copy from
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if pool is uninitialized
        */
        Ciphertext(const Ciphertext &copy, MemoryPoolHandle pool) : Ciphertext(std::move(pool))
        {
            *this = copy;
        }

        /**
        Allocates enough memory to accommodate the backing array of a ciphertext
        with given capacity. In addition to the capacity, the allocation size is
        determined by the encryption parameters corresponing to the given
        parms_id.

        @param[in] context The SEALContext
        @param[in] parms_id The parms_id corresponding to the encryption
        parameters to be used
        @param[in] size_capacity The capacity
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if size_capacity is less than 2 or too large
        */
        void reserve(const SEALContext &context, parms_id_type parms_id, std::size_t size_capacity);

        /**
        Allocates enough memory to accommodate the backing array of a ciphertext
        with given capacity. In addition to the capacity, the allocation size is
        determined by the highest-level parameters associated to the given
        SEALContext.

        @param[in] context The SEALContext
        @param[in] size_capacity The capacity
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if size_capacity is less than 2 or too large
        */
        inline void reserve(const SEALContext &context, std::size_t size_capacity)
        {
            auto parms_id = context.first_parms_id();
            reserve(context, parms_id, size_capacity);
        }

        /**
        Allocates enough memory to accommodate the backing array of a ciphertext
        with given capacity. In addition to the capacity, the allocation size is
        determined by the current encryption parameters.

        @param[in] size_capacity The capacity
        @throws std::invalid_argument if size_capacity is less than 2 or too large
        @throws std::logic_error if the encryption parameters are not
        */
        inline void reserve(std::size_t size_capacity)
        {
            // Note: poly_modulus_degree_ and coeff_modulus_size_ are either valid
            // or coeff_modulus_size_ is zero (in which case no memory is allocated).
            reserve_internal(size_capacity, poly_modulus_degree_, coeff_modulus_size_);
        }

        /**
        Resizes the ciphertext to given size, reallocating if the capacity
        of the ciphertext is too small. The ciphertext parameters are
        determined by the given SEALContext and parms_id.

        This function is mainly intended for internal use and is called
        automatically by functions such as Evaluator::multiply and
        Evaluator::relinearize. A normal user should never have a reason
        to manually resize a ciphertext.

        @param[in] context The SEALContext
        @param[in] parms_id The parms_id corresponding to the encryption
        parameters to be used
        @param[in] size The new size
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if size is less than 2 or too large
        */
        void resize(const SEALContext &context, parms_id_type parms_id, std::size_t size);

        void resize_pure_gpu(const SEALContext &context, parms_id_type parms_id, size_t size);


        /**
        Resizes the ciphertext to given size, reallocating if the capacity
        of the ciphertext is too small. The ciphertext parameters are
        determined by the highest-level parameters associated to the given
        SEALContext.

        This function is mainly intended for internal use and is called
        automatically by functions such as Evaluator::multiply and
        Evaluator::relinearize. A normal user should never have a reason
        to manually resize a ciphertext.

        @param[in] context The SEALContext
        @param[in] size The new size
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if size is less than 2 or too large
        */
        inline void resize(const SEALContext &context, std::size_t size)
        {
            auto parms_id = context.first_parms_id();
            resize(context, parms_id, size);
        }

        /**
        Resizes the ciphertext to given size, reallocating if the capacity
        of the ciphertext is too small.

        This function is mainly intended for internal use and is called
        automatically by functions such as Evaluator::multiply and
        Evaluator::relinearize. A normal user should never have a reason
        to manually resize a ciphertext.

        @param[in] size The new size
        @throws std::invalid_argument if size is less than 2 or too large
        */
        inline void resize(std::size_t size)
        {
            // Note: poly_modulus_degree_ and coeff_modulus_size_ are either valid
            // or coeff_modulus_size_ is zero (in which case no memory is allocated).
            resize_internal(size, poly_modulus_degree_, coeff_modulus_size_);
        }

        inline void resize_gpu(size_t size, size_t poly_modulus_degree, size_t coeff_modulus_size){
            size_t new_size = size * poly_modulus_degree * coeff_modulus_size;
            resize_gpu(new_size);
        }

        inline void resize_gpu(size_t size) {
            if (size <= d_capacity_){
                if (size > d_size_){
                    checkCudaErrors(cudaMemset(d_data_.data() + d_size_, 0, (size - d_size_) * sizeof(uint64_t)));
                }
                d_size_ = size;
                return;
            }

            std::shared_ptr<uint64_t> new_d_data_;
            allocate_gpu<uint64_t>(new_d_data_, size);

            if (d_size_ > 0){
                checkCudaErrors(cudaMemcpy(new_d_data_.get(), d_data_.data(), d_size_ * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
            }
            if (d_capacity_ > 0){
                d_data_.release();
            }

            if (size > d_size_){
                checkCudaErrors(cudaMemset(new_d_data_.get() + d_size_, 0, (size - d_size_) * sizeof(uint64_t)));
            }

            d_data_.setData(new_d_data_);
            d_capacity_ = size;
            d_size_ = size;
            
        }


        /**
        Resets the ciphertext. This function releases any memory allocated
        by the ciphertext, returning it to the memory pool. It also sets all
        encryption parameter specific size information to zero.
        */
        inline void release() noexcept
        {
            parms_id_ = parms_id_zero;
            is_ntt_form_ = false;
            size_ = 0;
            poly_modulus_degree_ = 0;
            coeff_modulus_size_ = 0;
            scale_ = 1.0;
            correction_factor_ = 1;
            data_.release();
            // deallocate_gpu<uint64_t>(&d_data_, d_capacity_);
            d_data_.release();
            // checkCudaErrors(cudaFree(d_data_));
            d_size_ = 0;
            d_capacity_ = 0;
        }

        /**
        Copies a given ciphertext to the current one.

        @param[in] assign The ciphertext to copy from
        */
        Ciphertext &operator=(const Ciphertext &assign);
        // Ciphertext &operator=(Ciphertext &assign);

        /**
        Moves a given ciphertext to the current one.

        @param[in] assign The ciphertext to move from
        */
        Ciphertext &operator=(Ciphertext &&assign) = default;

        /**
        Returns a reference to the backing DynArray object.
        */
        SEAL_NODISCARD inline const auto &dyn_array() const noexcept
        {
            return data_;
        }

        SEAL_NODISCARD inline uint64_t *d_data() noexcept
        {
            return d_data_.data();
        }

        inline void to_cpu() noexcept
        {
            auto new_data(util::allocate<std::uint64_t>(d_size_, MemoryManager::GetPool()));
            std::swap(data_.data_, new_data);

            checkCudaErrors(cudaMemcpy(data_.begin(), 
                                        d_data_.data(), 
                                        d_size_ * sizeof(std::uint64_t), 
                                        cudaMemcpyDeviceToHost));
        }

        inline void to_gpu() noexcept
        {
            auto poly_uint64_count = util::mul_safe(poly_modulus_degree_, coeff_modulus_size_);
            resize_gpu(size_, poly_modulus_degree_,  coeff_modulus_size_);

            checkCudaErrors(cudaMemcpy(d_data_.data(), 
                                        data_.begin(), 
                                        size_ * poly_uint64_count *sizeof(uint64_t), 
                                        cudaMemcpyHostToDevice));
        }

        SEAL_NODISCARD inline uint64_t *d_data() const noexcept
        {
            return d_data_.data();
        }

        inline void d_data_malloc(size_t size)
        {
            // checkCudaErrors(cudaMalloc((void **)&d_data_, 10 * size * sizeof(uint64_t)));
            // allocate_gpu<uint64_t>(&d_data_, size);
            d_data_.alloc(size);
            d_capacity_ = size;
        }

        /**
        Returns a pointer to the beginning of the ciphertext data.
        */
        SEAL_NODISCARD inline ct_coeff_type *data() noexcept
        {
            return data_.begin();
        }

        /**
        Returns a const pointer to the beginning of the ciphertext data.
        */
        SEAL_NODISCARD inline const ct_coeff_type *data() const noexcept
        {
            return data_.cbegin();
        }

        /**
        Returns a pointer to a particular polynomial in the ciphertext
        data. Note that IDEA SEAL_GPU stores each polynomial in the ciphertext
        modulo all of the K primes in the coefficient modulus. The pointer
        returned by this function is to the beginning (constant coefficient)
        of the first one of these K polynomials.

        @param[in] poly_index The index of the polynomial in the ciphertext
        @throws std::out_of_range if poly_index is less than 0 or bigger
        than the size of the ciphertext
        */
        SEAL_NODISCARD inline ct_coeff_type *data(std::size_t poly_index)
        {
            auto poly_uint64_count = util::mul_safe(poly_modulus_degree_, coeff_modulus_size_);
            if (poly_uint64_count == 0)
            {
                return nullptr;
            }
            if (poly_index >= size_)
            {
                throw std::out_of_range("poly_index must be within [0, size)");
            }
            return data_.begin() + util::safe_cast<std::size_t>(util::mul_safe(poly_index, poly_uint64_count));
        }

        /**
        Returns a const pointer to a particular polynomial in the
        ciphertext data. Note that IDEA SEAL_GPU stores each polynomial in the
        ciphertext modulo all of the K primes in the coefficient modulus.
        The pointer returned by this function is to the beginning
        (constant coefficient) of the first one of these K polynomials.

        @param[in] poly_index The index of the polynomial in the ciphertext
        @throws std::out_of_range if poly_index is out of range
        */
        SEAL_NODISCARD inline const ct_coeff_type *data(std::size_t poly_index) const
        {
            auto poly_uint64_count = util::mul_safe(poly_modulus_degree_, coeff_modulus_size_);
            if (poly_uint64_count == 0)
            {
                return nullptr;
            }
            if (poly_index >= size_)
            {
                throw std::out_of_range("poly_index must be within [0, size)");
            }
            return data_.cbegin() + util::safe_cast<std::size_t>(util::mul_safe(poly_index, poly_uint64_count));
        }

        /**
        Returns a reference to a polynomial coefficient at a particular
        index in the ciphertext data. If the polynomial modulus has degree N,
        and the number of primes in the coefficient modulus is K, then the
        ciphertext contains size*N*K coefficients. Thus, the coeff_index has
        a range of [0, size*N*K).

        @param[in] coeff_index The index of the coefficient
        @throws std::out_of_range if coeff_index is out of range
        */
        SEAL_NODISCARD inline ct_coeff_type &operator[](std::size_t coeff_index)
        {
            return data_.at(coeff_index);
        }

        /**
        Returns a const reference to a polynomial coefficient at a particular
        index in the ciphertext data. If the polynomial modulus has degree N,
        and the number of primes in the coefficient modulus is K, then the
        ciphertext contains size*N*K coefficients. Thus, the coeff_index has
        a range of [0, size*N*K).

        @param[in] coeff_index The index of the coefficient
        @throws std::out_of_range if coeff_index is out of range
        */
        SEAL_NODISCARD inline const ct_coeff_type &operator[](std::size_t coeff_index) const
        {
            return data_.at(coeff_index);
        }

        /**
        Returns the number of primes in the coefficient modulus of the
        associated encryption parameters. This directly affects the
        allocation size of the ciphertext.
        */
        SEAL_NODISCARD inline std::size_t coeff_modulus_size() const noexcept
        {
            return coeff_modulus_size_;
        }

        /**
        Returns the degree of the polynomial modulus of the associated
        encryption parameters. This directly affects the allocation size
        of the ciphertext.
        */
        SEAL_NODISCARD inline std::size_t poly_modulus_degree() const noexcept
        {
            return poly_modulus_degree_;
        }

        /**
        Returns the size of the ciphertext.
        */
        SEAL_NODISCARD inline std::size_t size() const noexcept
        {
            return size_;
        }

        /**
        Returns the capacity of the allocation. This means the largest size
        of the ciphertext that can be stored in the current allocation with
        the current encryption parameters.
        */
        SEAL_NODISCARD inline std::size_t size_capacity() const noexcept
        {
            std::size_t poly_uint64_count = poly_modulus_degree_ * coeff_modulus_size_;
            return poly_uint64_count ? data_.capacity() / poly_uint64_count : std::size_t(0);
        }

        /**
        Check whether the current ciphertext is transparent, i.e. does not require
        a secret key to decrypt. In typical security models such transparent
        ciphertexts would not be considered to be valid. Starting from the second
        polynomial in the current ciphertext, this function returns true if all
        following coefficients are identically zero. Otherwise, returns false.
        */
        SEAL_NODISCARD inline bool is_transparent() const
        {
            return (
                !data_.size() || (size_ < SEAL_CIPHERTEXT_SIZE_MIN) ||
                std::all_of(data(1), data_.cend(), util::is_zero<ct_coeff_type>));
        }

        /**
        Returns an upper bound on the size of the ciphertext, as if it was written
        to an output stream.

        @param[in] compr_mode The compression mode
        @throws std::invalid_argument if the compression mode is not supported
        @throws std::logic_error if the size does not fit in the return type
        */
        SEAL_NODISCARD std::streamoff save_size(compr_mode_type compr_mode = Serialization::compr_mode_default) const;

        /**
        Saves the ciphertext to an output stream. The output is in binary format
        and not human-readable. The output stream must have the "binary" flag set.

        @param[out] stream The stream to save the ciphertext to
        @param[in] compr_mode The desired compression mode
        @throws std::invalid_argument if the compression mode is not supported
        @throws std::logic_error if the data to be saved is invalid, or if
        compression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff save(
            std::ostream &stream, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Ciphertext::save_members, this, _1), save_size(compr_mode_type::none), stream, compr_mode,
                false);
        }

        /**
        Loads a ciphertext from an input stream overwriting the current ciphertext.
        No checking of the validity of the ciphertext data against encryption
        parameters is performed. This function should not be used unless the
        ciphertext comes from a fully trusted source.

        @param[in] context The SEALContext
        @param[in] stream The stream to load the ciphertext from
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::logic_error if the data cannot be loaded by this version of
        IDEA SEAL_GPU, if the loaded data is invalid, or if decompression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff unsafe_load(const SEALContext &context, std::istream &stream)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Ciphertext::load_members, this, context, _1, _2), stream, false);
        }

        /**
        Loads a ciphertext from an input stream overwriting the current ciphertext.
        The loaded ciphertext is verified to be valid for the given SEALContext.

        @param[in] context The SEALContext
        @param[in] stream The stream to load the ciphertext from
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::logic_error if the data cannot be loaded by this version of
        IDEA SEAL_GPU, if the loaded data is invalid, or if decompression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff load(const SEALContext &context, std::istream &stream)
        {
            Ciphertext new_data(pool());
            auto in_size = new_data.unsafe_load(context, stream);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("ciphertext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        /**
        Saves the ciphertext to a given memory location. The output is in binary
        format and is not human-readable.

        @param[out] out The memory location to write the ciphertext to
        @param[in] size The number of bytes available in the given memory location
        @param[in] compr_mode The desired compression mode
        @throws std::invalid_argument if out is null or if size is too small to
        contain a SEALHeader, or if the compression mode is not supported
        @throws std::logic_error if the data to be saved is invalid, or if
        compression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff save(
            seal_byte *out, std::size_t size, compr_mode_type compr_mode = Serialization::compr_mode_default) const
        {
            using namespace std::placeholders;
            return Serialization::Save(
                std::bind(&Ciphertext::save_members, this, _1), save_size(compr_mode_type::none), out, size, compr_mode,
                false);
        }

        /**
        Loads a ciphertext from a given memory location overwriting the current
        ciphertext. No checking of the validity of the ciphertext data against
        encryption parameters is performed. This function should not be used
        unless the ciphertext comes from a fully trusted source.

        @param[in] context The SEALContext
        @param[in] in The memory location to load the ciphertext from
        @param[in] size The number of bytes available in the given memory location
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if in is null or if size is too small to
        contain a SEALHeader
        @throws std::logic_error if the data cannot be loaded by this version of
        IDEA SEAL_GPU, if the loaded data is invalid, or if decompression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff unsafe_load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            using namespace std::placeholders;
            return Serialization::Load(std::bind(&Ciphertext::load_members, this, context, _1, _2), in, size, false);
        }

        /**
        Loads a ciphertext from a given memory location overwriting the current
        ciphertext. The loaded ciphertext is verified to be valid for the given
        SEALContext.

        @param[in] context The SEALContext
        @param[in] in The memory location to load the ciphertext from
        @param[in] size The number of bytes available in the given memory location
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if in is null or if size is too small to
        contain a SEALHeader
        @throws std::logic_error if the data cannot be loaded by this version of
        IDEA SEAL_GPU, if the loaded data is invalid, or if decompression failed
        @throws std::runtime_error if I/O operations failed
        */
        inline std::streamoff load(const SEALContext &context, const seal_byte *in, std::size_t size)
        {
            Ciphertext new_data(pool());
            auto in_size = new_data.unsafe_load(context, in, size);
            if (!is_valid_for(new_data, context))
            {
                throw std::logic_error("ciphertext data is invalid");
            }
            std::swap(*this, new_data);
            return in_size;
        }

        /**
        Returns whether the ciphertext is in NTT form.
        */
        SEAL_NODISCARD inline bool is_ntt_form() const noexcept
        {
            return is_ntt_form_;
        }

        /**
        Returns whether the ciphertext is in NTT form.
        */
        SEAL_NODISCARD inline bool &is_ntt_form() noexcept
        {
            return is_ntt_form_;
        }

        /**
        Returns a reference to parms_id.

        @see EncryptionParameters for more information about parms_id.
        */
        SEAL_NODISCARD inline parms_id_type &parms_id() noexcept
        {
            return parms_id_;
        }

        /**
        Returns a const reference to parms_id.

        @see EncryptionParameters for more information about parms_id.
        */
        SEAL_NODISCARD inline const parms_id_type &parms_id() const noexcept
        {
            return parms_id_;
        }

        /**
        Returns a reference to the scale. This is only needed when using the CKKS encryption scheme. The user should
        have little or no reason to ever change the scale by hand.
        */
        SEAL_NODISCARD inline double &scale() noexcept
        {
            return scale_;
        }

        /**
        Returns a constant reference to the scale. This is only needed when using the CKKS encryption scheme.
        */
        SEAL_NODISCARD inline const double &scale() const noexcept
        {
            return scale_;
        }

        /**
        Returns a reference to the correction factor. This is only needed when using the BGV encryption scheme. The user
        should have little or no reason to ever change the correction factor by hand.
        */
        SEAL_NODISCARD inline std::uint64_t &correction_factor() noexcept
        {
            return correction_factor_;
        }

        /**
        Returns a constant reference to the correction factor. This is only needed when using the BGV encryption scheme.
        */
        SEAL_NODISCARD inline const std::uint64_t &correction_factor() const noexcept
        {
            return correction_factor_;
        }

        /**
        Returns the currently used MemoryPoolHandle.
        */
        SEAL_NODISCARD inline MemoryPoolHandle pool() const noexcept
        {
            return data_.pool();
        }

        /**
        Enables access to private members of seal::Ciphertext for SEAL_C.
        */
        struct CiphertextPrivateHelper;

    private:
        void reserve_internal(
            std::size_t size_capacity, std::size_t poly_modulus_degree, std::size_t coeff_modulus_size);

        void resize_internal(std::size_t size, std::size_t poly_modulus_degree, std::size_t coeff_modulus_size);

        void expand_seed(const SEALContext &context, const UniformRandomGeneratorInfo &prng_info, SEALVersion version);

        void save_members(std::ostream &stream) const;

        void load_members(const SEALContext &context, std::istream &stream, SEALVersion version);

        inline bool has_seed_marker() const noexcept
        {
            return (data_.size() && (size_ == 2)) ? (data(1)[0] == 0xFFFFFFFFFFFFFFFFULL) : false;
        }

        parms_id_type parms_id_ = parms_id_zero;

        bool is_ntt_form_ = false;

        std::size_t size_ = 0;

        std::size_t poly_modulus_degree_ = 0;

        std::size_t coeff_modulus_size_ = 0;

        double scale_ = 1.0;

        std::uint64_t correction_factor_ = 1;

        DynArray<ct_coeff_type> data_;

        // GPU数据
        // uint64_t *d_data_ = nullptr;

        GPUData d_data_;
        size_t d_size_ = 0;
        size_t d_capacity_ = 0;
    };
} // namespace seal
