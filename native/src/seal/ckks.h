// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.cuh"
#include "seal/plaintext.cuh"
#include "seal/util/common.cuh"
#include "seal/util/common.h"
#include "seal/util/croots.h"
#include "seal/util/defines.h"
#include "seal/util/dwthandler.h"
#include "seal/util/polyarithsmallmod.cuh"
#include "seal/util/rns.cuh"
#include "seal/util/uintarithmod.cuh"
#include "seal/util/uintarithsmallmod.cuh"
#include "seal/util/uintcore.h"
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef SEAL_USE_MSGSL
#include "gsl/span"
#endif

namespace seal
{
    template <
        typename T_out, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T_out>, double>::value ||
                            std::is_same<std::remove_cv_t<T_out>, std::complex<double>>::value>>
    SEAL_NODISCARD inline T_out from_complex(std::complex<double> in);

    template <>
    SEAL_NODISCARD inline double from_complex(std::complex<double> in)
    {
        return in.real();
    }

    template <>
    SEAL_NODISCARD inline std::complex<double> from_complex(std::complex<double> in)
    {
        return in;
    }
    namespace util
    {
        template <>
        class Arithmetic<std::complex<double>, std::complex<double>, double>
        {
        public:
            Arithmetic()
            {}

            inline std::complex<double> add(const std::complex<double> &a, const std::complex<double> &b) const
            {
                return a + b;
            }

            inline std::complex<double> sub(const std::complex<double> &a, const std::complex<double> &b) const
            {
                return a - b;
            }

            inline std::complex<double> mul_root(const std::complex<double> &a, const std::complex<double> &r) const
            {
                return a * r;
            }

            inline std::complex<double> mul_scalar(const std::complex<double> &a, const double &s) const
            {
                return a * s;
            }

            inline std::complex<double> mul_root_scalar(const std::complex<double> &r, const double &s) const
            {
                return r * s;
            }

            inline std::complex<double> guard(const std::complex<double> &a) const
            {
                return a;
            }
        };
    } // namespace util

    /**
    Provides functionality for encoding vectors of complex or real numbers into
    plaintext polynomials to be encrypted and computed on using the CKKS scheme.
    If the polynomial modulus degree is N, then CKKSEncoder converts vectors of
    N/2 complex numbers into plaintext elements. Homomorphic operations performed
    on such encrypted vectors are applied coefficient (slot-)wise, enabling
    powerful SIMD functionality for computations that are vectorizable. This
    functionality is often called "batching" in the homomorphic encryption
    literature.

    @par Mathematical Background
    Mathematically speaking, if the polynomial modulus is X^N+1, N is a power of
    two, the CKKSEncoder implements an approximation of the canonical embedding
    of the ring of integers Z[X]/(X^N+1) into C^(N/2), where C denotes the complex
    numbers. The Galois group of the extension is (Z/2NZ)* ~= Z/2Z x Z/(N/2)
    whose action on the primitive roots of unity modulo coeff_modulus is easy to
    describe. Since the batching slots correspond 1-to-1 to the primitive roots
    of unity, applying Galois automorphisms on the plaintext acts by permuting
    the slots. By applying generators of the two cyclic subgroups of the Galois
    group, we can effectively enable cyclic rotations and complex conjugations
    of the encrypted complex vectors.
    */
    class CKKSEncoder
    {
        using ComplexArith = util::Arithmetic<std::complex<double>, std::complex<double>, double>;
        using FFTHandler = util::DWTHandler<std::complex<double>, std::complex<double>, double>;

    public:
        /**
        Creates a CKKSEncoder instance initialized with the specified SEALContext.

        @param[in] context The SEALContext
        @throws std::invalid_argument if the encryption parameters are not valid
        @throws std::invalid_argument if scheme is not scheme_type::CKKS
        */
        CKKSEncoder(const SEALContext &context);

        /**
        Encodes a vector of double-precision floating-point real or complex numbers
        into a plaintext polynomial. Append zeros if vector size is less than N/2.
        Dynamic memory allocations in the process are allocated from the memory
        pool pointed to by the given MemoryPoolHandle.

        @tparam T Vector value type (double or std::complex<double>)
        @param[in] values The vector of double-precision floating-point numbers
        (of type T) to encode
        @param[in] parms_id parms_id determining the encryption parameters to
        be used by the result plaintext
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if values has invalid size
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void encode(
            const std::vector<T> &values, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode_internal(values.data(), values.size(), parms_id, scale, destination, std::move(pool));
        }

        /**
        Encodes a vector of double-precision floating-point real or complex numbers
        into a plaintext polynomial. Append zeros if vector size is less than N/2.
        The encryption parameters used are the top level parameters for the given
        context. Dynamic memory allocations in the process are allocated from the
        memory pool pointed to by the given MemoryPoolHandle.

        @tparam T Vector value type (double or std::complex<double>)
        @param[in] values The vector of double-precision floating-point numbers
        (of type T) to encode
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if values has invalid size
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void encode(
            const std::vector<T> &values, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode(values, context_.first_parms_id(), scale, destination, std::move(pool));
        }
#ifdef SEAL_USE_MSGSL
        /**
        Encodes a vector of double-precision floating-point real or complex numbers
        into a plaintext polynomial. Append zeros if vector size is less than N/2.
        Dynamic memory allocations in the process are allocated from the memory
        pool pointed to by the given MemoryPoolHandle.

        @tparam T Array value type (double or std::complex<double>)
        @param[in] values The array of double-precision floating-point numbers
        (of type T) to encode
        @param[in] parms_id parms_id determining the encryption parameters to
        be used by the result plaintext
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if values has invalid size
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void encode(
            gsl::span<const T> values, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode_internal(
                values.data(), static_cast<std::size_t>(values.size()), parms_id, scale, destination, std::move(pool));
        }

        /**
        Encodes a vector of double-precision floating-point real or complex numbers
        into a plaintext polynomial. Append zeros if vector size is less than N/2.
        The encryption parameters used are the top level parameters for the given
        context. Dynamic memory allocations in the process are allocated from the
        memory pool pointed to by the given MemoryPoolHandle.

        @tparam T Array value type (double or std::complex<double>)
        @param[in] values The array of double-precision floating-point numbers
        (of type T) to encode
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if values has invalid size
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void encode(
            gsl::span<const T> values, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode(values, context_.first_parms_id(), scale, destination, std::move(pool));
        }
#endif
        /**
        Encodes a double-precision floating-point real number into a plaintext
        polynomial. The number repeats for N/2 times to fill all slots. Dynamic
        memory allocations in the process are allocated from the memory pool
        pointed to by the given MemoryPoolHandle.

        @param[in] value The double-precision floating-point number to encode
        @param[in] parms_id parms_id determining the encryption parameters to be
        used by the result plaintext
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        inline void encode(
            double value, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            // cudaStream_t stream;
            // cudaStreamCreate(&stream);
            encode_internal(value, parms_id, scale, destination, std::move(pool));
        }

        /**
        Encodes a double-precision floating-point real number into a plaintext
        polynomial. The number repeats for N/2 times to fill all slots. The
        encryption parameters used are the top level parameters for the given
        context. Dynamic memory allocations in the process are allocated from
        the memory pool pointed to by the given MemoryPoolHandle.

        @param[in] value The double-precision floating-point number to encode
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        inline void encode(
            double value, double scale, Plaintext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
        }

        /**
        Encodes a double-precision complex number into a plaintext polynomial.
        Append zeros to fill all slots. Dynamic memory allocations in the process
        are allocated from the memory pool pointed to by the given MemoryPoolHandle.

        @param[in] value The double-precision complex number to encode
        @param[in] parms_id parms_id determining the encryption parameters to be
        used by the result plaintext
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        inline void encode(
            std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode_internal(value, parms_id, scale, destination, std::move(pool));
        }

        /**
        Encodes a double-precision complex number into a plaintext polynomial.
        Append zeros to fill all slots. The encryption parameters used are the
        top level parameters for the given context. Dynamic memory allocations
        in the process are allocated from the memory pool pointed to by the
        given MemoryPoolHandle.

        @param[in] value The double-precision complex number to encode
        @param[in] scale Scaling parameter defining encoding precision
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if scale is not strictly positive
        @throws std::invalid_argument if encoding is too large for the encryption
        parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        inline void encode(
            std::complex<double> value, double scale, Plaintext &destination,
            MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
        }

        /**
        Encodes an integer number into a plaintext polynomial without any scaling.
        The number repeats for N/2 times to fill all slots.
        @param[in] value The integer number to encode
        @param[in] parms_id parms_id determining the encryption parameters to be
        used by the result plaintext
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        @throws std::invalid_argument if parms_id is not valid for the encryption
        parameters
        */
        inline void encode(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const
        {
            encode_internal(value, parms_id, destination);
        }

        /**
        Encodes an integer number into a plaintext polynomial without any scaling.
        The number repeats for N/2 times to fill all slots. The encryption
        parameters used are the top level parameters for the given context.

        @param[in] value The integer number to encode
        @param[out] destination The plaintext polynomial to overwrite with the
        result
        */
        inline void encode(std::int64_t value, Plaintext &destination) const
        {
            encode(value, context_.first_parms_id(), destination);
        }

        /**
        Decodes a plaintext polynomial into double-precision floating-point
        real or complex numbers. Dynamic memory allocations in the process are
        allocated from the memory pool pointed to by the given MemoryPoolHandle.

        @tparam T Vector value type (double or std::complex<double>)
        @param[in] plain The plaintext to decode
        @param[out] destination The vector to be overwritten with the values in
        the slots
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if plain is not in NTT form or is invalid
        for the encryption parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void decode(
            const Plaintext &plain, std::vector<T> &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            destination.resize(slots_);
            decode_internal(plain, destination.data(), std::move(pool));
        }
#ifdef SEAL_USE_MSGSL
        /**
        Decodes a plaintext polynomial into double-precision floating-point
        real or complex numbers. Dynamic memory allocations in the process are
        allocated from the memory pool pointed to by the given MemoryPoolHandle.

        @tparam T Array value type (double or std::complex<double>)
        @param[in] plain The plaintext to decode
        @param[out] destination The array to be overwritten with the values in
        the slots
        @param[in] pool The MemoryPoolHandle pointing to a valid memory pool
        @throws std::invalid_argument if plain is not in NTT form or is invalid
        for the encryption parameters
        @throws std::invalid_argument if pool is uninitialized
        */
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        inline void decode(
            const Plaintext &plain, gsl::span<T> destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
        {
            if (destination.size() != slots_)
            {
                throw std::invalid_argument("destination has invalid size");
            }
            decode_internal(plain, destination.data(), std::move(pool));
        }
#endif
        /**
        Returns the number of complex numbers encoded.
        */
        SEAL_NODISCARD inline std::size_t slot_count() const noexcept
        {
            return slots_;
        }

    private:
        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        void encode_internal(
            const T *values, std::size_t values_size, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool) const;      

        template <
            typename T, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T>, double>::value ||
                            std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
        void decode_internal(const Plaintext &plain, T *destination, MemoryPoolHandle pool) const;

        void encode_internal(
            double value, parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool) const;

        // void encode_internal(
        //     double value, parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool) const;

        inline void encode_internal(
            std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool) const
        {
            auto input = util::allocate<std::complex<double>>(slots_, pool_, value);
            encode_internal(input.get(), slots_, parms_id, scale, destination, std::move(pool));
        }

        void encode_internal(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const;

        MemoryPoolHandle pool_ = MemoryManager::GetPool();

        SEALContext context_;

        std::size_t slots_;

        std::shared_ptr<util::ComplexRoots> complex_roots_;

        // Holds 1~(n-1)-th powers of root in bit-reversed order, the 0-th power is left unset.
        util::Pointer<std::complex<double>> root_powers_;

        // Holds 1~(n-1)-th powers of inverse root in scrambled order, the 0-th power is left unset.
        util::Pointer<std::complex<double>> inv_root_powers_;

        util::Pointer<std::size_t> matrix_reps_index_map_;

        ComplexArith complex_arith_;

        FFTHandler fft_handler_;
    };
} // namespace seal
