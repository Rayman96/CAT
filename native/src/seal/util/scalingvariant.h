// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/context.cuh"
#include "seal/memorymanager.h"
#include "seal/plaintext.cuh"
#include "seal/util/iterator.h"
#include <cstdint>

namespace seal
{
    namespace util
    {
        void add_plain_without_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination);

        void sub_plain_without_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination);

        void multiply_add_plain_with_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination);
        
        void multiply_add_plain_with_scaling_variant_cuda(
            const Plaintext &plain, const SEALContext::ContextData &context_data, uint64_t *destination);

        void multiply_sub_plain_with_scaling_variant(
            const Plaintext &plain, const SEALContext::ContextData &context_data, RNSIter destination);

        void multiply_sub_plain_with_scaling_variant_cuda(
            const Plaintext &plain, const SEALContext::ContextData &context_data, uint64_t *destination);
    } // namespace util
} // namespace seal
