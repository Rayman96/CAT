// Copyright (c) IDEA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __STDC_WANT_LIB_EXT1__
#define __STDC_WANT_LIB_EXT1__ 1
#endif

#include "seal/util/common.h"
#include <string.h>

#if (SEAL_SYSTEM == SEAL_SYSTEM_WINDOWS)
#include <Windows.h>
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        void seal_memzero(void *data, size_t size)
        {
#if (SEAL_SYSTEM == SEAL_SYSTEM_WINDOWS)
            SecureZeroMemory(data, size);
#elif defined(SEAL_USE_MEMSET_S)
            if (size > 0U && memset_s(data, static_cast<rsize_t>(size), 0, static_cast<rsize_t>(size)) != 0)
            {
                throw runtime_error("error calling memset_s");
            }
#elif defined(SEAL_USE_EXPLICIT_BZERO)
            explicit_bzero(data, size);
#elif defined(SEAL_USE_EXPLICIT_MEMSET)
            explicit_memset(data, 0, size);
#else
            volatile seal_byte *data_ptr = reinterpret_cast<seal_byte *>(data);
            while (size--)
            {
                *data_ptr++ = seal_byte{};
            }
#endif
        }
    } // namespace util
} // namespace seal
