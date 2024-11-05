# Copyright (c) IDEA Corporation. All rights reserved.
# Licensed under the MIT license.
set(GPU_SUPPORT_OPTION_STR "Use GPU to accelerate computation")
option(GPU_SUPPORT ${GPU_SUPPORT_OPTION_STR} ON)
message(STATUS "GPU_SUPPORT: ${GPU_SUPPORT}")

if(GPU_SUPPORT)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD 17)
endif()
# Source files in this directory
if(GPU_SUPPORT)
    set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/helper.cu
    ${CMAKE_CURRENT_LIST_DIR}/ntt_60bit.cu


    ${CMAKE_CURRENT_LIST_DIR}/blake2b.c
    ${CMAKE_CURRENT_LIST_DIR}/blake2xb.c
    ${CMAKE_CURRENT_LIST_DIR}/clipnormal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/croots.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fips202.c
    ${CMAKE_CURRENT_LIST_DIR}/globals.cpp
    ${CMAKE_CURRENT_LIST_DIR}/galois.cu
    ${CMAKE_CURRENT_LIST_DIR}/hash.cpp
    ${CMAKE_CURRENT_LIST_DIR}/iterator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mempool.cpp
    ${CMAKE_CURRENT_LIST_DIR}/numth.cpp
    ${CMAKE_CURRENT_LIST_DIR}/polyarithsmallmod.cu
    ${CMAKE_CURRENT_LIST_DIR}/rlwe.cu
    ${CMAKE_CURRENT_LIST_DIR}/rns.cu
    ${CMAKE_CURRENT_LIST_DIR}/scalingvariant.cu
    ${CMAKE_CURRENT_LIST_DIR}/ntt.cpp
    ${CMAKE_CURRENT_LIST_DIR}/streambuf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/uintarith.cu
    ${CMAKE_CURRENT_LIST_DIR}/uintarithmod.cu
    ${CMAKE_CURRENT_LIST_DIR}/uintarithsmallmod.cu
    ${CMAKE_CURRENT_LIST_DIR}/uintcore.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ztools.cpp

    ${CMAKE_CURRENT_LIST_DIR}/ntt_60bit.cu
    ${CMAKE_CURRENT_LIST_DIR}/helper.cu
    ${CMAKE_CURRENT_LIST_DIR}/ntt_helper.cu
    

    )

else()
    set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/blake2b.c
        ${CMAKE_CURRENT_LIST_DIR}/blake2xb.c
        ${CMAKE_CURRENT_LIST_DIR}/clipnormal.cpp
        ${CMAKE_CURRENT_LIST_DIR}/common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/croots.cpp
        ${CMAKE_CURRENT_LIST_DIR}/fips202.c
        ${CMAKE_CURRENT_LIST_DIR}/globals.cpp
        ${CMAKE_CURRENT_LIST_DIR}/galois.cpp
        ${CMAKE_CURRENT_LIST_DIR}/hash.cpp
        ${CMAKE_CURRENT_LIST_DIR}/iterator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/mempool.cpp
        ${CMAKE_CURRENT_LIST_DIR}/numth.cpp
        ${CMAKE_CURRENT_LIST_DIR}/polyarithsmallmod.cpp
        ${CMAKE_CURRENT_LIST_DIR}/rlwe.cpp
        ${CMAKE_CURRENT_LIST_DIR}/rns.cpp
        ${CMAKE_CURRENT_LIST_DIR}/scalingvariant.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ntt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/streambuf.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uintarith.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uintarithmod.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uintarithsmallmod.cpp
        ${CMAKE_CURRENT_LIST_DIR}/uintcore.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ztools.cpp
    )

endif()

# Add header files for installation
if(GPU_SUPPORT)
    install(
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/blake2.h
            ${CMAKE_CURRENT_LIST_DIR}/blake2-impl.h
            ${CMAKE_CURRENT_LIST_DIR}/clang.h
            ${CMAKE_CURRENT_LIST_DIR}/clipnormal.h
            ${CMAKE_CURRENT_LIST_DIR}/common.h
            ${CMAKE_CURRENT_LIST_DIR}/common.cuh

            ${CMAKE_CURRENT_LIST_DIR}/helper.cuh
            ${CMAKE_CURRENT_LIST_DIR}/ntt_60bit.cuh
            ${CMAKE_CURRENT_LIST_DIR}/kuint128.cuh
            ${CMAKE_CURRENT_LIST_DIR}/ntt_helper.cuh

            ${CMAKE_CURRENT_LIST_DIR}/croots.h
            ${CMAKE_CURRENT_LIST_DIR}/defines.h
            ${CMAKE_CURRENT_LIST_DIR}/dwthandler.h
            ${CMAKE_CURRENT_LIST_DIR}/fips202.h
            ${CMAKE_CURRENT_LIST_DIR}/galois.h
            ${CMAKE_CURRENT_LIST_DIR}/gcc.h
            ${CMAKE_CURRENT_LIST_DIR}/globals.h
            ${CMAKE_CURRENT_LIST_DIR}/hash.h
            ${CMAKE_CURRENT_LIST_DIR}/hestdparms.h
            ${CMAKE_CURRENT_LIST_DIR}/iterator.h
            ${CMAKE_CURRENT_LIST_DIR}/locks.h
            ${CMAKE_CURRENT_LIST_DIR}/mempool.h
            ${CMAKE_CURRENT_LIST_DIR}/msvc.h
            ${CMAKE_CURRENT_LIST_DIR}/numth.h
            ${CMAKE_CURRENT_LIST_DIR}/pointer.h
            ${CMAKE_CURRENT_LIST_DIR}/polyarithsmallmod.cuh
            ${CMAKE_CURRENT_LIST_DIR}/polycore.h
            ${CMAKE_CURRENT_LIST_DIR}/rlwe.h
            ${CMAKE_CURRENT_LIST_DIR}/rns.cuh
            ${CMAKE_CURRENT_LIST_DIR}/scalingvariant.h
            ${CMAKE_CURRENT_LIST_DIR}/ntt.h
            ${CMAKE_CURRENT_LIST_DIR}/streambuf.h
            ${CMAKE_CURRENT_LIST_DIR}/uintarith.cuh
            # ${CMAKE_CURRENT_LIST_DIR}/uintarithmod.h
            ${CMAKE_CURRENT_LIST_DIR}/uintarithmod.cuh
            ${CMAKE_CURRENT_LIST_DIR}/uintarithsmallmod.cuh
            ${CMAKE_CURRENT_LIST_DIR}/uintcore.h
            ${CMAKE_CURRENT_LIST_DIR}/ztools.h
            ${CMAKE_CURRENT_LIST_DIR}/gpu_data.h
        DESTINATION
            ${SEAL_INCLUDES_INSTALL_DIR}/seal/util
    )
else()
    install(
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/blake2.h
            ${CMAKE_CURRENT_LIST_DIR}/blake2-impl.h
            ${CMAKE_CURRENT_LIST_DIR}/clang.h
            ${CMAKE_CURRENT_LIST_DIR}/clipnormal.h
            ${CMAKE_CURRENT_LIST_DIR}/common.h
            ${CMAKE_CURRENT_LIST_DIR}/common.cuh

            ${CMAKE_CURRENT_LIST_DIR}/helper.cuh
            ${CMAKE_CURRENT_LIST_DIR}/ntt_60bit.cuh
            ${CMAKE_CURRENT_LIST_DIR}/kuint128.cuh

            ${CMAKE_CURRENT_LIST_DIR}/croots.h
            ${CMAKE_CURRENT_LIST_DIR}/defines.h
            ${CMAKE_CURRENT_LIST_DIR}/dwthandler.h
            ${CMAKE_CURRENT_LIST_DIR}/fips202.h
            ${CMAKE_CURRENT_LIST_DIR}/galois.h
            ${CMAKE_CURRENT_LIST_DIR}/gcc.h
            ${CMAKE_CURRENT_LIST_DIR}/globals.h
            ${CMAKE_CURRENT_LIST_DIR}/hash.h
            ${CMAKE_CURRENT_LIST_DIR}/hestdparms.h
            ${CMAKE_CURRENT_LIST_DIR}/iterator.h
            ${CMAKE_CURRENT_LIST_DIR}/locks.h
            ${CMAKE_CURRENT_LIST_DIR}/mempool.h
            ${CMAKE_CURRENT_LIST_DIR}/msvc.h
            ${CMAKE_CURRENT_LIST_DIR}/numth.h
            ${CMAKE_CURRENT_LIST_DIR}/pointer.h
            ${CMAKE_CURRENT_LIST_DIR}/polyarithsmallmod.h
            ${CMAKE_CURRENT_LIST_DIR}/polycore.h
            ${CMAKE_CURRENT_LIST_DIR}/rlwe.h
            ${CMAKE_CURRENT_LIST_DIR}/rns.cuh
            ${CMAKE_CURRENT_LIST_DIR}/scalingvariant.h
            ${CMAKE_CURRENT_LIST_DIR}/ntt.h
            ${CMAKE_CURRENT_LIST_DIR}/streambuf.h
            ${CMAKE_CURRENT_LIST_DIR}/uintarith.cuh
            ${CMAKE_CURRENT_LIST_DIR}/uintarithmod.h
            ${CMAKE_CURRENT_LIST_DIR}/uintarithsmallmod.h
            ${CMAKE_CURRENT_LIST_DIR}/uintcore.h
            ${CMAKE_CURRENT_LIST_DIR}/ztools.h
        DESTINATION
            ${SEAL_INCLUDES_INSTALL_DIR}/seal/util
    )
endif()

set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES} PARENT_SCOPE)
