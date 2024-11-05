# Copyright (c) IDEA Corporation. All rights reserved.
# Licensed under the MIT license.


set(GPU_SUPPORT_OPTION_STR "Use GPU to accelerate computation")
option(GPU_SUPPORT ${GPU_SUPPORT_OPTION_STR} OFF)
message(STATUS "GPU_SUPPORT: ${GPU_SUPPORT}")

if(GPU_SUPPORT)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD 17)
endif()
# Source files in this directory

if(GPU_SUPPORT)

    set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES}

        # ${CMAKE_CURRENT_LIST_DIR}/batchencoder.cu
        ${CMAKE_CURRENT_LIST_DIR}/batchencoder.cu
        ${CMAKE_CURRENT_LIST_DIR}/ciphertext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ckks.cu
        ${CMAKE_CURRENT_LIST_DIR}/context.cu
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.cu
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptor.cu
        ${CMAKE_CURRENT_LIST_DIR}/evaluator.cu
        ${CMAKE_CURRENT_LIST_DIR}/keygenerator.cu
        ${CMAKE_CURRENT_LIST_DIR}/kswitchkeys.cpp
        ${CMAKE_CURRENT_LIST_DIR}/memorymanager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/modulus.cpp
        ${CMAKE_CURRENT_LIST_DIR}/plaintext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/randomgen.cpp
        ${CMAKE_CURRENT_LIST_DIR}/serialization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/valcheck.cpp
    )
else()
    set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/batchencoder.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ciphertext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ckks.cpp
        ${CMAKE_CURRENT_LIST_DIR}/context.cpp
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.cpp
        ${CMAKE_CURRENT_LIST_DIR}/encryptor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/evaluator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/keygenerator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/kswitchkeys.cpp
        ${CMAKE_CURRENT_LIST_DIR}/memorymanager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/modulus.cpp
        ${CMAKE_CURRENT_LIST_DIR}/plaintext.cpp
        ${CMAKE_CURRENT_LIST_DIR}/randomgen.cpp
        ${CMAKE_CURRENT_LIST_DIR}/serialization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/valcheck.cpp
    )
endif()
# Add header files for installation
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/batchencoder.h
        ${CMAKE_CURRENT_LIST_DIR}/ciphertext.cuh
        ${CMAKE_CURRENT_LIST_DIR}/ckks.h
        ${CMAKE_CURRENT_LIST_DIR}/modulus.h
        ${CMAKE_CURRENT_LIST_DIR}/context.cuh
        ${CMAKE_CURRENT_LIST_DIR}/decryptor.h
        ${CMAKE_CURRENT_LIST_DIR}/dynarray.h
        ${CMAKE_CURRENT_LIST_DIR}/encryptionparams.cuh
        ${CMAKE_CURRENT_LIST_DIR}/encryptor.h
        ${CMAKE_CURRENT_LIST_DIR}/evaluator.h
        ${CMAKE_CURRENT_LIST_DIR}/galoiskeys.h
        ${CMAKE_CURRENT_LIST_DIR}/keygenerator.h
        ${CMAKE_CURRENT_LIST_DIR}/kswitchkeys.h
        ${CMAKE_CURRENT_LIST_DIR}/memorymanager.h
        ${CMAKE_CURRENT_LIST_DIR}/plaintext.cuh
        ${CMAKE_CURRENT_LIST_DIR}/publickey.h
        ${CMAKE_CURRENT_LIST_DIR}/randomgen.h
        ${CMAKE_CURRENT_LIST_DIR}/randomtostd.h
        ${CMAKE_CURRENT_LIST_DIR}/relinkeys.h
        ${CMAKE_CURRENT_LIST_DIR}/seal.h
        ${CMAKE_CURRENT_LIST_DIR}/secretkey.cuh
        ${CMAKE_CURRENT_LIST_DIR}/serializable.h
        ${CMAKE_CURRENT_LIST_DIR}/serialization.h
        ${CMAKE_CURRENT_LIST_DIR}/valcheck.h
        ${CMAKE_CURRENT_LIST_DIR}/version.h
        ${CMAKE_CURRENT_LIST_DIR}/gpu_memorypool.h

    DESTINATION
        ${SEAL_INCLUDES_INSTALL_DIR}/seal
)

add_subdirectory(util)
set(SEAL_SOURCE_FILES ${SEAL_SOURCE_FILES} PARENT_SCOPE)
