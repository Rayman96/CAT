# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
project(SEALExamples VERSION 4.1.1 LANGUAGES CXX CUDA)
# set(CMAKE_EXE_LINKER_FLAGS "-static")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# If not called from root CMakeLists.txt
if(NOT DEFINED SEAL_BUILD_EXAMPLES)
    set(SEAL_BUILD_EXAMPLES ON)

    # Import Microsoft SEAL
    find_package(SEAL 4.1.1 EXACT REQUIRED)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

if(SEAL_BUILD_EXAMPLES)
    add_executable(sealexamples)
    # target_link_libraries(sealexamples PRIVATE "/home/liqirui/SEAL/build/lib/libseal-4.1.a")
    # target_include_directories(sealexamples PRIVATE "/home/liqirui/SEAL/native/src/")
    # target_include_directories(sealexamples PRIVATE "/home/liqirui/SEAL/native/src/seal/util/config.h")
    target_sources(sealexamples
        PRIVATE
            # ${CMAKE_CURRENT_LIST_DIR}/examples.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/1_bfv_basics.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/2_encoders.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/3_levels.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/4_bgv_basics.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/5_ckks_basics.cpp
            ${CMAKE_CURRENT_LIST_DIR}/5_ckks_basics_copy.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/6_rotation.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/7_serialization.cpp
            # ${CMAKE_CURRENT_LIST_DIR}/8_performance.cpp
    )

    if(TARGET SEAL::seal)
        target_link_libraries(sealexamples PRIVATE SEAL::seal)
        message(STATUS "Using SEAL::seal")
    elseif(TARGET SEAL::seal_shared)
        set_target_properties(sealexamples PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            POSITION_INDEPENDENT_CODE ON
        )

        target_link_libraries(sealexamples PRIVATE SEAL::seal_shared  "/usr/local/cuda-11.7/lib64/libcudart.so.11.0")
    else()
        message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
    endif()
endif()
