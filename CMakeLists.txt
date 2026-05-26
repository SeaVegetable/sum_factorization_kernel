cmake_minimum_required(VERSION 3.16)

project(sum_factorization_kernel LANGUAGES CXX)

include(CTest)

add_library(sum_factorization INTERFACE)
target_include_directories(sum_factorization
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(sum_factorization INTERFACE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(sum_factorization
        INTERFACE
            -Wall
            -Wextra
    )
elseif(MSVC)
    target_compile_options(sum_factorization
        INTERFACE
            /W4
    )
endif()

add_executable(test_sum_factorization
    tests/test_sum_factorization.cpp
)
target_link_libraries(test_sum_factorization PRIVATE sum_factorization)

if(BUILD_TESTING)
    add_test(NAME test_sum_factorization COMMAND $<TARGET_FILE:test_sum_factorization>)
endif()

add_executable(bench_sum_factorization
    bench/bench_sum_factorization.cpp
)
target_link_libraries(bench_sum_factorization PRIVATE sum_factorization)
