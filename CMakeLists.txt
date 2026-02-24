cmake_minimum_required(VERSION 3.22)
project(CUDA_CNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# --- BASELINE CUDA ---
# =========================
set(BASELINE_SRC
    src/naive/main.cu
    src/utils/load_data.cu
    src/utils/printing.cu
    src/naive/kernels/forward.cu
    src/naive/kernels/backward.cu
)

add_executable(naive_cnn ${BASELINE_SRC})

set_target_properties(naive_cnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(naive_cnn PRIVATE
    ${CMAKE_SOURCE_DIR}      # config.h
    ${CMAKE_SOURCE_DIR}/src  # utils/, naive/, optimized/
    src/naive
    src/naive/kernels
)

# =========================
# --- OPTIMIZED CUDA ---
# =========================
set(OPTIMIZED_SRC
    src/optimized/main.cu
    src/utils/load_data.cu
    src/utils/printing.cu
    src/optimized/kernels/forward.cu
    src/optimized/kernels/backward.cu
)

add_executable(optimized_cnn ${OPTIMIZED_SRC})

set_target_properties(optimized_cnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(optimized_cnn PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    src/optimized
    src/optimized/kernels
)
