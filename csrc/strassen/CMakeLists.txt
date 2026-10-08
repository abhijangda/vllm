cmake_minimum_required(VERSION 3.26)
project(vllm_strassen LANGUAGES CXX CUDA)

set(STRASSEN_SOURCE_DIR "" CACHE PATH "Path to the strassen-merged checkout")
if(NOT EXISTS "${STRASSEN_SOURCE_DIR}/include/cutlass/gemm/device/strassen_decls.h")
  message(FATAL_ERROR "STRASSEN_SOURCE_DIR must point to a strassen-merged checkout")
endif()

add_library(vllm_strassen SHARED bf16_gemm.cu)
target_include_directories(vllm_strassen PRIVATE
  "${STRASSEN_SOURCE_DIR}/include"
  "${STRASSEN_SOURCE_DIR}/cutlass/include"
  "${STRASSEN_SOURCE_DIR}/cutlass/tools/util/include")
set_target_properties(vllm_strassen PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON)
target_compile_options(vllm_strassen PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--generate-code=arch=compute_90a,code=sm_90a>)
target_compile_definitions(vllm_strassen PRIVATE
  NDEBUG
  CUTLASS_API_v3
  CUTLASS_ENABLE_TENSOR_CORE_MMA=1
  CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
