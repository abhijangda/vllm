// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation.
// Adapted from strassen-merged's cooperative_max_fusion_tma_reduce_2x256 OptNo.

#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

#define MY_PRINTF(...) ((void)0)

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/layout/strassen_layout.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace vllm_strassen {
using namespace cute;
using namespace MmaStrassen;
using Element = cutlass::bfloat16_t;
using Presums0 = AllPresums<PresumCompute, PresumCompute, PresumCompute,
                            PresumCompute, PresumAvailable, PresumAvailable,
                            PresumAvailable, PresumAvailable>;
using PresumsRest =
    AllPresums<PresumAvailable, PresumAvailable, PresumAvailable,
               PresumAvailable, PresumAvailable, PresumAvailable,
               PresumAvailable, PresumAvailable>;

int status_code(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? 0 : 1000 + int(status);
}

template <typename Cluster, bool Pingpong = false, int PresumRows = 2>
struct StrassenKernel {
  using Tile = std::conditional_t<Pingpong, Shape<_128, _128, _64>,
                                  Shape<_128, _256, _64>>;
  static constexpr int Stages = Pingpong ? 6 : 4;
  using Schedule =
      std::conditional_t<Pingpong,
                         cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                         cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using Epilogue =
      std::conditional_t<Pingpong, cutlass::epilogue::TmaWarpSpecialized,
                         cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using PresumTile = std::conditional_t<Pingpong, Shape<Int<PresumRows>, _128>,
                                        Shape<Int<PresumRows>, _256>>;
  using Writes236 = RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>,
                                 Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                             CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>,
                             CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>;
  using Writes45 = RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>,
                                Expr<Plus<3, MemGlobal, LayoutFinal>>>,
                            CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>,
                                Expr<Plus<1, MemGlobal, LayoutFinal>>>>;
  using Groups = StrassenLevel1Groups<
      StrassenPresum<1, 0, Tile, Presums0>,
      StrassenLevel1MiGroup<
          1, 0, Tile, Cluster, Stages, RWMTypes<>,
          RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                   CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
          Presums0, 0, 0, 1>,
      StrassenLevel1M1Group<
          1, 0, Tile, Cluster, Stages, RWMTypes<>,
          RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>,
                       Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
          Presums0>,
      StrassenLevel1MiGroup<1, 0, Tile, Cluster, Stages, RWMTypes<>, Writes236,
                            PresumsRest, 0, 2, 3, 6>,
      StrassenLevel1M3Group<
          1, 0, Tile, Cluster, Stages, RWMTypes<>,
          RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>,
                       Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
          PresumsRest>,
      StrassenLevel1MiGroup<1, 0, Tile, Cluster, Stages, RWMTypes<>, Writes45,
                            PresumsRest, 0, 4, 5>,
      StrassenLevel1M5Group<
          1, 0, Tile, Cluster, Stages, RWMTypes<>,
          RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>,
                       Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                            Plus<0, MemGlobal, LayoutInterim1D>>>>,
          PresumsRest>,
      StrassenLevel1M6Group<
          1, 0, Tile, Cluster, Stages, RWMTypes<>,
          RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>,
                       Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
          PresumsRest>>;
  using Schedules = std::conditional_t<
      PresumRows == 4,
      ScheduleStrassenGroups<
          ParallelMiGroups<Schedule, Epilogue, false, FusedMiGroup<7, 0>>,
          ParallelMiGroups<Schedule, Epilogue, false, FusedMiGroup<7, 2>>,
          ParallelMiGroups<Schedule, Epilogue, false, FusedMiGroup<7, 4>>>,
      ScheduleStrassenGroups<
          ParallelMiGroups<Schedule, Epilogue, false, FusedMiGroup<7, 0>>,
          ParallelMiGroups<Schedule, Epilogue, false, FusedMiGroup<7, 2>,
                           FusedMiGroup<7, 4>>>>;
  using Kernels = cutlass::gemm::device::StrassenGemmKernels<
      Groups, Schedules, Shape<int, int, int>, cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp, Element, cutlass::layout::RowMajor,
      cutlass::layout::StrassenLayout, Element, cutlass::layout::RowMajor,
      cutlass::layout::StrassenLayout, void, cutlass::layout::RowMajor,
      cutlass::layout::OriginalLayout, float, Cluster, Int<Stages>, PresumTile,
      PresumTile, cutlass::gemm::device::PresumOpt<>, 8, 8, 8, Element>;
  using Gemm = cutlass::gemm::device::StrassenGemmUniversalAdapter<Kernels>;
  using K0 = typename Gemm::GemmKernelM0;
  using K1 = typename Gemm::GemmKernelM1;
  using K2 = typename Gemm::GemmKernelM2;
  using K3 = typename Gemm::GemmKernelM3;
  using K4 = typename Gemm::GemmKernelM4;
  using K5 = typename Gemm::GemmKernelM5;
  using K6 = typename Gemm::GemmKernelM6;
  template <typename Group>
  using Parallel =
      MmaStrassen::ParallelMiKernels<Group, K0, K1, K2, K3, K4, K5, K6>;
  using P0 = Parallel<typename Schedules::ParallelGroups0>;
  using P1 = Parallel<typename Schedules::ParallelGroups1>;
  using P2 = Parallel<typename Schedules::ParallelGroups2>;

  static typename Gemm::Arguments arguments(int m, int n, int k,
                                            Element const* a, Element const* b,
                                            Element* d, int device, int sms,
                                            int swizzle, int raster) {
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    hw.sm_count = sms;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        {a,
         cutlass::make_cute_packed_stride(typename Gemm::StrideA{}, {m, k, 1}),
         b,
         cutlass::make_cute_packed_stride(typename Gemm::StrideB{}, {n, k, 1})},
        {{1.f, 0.f},
         nullptr,
         cutlass::make_cute_packed_stride(typename Gemm::StrideC{}, {m, n, 1}),
         d,
         cutlass::make_cute_packed_stride(typename Gemm::StrideD{}, {m, n, 1})},
        hw};
    args.scheduler.max_swizzle_size = swizzle;
    using Raster = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = raster == 0 ? Raster::AlongN : Raster::AlongM;
    return args;
  }

  static int configure() {
    int result = int(cudaFuncSetAttribute(
        cutlass::KernelParallelMiGroup<P0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, P0::SharedStorageSize()));
    if (result != 0) return result;
    result = int(cudaFuncSetAttribute(
        cutlass::KernelParallelMiGroup<P1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, P1::SharedStorageSize()));
    if (result != 0) return result;
    if constexpr (P2::HasAKernel()) {
      return int(
          cudaFuncSetAttribute(cutlass::KernelParallelMiGroup<P2>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               P2::SharedStorageSize()));
    }
    return 0;
  }

  static size_t workspace_size(int m, int n, int k) {
    auto args = arguments(m, n, k, nullptr, nullptr, nullptr, 0, 132, 1, 0);
    return Gemm::get_workspace_size(args) -
           Gemm::get_presum_b_workspace_size(args);
  }

  static int run(Element const* a, Element const* b, Element* presums_b,
                 Element* d, void* workspace, int m, int n, int k, int device,
                 int sms, int swizzle, int raster, cudaStream_t stream) {
    auto args = arguments(m, n, k, a, b, d, device, sms, swizzle, raster);
    auto status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) return status_code(status);
    auto presum_a = static_cast<Element*>(workspace);
    auto postsum =
        reinterpret_cast<Element*>(static_cast<char*>(workspace) +
                                   Gemm::get_presum_a_workspace_size(args));
    auto sem = reinterpret_cast<int*>(reinterpret_cast<char*>(postsum) +
                                      Gemm::get_postsum_m_workspace_size(args));
    size_t offset = 0;
#define MAKE_PARAMS(I)                                                 \
  status = K##I::initialize_workspace(                                 \
      args, reinterpret_cast<char*>(sem) + offset, stream);            \
  if (status != cutlass::Status::kSuccess) return status_code(status); \
  offset += K##I::get_workspace_size(args);                            \
  auto p##I =                                                          \
      K##I::to_underlying_arguments(args, presum_a, presums_b, postsum, sem);
    MAKE_PARAMS(0)
    MAKE_PARAMS(1)
    MAKE_PARAMS(2)
    MAKE_PARAMS(3)
    MAKE_PARAMS(4)
    MAKE_PARAMS(5)
    MAKE_PARAMS(6)
#undef MAKE_PARAMS
    status =
        Gemm::template run_parallel<P0>(p0, p1, p2, p3, p4, p5, p6, stream);
    if (status != cutlass::Status::kSuccess) return status_code(status);
    status =
        Gemm::template run_parallel<P1>(p0, p1, p2, p3, p4, p5, p6, stream);
    if (status != cutlass::Status::kSuccess) return status_code(status);
    if constexpr (P2::HasAKernel()) {
      // M4/M5 reduce into outputs initialized by M2/M3 on the same stream.
      status =
          Gemm::template run_parallel<P2>(p0, p1, p2, p3, p4, p5, p6, stream);
      if (status != cutlass::Status::kSuccess) return status_code(status);
    }
    return int(cudaGetLastError());
  }
};

using Cluster2x1 = StrassenKernel<Shape<_2, _1, _1>>;
using Cluster1x2 = StrassenKernel<Shape<_1, _2, _1>>;

using Mlp2Pingpong2x1 = StrassenKernel<Shape<_2, _1, _1>, true, 4>;
using Mlp2Pingpong1x2 = StrassenKernel<Shape<_1, _2, _1>, true, 4>;
using Mlp2Cooperative2x1 = StrassenKernel<Shape<_2, _1, _1>, false, 4>;
using Mlp2Cooperative1x2 = StrassenKernel<Shape<_1, _2, _1>, false, 4>;

struct KernelFunctions {
  decltype(&Cluster2x1::configure) configure;
  decltype(&Cluster2x1::workspace_size) workspace_size;
  decltype(&Cluster2x1::run) run;
};
template <typename Kernel>
constexpr KernelFunctions kernel_functions() {
  return {Kernel::configure, Kernel::workspace_size, Kernel::run};
}
constexpr KernelFunctions Mlp2Kernels[] = {
    kernel_functions<Mlp2Pingpong2x1>(), kernel_functions<Mlp2Pingpong1x2>(),
    kernel_functions<Mlp2Cooperative2x1>(),
    kernel_functions<Mlp2Cooperative1x2>()};
constexpr int Mlp2KernelCount = sizeof(Mlp2Kernels) / sizeof(Mlp2Kernels[0]);

__global__ void pack_activation(Element const* x, Element* packed, int m,
                                int k) {
  size_t quarter = size_t(m / 2) * (k / 2);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < size_t(m) * k; i += size_t(blockDim.x) * gridDim.x) {
    size_t row = i / k, col = i % k;
    size_t q = (row / (m / 2)) * 2 + col / (k / 2);
    packed[q * quarter + (row % (m / 2)) * (k / 2) + col % (k / 2)] = x[i];
  }
}

__global__ void prepare_weight(Element const* w, Element* packed,
                               Element* presums, int n, int padded_n, int k) {
  size_t quarter = size_t(k / 2) * (padded_n / 2);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < quarter;
       i += size_t(blockDim.x) * gridDim.x) {
    size_t row = i / (padded_n / 2), col = i % (padded_n / 2);
    float b0 = col < n / 2 ? float(w[col * k + row]) : 0.f;
    float b1 = col < n / 2 ? float(w[(col + n / 2) * k + row]) : 0.f;
    float b2 = col < n / 2 ? float(w[col * k + row + k / 2]) : 0.f;
    float b3 = col < n / 2 ? float(w[(col + n / 2) * k + row + k / 2]) : 0.f;
    packed[i] = Element(b0);
    packed[quarter + i] = Element(b1);
    packed[2 * quarter + i] = Element(b2);
    packed[3 * quarter + i] = Element(b3);
    float b31 = b3 - b1;
    float s3 = b31 + b0;
    presums[Presums0::indexBPresum(BPresums::B31) * quarter + i] = Element(b31);
    presums[Presums0::indexBPresum(BPresums::B10) * quarter + i] =
        Element(b1 - b0);
    presums[Presums0::indexBPresum(BPresums::S3) * quarter + i] = Element(s3);
    presums[Presums0::indexBPresum(BPresums::S3B2) * quarter + i] =
        Element(s3 - b2);
  }
}

}  // namespace vllm_strassen

using namespace vllm_strassen;

extern "C" int vllm_strassen_configure() {
  int result = Cluster2x1::configure();
  if (result != 0) return result;
  return Cluster1x2::configure();
}

extern "C" size_t vllm_strassen_workspace_size(int m, int n, int k) {
  return std::max(Cluster2x1::workspace_size(m, n, k),
                  Cluster1x2::workspace_size(m, n, k));
}

extern "C" int vllm_strassen_prepare(void const* weight, void* packed,
                                     void* presums, int n, int padded_n, int k,
                                     void* stream) {
  prepare_weight<<<4096, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<Element const*>(weight), static_cast<Element*>(packed),
      static_cast<Element*>(presums), n, padded_n, k);
  return int(cudaGetLastError());
}

extern "C" int vllm_strassen_run_padded_v3(
    void const* x, void* packed_a, void const* packed_b, void* presums_b,
    void* output, void* workspace, int m, int n, int k, int device, int sms,
    int swizzle, int raster, int skip_activation_packing, int cluster_m,
    void* stream_ptr) {
  if (m <= 0 || m % 512 != 0 || (cluster_m != 1 && cluster_m != 2) ||
      (cluster_m == 2 && m % 1024 != 0)) {
    return status_code(cutlass::Status::kErrorInvalidProblem);
  }

  auto stream = static_cast<cudaStream_t>(stream_ptr);
  auto a = skip_activation_packing ? static_cast<Element const*>(x)
                                   : static_cast<Element const*>(packed_a);
  auto d = static_cast<Element*>(output);
  if (!skip_activation_packing) {
    pack_activation<<<4096, 256, 0, stream>>>(
        static_cast<Element const*>(x), static_cast<Element*>(packed_a), m, k);
  }
  auto b = static_cast<Element const*>(packed_b);
  auto sums = static_cast<Element*>(presums_b);
  if (cluster_m == 1) {
    return Cluster1x2::run(a, b, sums, d, workspace, m, n, k, device, sms,
                           swizzle, raster, stream);
  }
  return Cluster2x1::run(a, b, sums, d, workspace, m, n, k, device, sms,
                         swizzle, raster, stream);
}

extern "C" int vllm_strassen_mlp2_configure() {
  for (auto const& kernel : Mlp2Kernels) {
    int status = kernel.configure();
    if (status != 0) return status;
  }
  return 0;
}

extern "C" size_t vllm_strassen_mlp2_workspace_size(int m, int n, int k) {
  size_t size = 0;
  for (auto const& kernel : Mlp2Kernels) {
    size = std::max(size, kernel.workspace_size(m, n, k));
  }
  return size;
}

extern "C" int vllm_strassen_mlp2_run_v2(void const* x, void* packed_a,
                                         void const* packed_b, void* presums_b,
                                         void* output, void* workspace, int m,
                                         int n, int k, int device, int sms,
                                         int swizzle, int raster,
                                         int skip_activation_packing,
                                         int kernel_id, void* stream_ptr) {
  if (m <= 0 || m % 1024 != 0 || n != 8192 || k != 29696 || kernel_id < 0 ||
      kernel_id >= Mlp2KernelCount ||
      (swizzle != 1 && swizzle != 2 && swizzle != 4) ||
      (raster != 0 && raster != 1)) {
    return status_code(cutlass::Status::kErrorInvalidProblem);
  }
  auto stream = static_cast<cudaStream_t>(stream_ptr);
  auto a = skip_activation_packing ? static_cast<Element const*>(x)
                                   : static_cast<Element const*>(packed_a);
  if (!skip_activation_packing) {
    pack_activation<<<4096, 256, 0, stream>>>(
        static_cast<Element const*>(x), static_cast<Element*>(packed_a), m, k);
  }
  return Mlp2Kernels[kernel_id].run(a, static_cast<Element const*>(packed_b),
                                    static_cast<Element*>(presums_b),
                                    static_cast<Element*>(output), workspace, m,
                                    n, k, device, sms, swizzle, raster, stream);
}
