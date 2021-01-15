#ifndef INCLUDE_COMPOSE_KOKKOS_HPP
#define INCLUDE_COMPOSE_KOKKOS_HPP

#include "compose.hpp"

#include <limits>

namespace Kokkos {

template <typename View>
using Const = typename View::const_type;

// GPU-friendly replacements for std::*.
template <typename T> KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
void swap (T& a, T& b) { const auto tmp = a; a = b; b = tmp; }

template <typename Real> struct NumericTraits;

template <> struct NumericTraits<double> {
  KOKKOS_INLINE_FUNCTION static double epsilon () {
    return
#ifdef KOKKOS_ENABLE_CUDA
      2.2204460492503131e-16
#else
      std::numeric_limits<double>::epsilon()
#endif
      ;
  }
};

template <> struct NumericTraits<float> {
  KOKKOS_INLINE_FUNCTION static float epsilon () {
    return
#ifdef KOKKOS_ENABLE_CUDA
      1.1920928955078125e-07
#else
      std::numeric_limits<float>::epsilon()
#endif
      ;
  }
};

template <typename ExeSpace>
struct DeviceType {
  typedef Kokkos::Device<typename ExeSpace::execution_space,
                         typename ExeSpace::memory_space> type;
};

#ifdef KOKKOS_HAVE_CUDA
typedef Kokkos::Device<Kokkos::CudaSpace::execution_space,
                       Kokkos::CudaSpace::memory_space> DefaultDeviceType;

template <> struct DeviceType<Kokkos::Cuda> {
  typedef DefaultDeviceType type;
};
#else
typedef Kokkos::Device<Kokkos::DefaultExecutionSpace::execution_space,
                       Kokkos::DefaultExecutionSpace::memory_space> DefaultDeviceType;
#endif

struct MachineTraits {
  // Host and device execution spaces.
#ifdef COMPOSE_PORT
  using HES = Kokkos::DefaultHostExecutionSpace;
  using DES = Kokkos::DefaultExecutionSpace;
#else
  using HES = Kokkos::Serial;
  using DES = Kokkos::Serial;
#endif
  using HDT = DeviceType<HES>::type;
  using DDT = DeviceType<DES>::type;
};

template <typename ES> struct OnGpu {
  enum : bool { value =
#ifdef COMPOSE_MIMIC_GPU
                true
#else
                false
#endif
  };
};
#ifdef KOKKOS_ENABLE_CUDA
template <> struct OnGpu<Kokkos::Cuda> { enum : bool { value = true }; };
template <> struct OnGpu<MachineTraits> {}; // flag as an error at compile time
#endif

template <typename MT> using EnableIfOnGpu
  = typename std::enable_if<Kokkos::OnGpu<typename MT::DES>::value>::type;
template <typename MT> using EnableIfNotOnGpu
  = typename std::enable_if< ! Kokkos::OnGpu<typename MT::DES>::value>::type;

template <typename MT> struct SameSpace {
  enum { value = std::is_same<typename MT::HES, typename MT::DES>::value };
};
template <typename MT> using EnableIfSameSpace
  = typename std::enable_if<SameSpace<MT>::value>::type;
template <typename MT> using EnableIfDiffSpace
  = typename std::enable_if< ! SameSpace<MT>::value>::type;

} // namespace Kokkos

#endif
