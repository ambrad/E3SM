#ifndef INCLUDE_COMPOSE_KOKKOS_HPP
#define INCLUDE_COMPOSE_KOKKOS_HPP

#include <Kokkos_Core.hpp>

#include <limits>

namespace Kokkos {

// GPU-friendly replacements for std::*.
template <typename T> KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }

template <typename Real> struct NumericTraits;

template <> struct NumericTraits<double> {
  static double epsilon () {
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
  static float epsilon () {
    return
#ifdef KOKKOS_ENABLE_CUDA
      1.1920928955078125e-07
#else
      std::numeric_limits<float>::epsilon()
#endif
      ;
  }
};

} // namespace Kokkos

#endif
