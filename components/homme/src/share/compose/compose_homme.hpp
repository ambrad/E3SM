#ifndef INCLUDE_COMPOSE_HOMME_HPP
#define INCLUDE_COMPOSE_HOMME_HPP

#include "compose.hpp"
#include "compose_slmm.hpp"

namespace homme {
typedef int Int;
typedef double Real;

// Fortran array wrappers with Fortran index order.
template <typename T> using FA2 =
  Kokkos::View<T**,    Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA3 =
  Kokkos::View<T***,   Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA4 =
  Kokkos::View<T****,  Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA5 =
  Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Fortran array wrappers with C index order.
template <typename T> using CA2 =
  Kokkos::View<T**,    Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T> using CA3 =
  Kokkos::View<T***,   Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T> using CA4 =
  Kokkos::View<T****,  Kokkos::LayoutRight, Kokkos::HostSpace>;
template <typename T> using CA5 =
  Kokkos::View<T*****, Kokkos::LayoutRight, Kokkos::HostSpace>;

struct Cartesian3D { Real x, y, z; };

} // namespace homme

#endif
