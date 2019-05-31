#ifndef INCLUDE_COMPOSE_SLMM_HOMME_HPP
#define INCLUDE_COMPOSE_SLMM_HOMME_HPP

#include "compose_slmm.hpp"

namespace homme {

// Fortran array wrappers.
template <typename T> using FA2 =
  Kokkos::View<T**,    Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA3 =
  Kokkos::View<T***,   Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA4 =
  Kokkos::View<T****,  Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA5 =
  Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

struct Cartesian3D { slmm::Real x, y, z; };

} // namespace homme

#endif
