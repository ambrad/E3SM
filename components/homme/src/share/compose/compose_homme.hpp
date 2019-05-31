#ifndef INCLUDE_COMPOSE_HOMME_HPP
#define INCLUDE_COMPOSE_HOMME_HPP

#include <Kokkos_Core.hpp>

namespace homme {
typedef int Int;
typedef double Real;

// Fortran array wrappers.
template <typename T> using FA2 =
  Kokkos::View<T**,    Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA3 =
  Kokkos::View<T***,   Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA4 =
  Kokkos::View<T****,  Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA5 =
  Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

struct Cartesian3D { Real x, y, z; };

// GLL-level indexing abstraction so we can work with both Fortran and Kokkos
// dycores. Kokkos::LayoutStride technically would solve this issue, but it has
// a runtime overhead. Kokkos::LayoutIndex, a compile-time permutation of
// indices, would solve the problem, but I think that feature is not going to be
// available in the near term.

template <typename View, typename Int>
typename View::value_type&
idx (const View& v, const Int& i, const Int& j,
     typename std::enable_if<View::rank == 2>::type* = 0)
{ return v(i,j); }
template <typename View, typename Int>
typename View::value_type&
idx (const View& v, const Int& i, const Int& j, const Int& lev,
     typename std::enable_if<View::rank == 3>::type* = 0)
{ return v(i,j,lev); }
template <typename View, typename Int>
typename View::value_type&
idx (const View& v, const Int& i, const Int& j, const Int& lev, const Int& q,
     typename std::enable_if<View::rank == 4>::type* = 0)
{ return v(i,j,lev,q); }
template <typename View, typename Int>
typename View::value_type&
idx (const View& v, const Int& i, const Int& j, const Int& lev, const Int& q, const Int& time,
     typename std::enable_if<View::rank == 5>::type* = 0)
{ return v(i,j,lev,q,time); }

} // namespace homme

#endif
