#ifndef INCLUDE_COMPOSE_HOMMEXX_HPP
#define INCLUDE_COMPOSE_HOMMEXX_HPP

#include <Kokkos_Core.hpp>

namespace homme {
namespace compose {

void set_views(const Kokkos::View<double***>& spheremp,
               const Kokkos::View<double****>& dp, const Kokkos::View<double*****>& dp3d,
               const Kokkos::View<double******>& qdp, const Kokkos::View<double*****>& q,
               const Kokkos::View<double*****>& dep_points);

void advect(const int np1, const int n0_qdp, const int np1_qdp);

bool property_preserve(const int limiter_option);

void finalize();

} // namespace compose
} // namespace homme

#endif
