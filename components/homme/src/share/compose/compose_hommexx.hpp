#ifndef INCLUDE_COMPOSE_HOMMEXX_HPP
#define INCLUDE_COMPOSE_HOMMEXX_HPP

#include <Kokkos_Core.hpp>

namespace homme {
namespace compose {

template <typename DataType>
using SetView = Kokkos::View<DataType, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

void set_views(const SetView<double***>& spheremp,
               const SetView<double****>& dp, const SetView<double*****>& dp3d,
               const SetView<double******>& qdp, const SetView<double*****>& q,
               const SetView<double*****>& dep_points);

void advect(const int np1, const int n0_qdp, const int np1_qdp);

bool property_preserve(const int limiter_option);
void property_preserve_check();

void finalize();

} // namespace compose
} // namespace homme

#endif
