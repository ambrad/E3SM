#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

// dep_points is const in principle, but if lev <=
// semi_lagrange_nearest_point_lev, a departure point may be altered if the
// winds take it outside of the comm halo.
template <typename MT>
void calc_trajectory (
  IslMpi<MT>& cm, const Int nets, const Int nete,
  Real* dep_points_r)
{
  using slmm::Timer;

  slmm_assert(cm.np == 4);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

#ifdef COMPOSE_PORT
  const auto& dep_points = cm.tracer_arrays->dep_points;
#else
  const DepPointsH<MT> dep_points(dep_points_r, cm.nelemd, cm.nlev, cm.np2);
#endif

}

template void calc_trajectory(IslMpi<ko::MachineTraits>&, const Int, const Int, Real*);

} // namespace islmpi
} // namespace homme
