#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

template <typename MT>
void calc_trajectory (IslMpi<MT>& cm, const Int nets, const Int nete,
                      const Int step, const Real* v01_r, const Real* v1gradv0_r,
                      Real* dep_points_r)
{
  using slmm::Timer;

  slmm_assert(cm.np == 4);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

#ifdef COMPOSE_PORT
  auto& dep_points = cm.tracer_arrays->dep_points;
#else
  DepPointsH<MT> dep_points(dep_points_r, cm.nelemd, cm.nlev, cm.np2);
#endif

  if (step == 0) {
    
    return;
  }

}

template void calc_trajectory(IslMpi<ko::MachineTraits>&, const Int, const Int,
                              const Int, const Real*, const Real*, Real*);

} // namespace islmpi
} // namespace homme
