#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

template <typename T> using CA4 = ko::View<T****,  ko::LayoutRight, ko::HostSpace>;
template <typename T> using CA5 = ko::View<T*****, ko::LayoutRight, ko::HostSpace>;

template <typename MT>
void calc_trajectory (IslMpi<MT>& cm, const Int nets, const Int nete,
                      const Int step, const Real dtsub, Real* v01_r,
                      const Real* v1gradv0_r, Real* dep_points_r)
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

  CA5<Real> v01(v01_r, cm.nelemd, 2, 2, cm.np2, cm.nlev);
  CA4<const Real> v1gradv0(v1gradv0_r, cm.nelemd, 2, cm.np2, cm.nlev);

  if (step == 0) {
    for (int ie = nets; ie <= nete; ++ie)
      for (int d = 0; d < 2; ++d)
        for (int k = 0; k < cm.np2; ++k)
          for (int lev = 0; lev < cm.nlev; ++lev)
            v01(ie,0,d,k,lev) = ((v01(ie,0,d,k,lev) + v01(ie,1,d,k,lev))/2 +
                                 dtsub*v1gradv0(ie,d,k,lev));
    return;
  }

}

template void calc_trajectory(IslMpi<ko::MachineTraits>&, const Int, const Int,
                              const Int, const Real, Real*, const Real*, Real*);

} // namespace islmpi
} // namespace homme
