/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "compose_test.hpp"
#include "profiling.hpp"

namespace Homme {

static void cp_v_to_vstar (const ComposeTransportImpl& cti, const int np1) {
  const auto vstar = cti.m_derived.m_vstar;
  const auto v = cti.m_state.m_v;
  const auto f = [&] (int ie, int lev, int i, int j) {
    for (int d = 0; d < 2; ++d)
      vstar(ie,d,i,j,lev) = v(ie,np1,d,i,j,lev);
  };
  cti.loop_device_ie_packlev_ij(f);
}

static void fill_v (const ComposeTransportImpl& cti, const Real t, const int np1) {
  const auto pll = cti.m_elements.m_geometry.m_sphere_latlon;
  const auto v = cti.m_state.m_v;
  constexpr auto packn = cti.packn;
  const compose::test::NonDivergentWindField wf;
  const auto f = [&] (int ie, int lev, int i, int j) {
    Real latlon[] = {pll(ie,i,j,0), pll(ie,i,j,1)};
    compose::test::offset_latlon(cti.num_phys_lev, lev, latlon[0], latlon[1]);
    Real uv[2];
    wf.eval(t, latlon, uv);
    for (int d = 0; d < 2; ++d)
      v(ie,np1,d,i,j,lev/packn)[lev%packn] = uv[d];
  };
  cti.loop_device_ie_physlev_ij(f);
}

void ComposeTransportImpl::test_2d (const int nstep) {
  SimulationParams& params = Context::singleton().get<SimulationParams>();
  assert(params.params_set);

  TimeLevel& tl = Context::singleton().get<TimeLevel>();
  tl.nstep = 0;
  tl.update_tracers_levels(params.qsplit);
  const Real twelve_days = 3600 * 24 * 12, dt = twelve_days/nstep;

  GPTLstart("compose_stt_step");
  for (int i = 0; i < nstep; ++i) {
    const auto tprev = dt*i;
    const auto t = dt*i;
    cp_v_to_vstar(*this, tl.np1);
    Kokkos::fence();
    fill_v(*this, t, tl.np1);
    Kokkos::fence();
    tl.nstep += params.qsplit;
    run(tl, dt);
    Kokkos::fence();
  }
  GPTLstop("compose_stt_step");
}

} // namespace Homme
