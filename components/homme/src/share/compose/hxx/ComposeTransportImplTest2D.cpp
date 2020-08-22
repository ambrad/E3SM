/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "compose_test.hpp"
#include "profiling.hpp"

namespace Homme {

static void fill_ics (const ComposeTransportImpl& cti, const int np1, const int n0_qdp) {
  const auto qdp = Kokkos::create_mirror_view(cti.m_tracers.qdp);
  const auto dp3d = Kokkos::create_mirror_view(cti.m_state.m_dp3d);
  const auto pll = Kokkos::create_mirror_view(cti.m_elements.m_geometry.m_sphere_latlon);
  const auto f = [&] (int ie, int lev, int i, int j) {
    auto lat = pll(ie,i,j,0), lon = pll(ie,i,j,1);
    compose::test::offset_latlon(cti.num_phys_lev, lev, lat, lon);
    for (int q = 0; q < cti.m_data.qsize; ++q)
      compose::test::InitialCondition::init(compose::test::get_ic(cti.m_data.qsize, lev, q),
                                            1, &lat, &lon,
                                            &qdp(ie,n0_qdp,q,i,j,lev/cti.packn)[lev%cti.packn]);
    dp3d(ie,np1,i,j,lev/cti.packn)[lev%cti.packn] = 1;
  };
  cti.loop_host_ie_plev_ij(f);
  Kokkos::deep_copy(cti.m_tracers.qdp, qdp);
  Kokkos::deep_copy(cti.m_state.m_dp3d, dp3d);
}

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

//todo Rewrite to return values. For now, just q&d.
static void finish (const ComposeTransportImpl& cti) {
  
}

void ComposeTransportImpl::test_2d (const int nstep) {
  SimulationParams& params = Context::singleton().get<SimulationParams>();
  params.qsplit = 1;

  TimeLevel& tl = Context::singleton().get<TimeLevel>();
  tl.nstep = 0;
  tl.update_tracers_levels(params.qsplit);
  const Real twelve_days = 3600 * 24 * 12, dt = twelve_days/nstep;

  fill_ics(*this, tl.np1, tl.n0_qdp);

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

  finish(*this);
}

} // namespace Homme
