/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "compose_hommexx.hpp"

extern "C" void sl_get_params(int* hv_q, int* hv_subcycle_q);

namespace Homme {

void ComposeTransportImpl::reset (const SimulationParams& params) {
  const auto num_elems = Context::singleton().get<Connectivity>().get_num_local_elements();
  if (m_data.nelemd == num_elems && m_data.qsize == params.qsize) return;

  m_data.qsize = params.qsize;
  Errors::runtime_check(m_data.qsize > 0,
                        "SL transport requires qsize > 0; if qsize == 0, use Eulerian.");
  m_data.nelemd = num_elems;
  m_data.limiter_option = params.limiter_option;
  sl_get_params(&m_data.hv_q, &m_data.hv_subcycle_q);
  Errors::runtime_check(m_data.hv_q >= 0 && m_data.hv_q <= m_data.qsize,
                        "semi_lagrange_hv_q should be in [0, qsize].");
  Errors::runtime_check(m_data.hv_subcycle_q >= 0,
                        "hypervis_subcycle_q should be >= 0.");
  m_data.hv_scaling = params.hypervis_scaling;
  m_data.nu_q = params.nu_q;

  m_data.dep_pts = DeparturePoints("dep_pts", m_data.nelemd);

  m_tp_ne = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd);
  m_tp_ne_qsize = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd * m_data.qsize);
  m_tu_ne = TeamUtils<ExecSpace>(m_tp_ne);
  m_tu_ne_qsize = TeamUtils<ExecSpace>(m_tp_ne_qsize);

  m_sphere_ops.allocate_buffers(m_tu_ne_qsize);

  m_data.nslot = std::min(m_data.nelemd, m_tu_ne.get_num_ws_slots());

  if (Context::singleton().get<Connectivity>().get_comm().root())
    printf("nelemd %d qsize %d hv_q %d np1_qdp %d independent_time_steps %d\n",
           m_data.nelemd, m_data.qsize, m_data.hv_q, m_data.np1_qdp,
           (int) m_data.independent_time_steps);

  {
    const auto& g = m_elements.m_geometry;
    const auto& t = m_tracers;
    const auto& s = m_state;
    const auto& d = m_derived;
    const auto nel = m_data.nelemd;
    const auto nlev = NUM_LEV*packn;
    homme::compose::set_views(
      g.m_spheremp,
      homme::compose::SetView<Real****>  (reinterpret_cast<Real*>(d.m_dp.data()),
                                          nel, np, np, nlev),
      homme::compose::SetView<Real*****> (reinterpret_cast<Real*>(s.m_dp3d.data()),
                                          nel, NUM_TIME_LEVELS, np, np, nlev),
      homme::compose::SetView<Real******>(reinterpret_cast<Real*>(t.qdp.data()),
                                          nel, Q_NUM_TIME_LEVELS, QSIZE_D, np, np, nlev),
      homme::compose::SetView<Real*****> (reinterpret_cast<Real*>(t.Q.data()),
                                          nel, QSIZE_D, np, np, nlev),
      m_data.dep_pts);
  }
}

int ComposeTransportImpl::requested_buffer_size () const {
  // FunctorsBuffersManager wants the size in terms of sizeof(Real).
  return Buf1::shmem_size(m_data.nslot) + 2*Buf2::shmem_size(m_data.nslot);
}

void ComposeTransportImpl::init_buffers (const FunctorsBuffersManager& fbm) {
  Scalar* mem = reinterpret_cast<Scalar*>(fbm.get_memory());
  m_data.buf1 = Buf1(mem, m_data.nslot);
  mem += Buf1::shmem_size(m_data.nslot)/sizeof(Scalar);
  for (int i = 0; i < 2; ++i) {
    m_data.buf2[i] = Buf2(mem, m_data.nslot);
    mem += Buf2::shmem_size(m_data.nslot)/sizeof(Scalar);
  }
}

void ComposeTransportImpl::init_boundary_exchanges () {
  assert(m_data.qsize > 0); // after reset() called

  auto bm_exchange = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE];

  // For qdp DSS at end of transport step.
  for (int i = 0; i < Q_NUM_TIME_LEVELS; ++i) {
    m_qdp_dss_be[i] = std::make_shared<BoundaryExchange>();
    auto be = m_qdp_dss_be[i];
    be->set_buffers_manager(bm_exchange);
    be->set_num_fields(0, 0, m_data.qsize + 1);
    be->register_field(m_tracers.qdp, i, m_data.qsize, 0);
    be->register_field(m_derived.m_omega_p);
    be->registration_completed();
  }

  for (int i = 0; i < 2; ++i) {
    m_v_dss_be[i] = std::make_shared<BoundaryExchange>();
    auto be = m_v_dss_be[i];
    be->set_buffers_manager(bm_exchange);
    be->set_num_fields(0, 0, 2 + (i ? 1 : 0));
    be->register_field(m_derived.m_vstar, 2, 0);
    if (i) be->register_field(m_derived.m_divdp);
    be->registration_completed();
  }

  // For optional HV applied to q.
  if (m_data.hv_q > 0) {
    m_Q_dss_be = std::make_shared<BoundaryExchange>();
    auto be = m_Q_dss_be;
    be->set_buffers_manager(bm_exchange);
    be->set_num_fields(0, 0, m_data.hv_q);
    be->register_field(m_tracers.Q, m_data.hv_q, 0);
    be->registration_completed();
  }
}

void ComposeTransportImpl::run (const TimeLevel& tl, const Real dt) {
  m_data.np1 = tl.np1;
  m_data.np1_qdp = tl.np1_qdp;
  calc_trajectory(dt);
  homme::compose::advect(tl.np1, tl.n0_qdp, tl.np1_qdp);
  //todo optional hypervis
  const auto np1 = m_data.np1;
  const auto np1_qdp = m_data.np1_qdp;
  const auto qsize = m_data.qsize;
  if ( ! homme::compose::property_preserve(m_data.limiter_option)) {
    // For analysis purposes, property preservation was not run. Need to convert
    // Q to qdp.
    const auto qdp = m_tracers.qdp;
    const auto Q = m_tracers.Q;
    const auto dp3d = m_state.m_dp3d;
    const auto spheremp = m_elements.m_geometry.m_spheremp;
    const auto f = KOKKOS_LAMBDA (const int idx) {
      int ie, q, i, j, lev;
      idx_ie_q_ij_nlev<num_lev_pack>(qsize, idx, ie, q, i, j, lev);
      qdp(ie,np1_qdp,q,i,j,lev) = Q(ie,q,i,j,lev)/dp3d(ie,np1,i,j,lev);
    };
    launch_ie_q_ij_nlev<num_lev_pack>(f);
  }
  { // DSS qdp and omega
    const auto qdp = m_tracers.qdp;
    const auto spheremp = m_elements.m_geometry.m_spheremp;
    const auto f1 = KOKKOS_LAMBDA (const int idx) {
      int ie, q, i, j, lev;
      idx_ie_q_ij_nlev<num_lev_pack>(qsize, idx, ie, q, i, j, lev);
      qdp(ie,np1_qdp,q,i,j,lev) *= spheremp(ie,i,j);
    };
    launch_ie_q_ij_nlev<num_lev_pack>(f1);
    const auto omega = m_derived.m_omega_p;
    const auto f2 = KOKKOS_LAMBDA (const int idx) {
      int ie, i, j, lev;
      idx_ie_ij_nlev<num_lev_pack>(idx, ie, i, j, lev);
      omega(ie,i,j,lev) *= spheremp(ie,i,j);
    };
    launch_ie_ij_nlev<num_lev_pack>(f2);
    m_qdp_dss_be[tl.np1_qdp]->exchange(m_elements.m_geometry.m_rspheremp);
  }
  //todo semi_lagrange_cdr_check
}

} // namespace Homme
