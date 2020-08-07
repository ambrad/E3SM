/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"

namespace Homme {

void ComposeTransportImpl::reset (const SimulationParams& params) {
  m_data.nelemd = Context::singleton().get<Connectivity>().get_num_local_elements();
  m_data.qsize = m_tracers.num_tracers();
  slmm_throw_if(m_data.qsize == 0,
                "SL transport requires qsize > 0; if qsize == 0, use Eulerian.");
  if (OnGpu<ExecSpace>::value) {
    ThreadPreferences tp;
    tp.max_threads_usable = NUM_PHYSICAL_LEV;
    tp.max_vectors_usable = NP*NP;
    tp.prefer_threads = false;
    tp.prefer_larger_team = true;
    const auto p = DefaultThreadsDistribution<ExecSpace>
      ::team_num_threads_vectors(m_data.nelemd, tp);
    const auto
      nhwthr = p.first*p.second,
      nvec = std::min(NP*NP, nhwthr),
      nthr = nhwthr/nvec;
    m_policy = TeamPolicy(m_data.nelemd, nthr, nvec);
  } else {
    ThreadPreferences tp;
    tp.max_threads_usable = NUM_PHYSICAL_LEV;
    tp.max_vectors_usable = 1;
    tp.prefer_threads = true;
    const auto p = DefaultThreadsDistribution<ExecSpace>
      ::team_num_threads_vectors(m_data.nelemd, tp);
    m_policy = TeamPolicy(m_data.nelemd, p.first, 1);
  }
  m_tu = TeamUtils<ExecSpace>(m_policy);
  nslot = std::min(m_data.nelemd, m_tu.get_num_ws_slots());
  do {
    if ( ! Context::singleton().get<Connectivity>().get_comm().root()) break;
    printf("nelemd %d qsize %d hv_q %d np1_qdp %d independent_time_steps %d\n",
           m_data.nelemd, m_data.qsize, m_data.hv_q, m_data.np1_qdp,
           (int) m_data.independent_time_steps);
  } while (0);
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

  // For trajectory computation.
  m_v_dss_be = std::make_shared<BoundaryExchange>();
  auto be = m_v_dss_be;
  be->set_buffers_manager(bm_exchange);
  be->set_num_fields(0, 0, 2 + (m_data.independent_time_steps ? 1 : 0));
  be->register_field(m_derived.m_vstar, 2, 0);
  if (m_data.independent_time_steps) be->register_field(m_derived.m_divdp);
  be->registration_completed();

  // For optional HV applied to q.
  if (m_data.hv_q > 0) {
    m_Q_dss_be = std::make_shared<BoundaryExchange>();
    be = m_Q_dss_be;
    be->set_buffers_manager(bm_exchange);
    be->set_num_fields(0, 0, m_data.hv_q);
    be->register_field(m_tracers.Q, m_data.hv_q, 0);
    be->registration_completed();
  }
}

void ComposeTransportImpl::run () {
  Kokkos::fence();
}

} // namespace Homme
