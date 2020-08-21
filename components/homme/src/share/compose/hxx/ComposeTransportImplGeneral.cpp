/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"

namespace Homme {

void ComposeTransportImpl::reset (const SimulationParams& params) {
  const auto num_elems = Context::singleton().get<Connectivity>().get_num_local_elements();
  if (m_data.nelemd == num_elems && m_data.qsize == params.qsize) return;

  m_data.qsize = params.qsize;
  m_data.nelemd = num_elems;
  slmm_throw_if(m_data.qsize == 0,
                "SL transport requires qsize > 0; if qsize == 0, use Eulerian.");

  m_data.dep_pts = DeparturePoints("dep_pts", m_data.nelemd);

  m_tp_ne = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd);
  m_tp_ne_qsize = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd * m_data.qsize);
  m_tu_ne = TeamUtils<ExecSpace>(m_tu_ne);
  m_tu_ne_qsize = TeamUtils<ExecSpace>(m_tu_ne_qsize);

  m_sphere_ops.allocate_buffers(m_tu_ne_qsize);

  m_data.nslot = std::min(m_data.nelemd, m_tu_ne.get_num_ws_slots());

  if (Context::singleton().get<Connectivity>().get_comm().root())
    printf("nelemd %d qsize %d hv_q %d np1_qdp %d independent_time_steps %d\n",
           m_data.nelemd, m_data.qsize, m_data.hv_q, m_data.np1_qdp,
           (int) m_data.independent_time_steps);
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

void ComposeTransportImpl::run (const Real dt) {
  
}

} // namespace Homme
