/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "GllFvRemapImpl.hpp"

namespace Homme {

static int calc_nslot (const int nelemd) {
  const auto tp = Homme::get_default_team_policy<ExecSpace>(nelemd);
  const auto tu = TeamUtils<ExecSpace>(tp);
  return std::min(nelemd, tu.get_num_ws_slots());
}

GllFvRemapImpl::GllFvRemapImpl ()
  : m_hvcoord(Context::singleton().get<HybridVCoord>()),
    m_elements(Context::singleton().get<Elements>()),
    m_state(m_elements.m_state),
    m_derived(m_elements.m_derived),
    m_geometry(Context::singleton().get<ElementsGeometry>()),
    m_tracers(Context::singleton().get<Tracers>()),
    m_sphere_ops(Context::singleton().get<SphereOperators>()),
    m_tp_ne(1,1,1), m_tu_ne(m_tp_ne), // throwaway settings
    m_tp_ne_qsize(1,1,1), m_tu_ne_qsize(m_tp_ne_qsize) // throwaway settings
{
  nslot = calc_nslot(m_geometry.num_elems());
}

void GllFvRemapImpl::reset (const SimulationParams& params) {
  const auto num_elems = Context::singleton().get<Connectivity>().get_num_local_elements();

  m_data.qsize = params.qsize;
  Errors::runtime_check(m_data.qsize > 0, "GllFvRemapImpl requires qsize > 0");
  m_data.nelemd = num_elems;

  m_tp_ne = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd);
  m_tp_ne_qsize = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd * m_data.qsize);
  m_tu_ne = TeamUtils<ExecSpace>(m_tp_ne);
  m_tu_ne_qsize = TeamUtils<ExecSpace>(m_tp_ne_qsize);

  m_sphere_ops.allocate_buffers(m_tu_ne_qsize);

  if (Context::singleton().get<Connectivity>().get_comm().root())
    printf("gfr> nelemd %d qsize %d\n",
           m_data.nelemd, m_data.qsize);
}

int GllFvRemapImpl::requested_buffer_size () const {
  // FunctorsBuffersManager wants the size in terms of sizeof(Real).
  return (3*Buf1::shmem_size(nslot) +
          2*Buf2::shmem_size(nslot))/sizeof(Real);
}

void GllFvRemapImpl::init_buffers (const FunctorsBuffersManager& fbm) {
  Scalar* mem = reinterpret_cast<Scalar*>(fbm.get_memory());
  for (int i = 0; i < 3; ++i) {
    m_data.buf1[i] = Buf1(mem, nslot);
    mem += Buf1::shmem_size(nslot)/sizeof(Scalar);
  }
  for (int i = 0; i < 2; ++i) {
    m_data.buf2[i] = Buf2(mem, nslot);
    mem += Buf2::shmem_size(nslot)/sizeof(Scalar);
  }
}

void GllFvRemapImpl::init_boundary_exchanges () {
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
}

void GllFvRemapImpl
::run_dyn_to_fv (const int time_idx, const Phys0T& ps, const Phys0T& phis, const Phys1T& Ts,
                 const Phys1T& omegas, const Phys2T& uvs, const Phys2T& qs) {
  const auto nf2 = m_data.nf2;
  const auto nelemd = m_data.nelemd;
  const auto qsize = m_data.qsize;
  const auto ncol = nelemd*nf2;

  assert(ps.extent_int(0) >= ncol);
  assert(phis.extent_int(0) >= ncol);
  assert(Ts.extent_int(0) >= ncol && Ts.extent_int(1) % packn == 0);
  assert(omegas.extent_int(0) >= ncol && omegas.extent_int(1) % packn == 0);
  assert(uvs.extent_int(0) >= ncol && uvs.extent_int(1) == 2 && uvs.extent_int(2) % packn == 0);
  assert(qs.extent_int(0) >= ncol && qs.extent_int(1) >= qsize && qs.extent_int(2) % packn == 0);

  VPhys1T
    T(real2pack(Ts), Ts.extent_int(0), Ts.extent_int(1)/packn),
    omega(real2pack(omegas), omegas.extent_int(0), omegas.extent_int(1)/packn);
  VPhys2T
    uv(real2pack(uvs), uvs.extent_int(0), 2, uvs.extent_int(2)/packn),
    q(real2pack(qs), qs.extent_int(0), qs.extent_int(1), qs.extent_int(2)/packn);

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    
  };
  Kokkos::parallel_for(m_tp_ne, fe);

  const auto feq = KOKKOS_LAMBDA (const MT& team) {
    
  };
  Kokkos::parallel_for(m_tp_ne_qsize, feq);
}

void GllFvRemapImpl::
run_fv_to_dyn (const int time_idx, const Real dt, const CPhys1T& Ts, const CPhys2T& uvs,
               const CPhys2T& qs) {
  const auto nf2 = m_data.nf2;
  const auto nelemd = m_data.nelemd;
  const auto qsize = m_data.qsize;
  const auto ncol = nelemd*nf2;

  assert(Ts.extent_int(0) >= ncol && Ts.extent_int(1) % packn == 0);
  assert(uvs.extent_int(0) >= ncol && uvs.extent_int(1) == 2 && uvs.extent_int(2) % packn == 0);
  assert(qs.extent_int(0) >= ncol && qs.extent_int(1) >= qsize && qs.extent_int(2) % packn == 0);

  CVPhys1T
    T(creal2pack(Ts), Ts.extent_int(0), Ts.extent_int(1)/packn);
  CVPhys2T
    uv(creal2pack(uvs), uvs.extent_int(0), 2, uvs.extent_int(2)/packn),
    q(creal2pack(qs), qs.extent_int(0), qs.extent_int(1), qs.extent_int(2)/packn);

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    
  };
  Kokkos::parallel_for(m_tp_ne, fe);

  const auto feq = KOKKOS_LAMBDA (const MT& team) {
    
  };
  Kokkos::parallel_for(m_tp_ne_qsize, feq);

  // halo exchange

  const auto geq = KOKKOS_LAMBDA (const MT& team) {
    
  };
  Kokkos::parallel_for(m_tp_ne_qsize, geq);
}

} // namespace Homme
