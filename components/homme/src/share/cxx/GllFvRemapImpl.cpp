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
  return (Data::nbuf1*Buf1::shmem_size(nslot) +
          Data::nbuf2*Buf2::shmem_size(nslot))/sizeof(Real);
}

void GllFvRemapImpl::init_buffers (const FunctorsBuffersManager& fbm) {
  Scalar* mem = reinterpret_cast<Scalar*>(fbm.get_memory());
  for (int i = 0; i < Data::nbuf1; ++i) {
    m_data.buf1[i] = Buf1(mem, nslot);
    mem += Buf1::shmem_size(nslot)/sizeof(Scalar);
  }
  for (int i = 0; i < Data::nbuf2; ++i) {
    m_data.buf2[i] = Buf2(mem, nslot);
    mem += Buf2::shmem_size(nslot)/sizeof(Scalar);
  }
}

void GllFvRemapImpl::init_boundary_exchanges () {
  assert(m_data.qsize > 0); // after reset() called

  auto bm_exchange = Context::singleton().get<MpiBuffersManagerMap>()[MPI_EXCHANGE];

#pragma message "TODO: init_boundary_exchanges"
}

void init_gllfvremap_c (const int nf, const int nf_max, CF90Ptr fv_metdet, CF90Ptr g2f_remapd,
                        CF90Ptr f2g_remapd, CF90Ptr D_f, CF90Ptr Dinv_f) {
  auto& g = Context::singleton().get<GllFvRemap>();
  g.init_data(nf, nf_max, fv_metdet, g2f_remapd, f2g_remapd, D_f, Dinv_f);
}

template <typename T> using FV = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::HostSpace>;

void GllFvRemapImpl
::init_data (const int nf, const int nf_max, const Real* fv_metdet_r,
             const Real* g2f_remapd_r, const Real* f2g_remapd_r,
             const Real* D_f_r, const Real* Dinv_f_r) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;

  const int nf2 = nf*nf, nf2_max = nf_max*nf_max;
  auto& d = m_data;
  d.nf2 = nf2;
  const FV<const Real**>
    fg2f_remapd(g2f_remapd_r, np2, nf2_max),
    ff2g_remapd(f2g_remapd_r, nf2_max, np2),
    ffv_metdet(fv_metdet_r, nf2, d.nelemd);
  const FV<const Real****>
    fD_f(D_f_r, nf2, 2, 2, d.nelemd),
    fDinv_f(Dinv_f_r, nf2, 2, 2, d.nelemd);

  d.g2f_remapd = decltype(d.g2f_remapd)("g2f_remapd", nf2, np2);
  d.f2g_remapd = decltype(d.f2g_remapd)("f2g_remapd", np2, nf2);
  d.fv_metdet = decltype(d.fv_metdet)("fv_metdet", d.nelemd, nf2);
  d.D = decltype(d.D)("D", d.nelemd, np2, 2, 2);
  d.Dinv = decltype(d.D)("Dinv", d.nelemd, np2, 2, 2);
  d.D_f = decltype(d.D)("D_f", d.nelemd, nf2, 2, 2);
  d.Dinv_f = decltype(d.D)("Dinv_f", d.nelemd, nf2, 2, 2);

  const auto g2f_remapd = create_mirror_view(d.g2f_remapd);
  const auto f2g_remapd = create_mirror_view(d.f2g_remapd);
  const auto fv_metdet = create_mirror_view(d.fv_metdet);
  const auto D = create_mirror_view(d.D);
  const auto Dinv = create_mirror_view(d.Dinv);
  const auto D_f = create_mirror_view(d.D_f);
  const auto Dinv_f = create_mirror_view(d.Dinv_f);
  const auto cD = create_mirror_view(m_geometry.m_d); deep_copy(cD, m_geometry.m_d);
  const auto cDinv = create_mirror_view(m_geometry.m_dinv); deep_copy(cDinv, m_geometry.m_dinv);
  for (int i = 0; i < nf2; ++i)
    for (int j = 0; j < np2; ++j)
      g2f_remapd(i,j) = fg2f_remapd(j,i);
  for (int j = 0; j < np2; ++j)
    for (int i = 0; i < nf2; ++i)
      f2g_remapd(j,i) = ff2g_remapd(i,j);
  for (int ie = 0; ie < d.nelemd; ++ie) {
    for (int k = 0; k < nf2; ++k)
      fv_metdet(ie,k) = ffv_metdet(k,ie);
    for (int d0 = 0; d0 < 2; ++d0)
      for (int d1 = 0; d1 < 2; ++d1)
        for (int i = 0; i < np; ++i)
          for (int j = 0; j < np; ++j) {
            const auto k = np*i + j;
            D     (ie,k,d0,d1) = cD   (ie,d0,d1,i,j);
            Dinv  (ie,k,d0,d1) = cDinv(ie,d0,d1,i,j);
            D_f   (ie,k,d0,d1) = D_f   (ie,k,d0,d1);
            Dinv_f(ie,k,d0,d1) = Dinv_f(ie,k,d0,d1);
          }
  }
  deep_copy(d.fv_metdet, fv_metdet);
  deep_copy(d.g2f_remapd, g2f_remapd);
  deep_copy(d.f2g_remapd, f2g_remapd);
  deep_copy(d.D, D);
  deep_copy(d.Dinv, Dinv);
  deep_copy(d.D_f, D_f);
  deep_copy(d.Dinv_f, Dinv_f);
}

/* todo
   - add get_temperature to ElementOpts.hpp
   x compute_hydrostatic_p is already available
*/

void GllFvRemapImpl
::run_dyn_to_fv (const int time_idx, const Phys0T& ps, const Phys0T& phis, const Phys1T& Ts,
                 const Phys1T& omegas, const Phys2T& uvs, const Phys2T& qs) {
  const int np2 = GllFvRemapImpl::np2;
  const int nlevpk = num_lev_pack;
  const int nreal_per_slot1 = np2*max_num_lev_pack;
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

  const auto dp_fv = m_derived.m_divdp_proj; // store dp_fv between kernels
  const auto ps_v = m_state.m_ps_v;
  const auto metdet = m_geometry.m_metdet;
  const auto fv_metdet = m_data.fv_metdet;
  const auto g2f_remapd = m_data.g2f_remapd;
  const auto hvcoord = m_hvcoord;

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, m_tu_ne);
    const auto ie = kv.ie;
    
    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const EVU<Real*> rw1s(pack2real(rw1), nreal_per_slot1);
    
    const EVU<const Real*> fv_metdet_ie(&fv_metdet(ie,0), nf2), gll_metdet_ie(&metdet(ie,0,0), np2);

    const EVU<Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk); {
      const EVU<Real*> ps_v_fv(rw1s.data(), nf2); {
        const EVU<Real**> ps_v_ie(&ps_v(ie,time_idx,0,0), np2, 1), wrk(rw1s.data() + np2, np2, 1);
        remapd(team, nf2, np2, 1, g2f_remapd, gll_metdet_ie, fv_metdet_ie, ps_v_ie, wrk,
               EVU<Real**>(ps_v_fv.data(), nf2, 1));
      }
      calc_dp_fv(team, hvcoord, nf2, nlevpk, ps_v_fv, dp_fv_ie);
    }
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
