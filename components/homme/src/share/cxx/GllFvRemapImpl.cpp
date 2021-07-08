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

template <typename T> using FV = Kokkos::View<T, Kokkos::LayoutLeft, Kokkos::HostSpace>;

void GllFvRemapImpl
::init_data (const int nf, const int nf_max, const Real* fv_metdet_r,
             const Real* g2f_remapd_r, const Real* f2g_remapd_r,
             const Real* D_f_r, const Real* Dinv_f_r) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;

  if (nf <= 1)
    Errors::runtime_abort("GllFvRemap: In physics grid configuratoin nf x nf,"
                          " nf must be > 1.", Errors::err_not_implemented);

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

  d.w_ff = Real(4)/nf2;

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
      for (int d1 = 0; d1 < 2; ++d1) {
        for (int i = 0; i < np; ++i)
          for (int j = 0; j < np; ++j) {
            const auto k = np*i + j;
            D     (ie,k,d0,d1) = cD   (ie,d0,d1,i,j);
            Dinv  (ie,k,d0,d1) = cDinv(ie,d0,d1,i,j);
          }
        for (int k = 0; k < nf2; ++k) {
          D_f   (ie,k,d0,d1) = fD_f   (ie,k,d0,d1);
          Dinv_f(ie,k,d0,d1) = fDinv_f(ie,k,d0,d1);
        }
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

template <typename TA, typename TE>
static KOKKOS_FUNCTION void
calc_extrema (const KernelVariables& kv, const int n, const int nlev,
              const TA& q, const TE& qmin, const TE& qmax) {
  const int packn = GllFvRemapImpl::packn;
  GllFvRemapImpl::team_parallel_for_with_linear_index(
    kv.team, nlev, 
    [&] (const int k) {
      auto& qmink = qmin(k);
      auto& qmaxk = qmax(k);
      qmink = qmaxk = q(0,k);
      for (int i = 1; i < n; ++i) {
        const auto qik = q(i,k);
        VECTOR_SIMD_LOOP for (int s = 0; s < packn; ++s)
          qmink[s] = min(qmink[s], qik[s]);
        VECTOR_SIMD_LOOP for (int s = 0; s < packn; ++s)
          qmaxk[s] = max(qmaxk[s], qik[s]);
      }
    });  
}

// Remap a mixing ratio conservatively and preventing new extrema.
template <typename RT, typename GS, typename GT, typename DS, typename DT,
          typename QS, typename WT, typename QT>
static KOKKOS_FUNCTION void
g2f_mixing_ratio (const KernelVariables& kv, const int np2, const int nf2, const int nlev,
                  const RT& g2f_remap, const GS& geog, const Real sf, const GT& geof,
                  const DS& dpg, const DT& dpf, const QS& qg,
                  const WT& w1, const WT& w2, const int iqf, const QT& qf) {
  using g = GllFvRemapImpl;
  using Kokkos::parallel_for;
  const auto ttrg = Kokkos::TeamThreadRange(kv.team, np2);
  const auto ttrf = Kokkos::TeamThreadRange(kv.team, nf2);
  const auto tvr  = Kokkos::ThreadVectorRange(kv.team, nlev);

  // Linearly remap qdp GLL->FV. w2 holds q_f at end of this block.
  parallel_for( ttrg, [&] (const int i) {
    parallel_for(tvr, [&] (const int k) { w1(i,k) = dpg(i,k)*qg(i,k); }); });
  g::remapd(kv.team, nf2, np2, nlev, g2f_remap, 1, geog, sf, geof, w1, w1, w2);
  parallel_for( ttrf, [&] (const int i) {
    parallel_for(tvr, [&] (const int k) { w2(i,k) /= dpf(i,k); }); });

  // Compute extremal q values in element on GLL grid. Use qf as tmp space.
  const g::EVU<Scalar*> qmin(&qf(0,iqf,0), nlev), qmax(&qf(1,iqf,0), nlev);
  calc_extrema(kv, np2, nlev, qg, qmin, qmax);

  // Apply CAAS to w2, the provisional q_f values.
  g::limiter_clip_and_sum(kv.team, nf2, nlev, sf, geof, qmin, qmax, dpf, w1, w2);
  // Copy to qf array.
  parallel_for( ttrf, [&] (const int i) {
    parallel_for(tvr, [&] (const int k) { qf(i,iqf,k) = w2(i,k); }); });
}

template <typename RT, typename GS, typename GT, typename DS, typename DT, typename WT,
          typename QFT, typename QGT>
static KOKKOS_FUNCTION void
f2g_scalar_dp (const KernelVariables& kv, const int nf2, const int np2, const int nlev,
               const RT& f2g_remap, const GS& geof, const GT& geog, const DS& dpf,
               const DT& dpg, const QFT& qf, const WT& w1, const QGT& qg) {
  using g = GllFvRemapImpl;
  using Kokkos::parallel_for;
  const auto ttrf = Kokkos::TeamThreadRange(kv.team, nf2);
  const auto ttrg = Kokkos::TeamThreadRange(kv.team, np2);
  const auto tvr  = Kokkos::ThreadVectorRange(kv.team, nlev);
  const int iq = kv.iq;

  parallel_for( ttrf, [&] (const int i) {
    parallel_for(tvr, [&] (const int k) { w1(i,k) = dpf(i,k)*qf(i,k); }); });
  g::remapd(kv.team, np2, nf2, nlev, f2g_remap, 1, geof, 1, geog, w1, w1, qg);
  parallel_for( ttrg, [&] (const int i) {
    parallel_for(tvr, [&] (const int k) { qg(i,k) /= dpg(i,k); }); });  
}

void GllFvRemapImpl
::run_dyn_to_fv_phys (const int timeidx, const Phys1T& ps, const Phys1T& phis, const Phys2T& Ts,
                      const Phys2T& omegas, const Phys3T& uvs, const Phys3T& qs) {
  const int np2 = GllFvRemapImpl::np2;
  const int nlevpk = num_lev_pack;
  const int nreal_per_slot1 = np2*max_num_lev_pack;
  const auto nf2 = m_data.nf2;
  const auto nelemd = m_data.nelemd;
  const auto qsize = m_data.qsize;

  assert(ps.extent_int(0) >= nelemd && ps.extent_int(1) >= nf2);
  assert(phis.extent_int(0) >= nelemd && phis.extent_int(1) >= nf2);
  assert(Ts.extent_int(0) >= nelemd && Ts.extent_int(1) >= nf2 && Ts.extent_int(2) % packn == 0);
  assert(omegas.extent_int(0) >= nelemd && omegas.extent_int(1) >= nf2 &&
         omegas.extent_int(2) % packn == 0);
  assert(uvs.extent_int(0) >= nelemd && uvs.extent_int(1) >= nf2 && uvs.extent_int(2) == 2 &&
         uvs.extent_int(3) % packn == 0);
  assert(qs.extent_int(0) >= nelemd && qs.extent_int(1) >= nf2 && qs.extent_int(2) >= qsize &&
         qs.extent_int(3) % packn == 0);

  VPhys2T
    T(real2pack(Ts), Ts.extent_int(0), Ts.extent_int(1), Ts.extent_int(2)/packn),
    omega(real2pack(omegas), omegas.extent_int(0), omegas.extent_int(1),
          omegas.extent_int(2)/packn);
  VPhys3T
    uv(real2pack(uvs), uvs.extent_int(0), uvs.extent_int(1), uvs.extent_int(2),
       uvs.extent_int(2)/packn),
    q(real2pack(qs), qs.extent_int(0), qs.extent_int(1), qs.extent_int(2),
      qs.extent_int(3)/packn);

  const auto dp_fv = m_derived.m_divdp_proj; // store dp_fv between kernels
  const auto ps_v = m_state.m_ps_v;
  const auto gll_metdet = m_geometry.m_metdet;
  const auto fv_metdet = m_data.fv_metdet;
  const auto w_ff = m_data.w_ff;
  const auto g2f_remapd = m_data.g2f_remapd;
  const auto hvcoord = m_hvcoord;

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, m_tu_ne);
    const auto ie = kv.ie;
    
    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const EVU<Real*> rw1s(pack2real(rw1), nreal_per_slot1);
    
    const EVU<const Real*> fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);

    // ps and dp_fv
    const EVU<Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk); {
      const EVU<Real**> ps_v_ie(&ps_v(ie,timeidx,0,0), np2, 1), wrk(rw1s.data(), np2, 1);
      remapd(team, nf2, np2, 1, g2f_remapd,
             1, gll_metdet_ie, w_ff, fv_metdet_ie,
             ps_v_ie, wrk, EVU<Real**>(&ps(ie,0), nf2, 1));
      calc_dp_fv(team, hvcoord, nf2, nlevpk, EVU<Real*>(&ps(ie,0), nf2), dp_fv_ie);
    }

    // phis

    // T

    // (u,v)

    // omega

  };
  Kokkos::parallel_for(m_tp_ne, fe);

  const auto dp_g = m_state.m_dp3d;
  const auto q_g = m_tracers.Q;
  const auto feq = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, qsize, m_tu_ne);
    const auto ie = kv.ie, iq = kv.iq;

    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const auto rw2 = Kokkos::subview(m_data.buf1[1], kv.team_idx, all, all, all);
    
    const EVU<const Real*> fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);
    const EVU<const Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk);
    
    // q
    g2f_mixing_ratio(
      kv, np2, nf2, nlevpk, g2f_remapd, gll_metdet_ie, w_ff, fv_metdet_ie,
      EVU<const Scalar[NP*NP][NUM_LEV]>(&dp_g(ie,timeidx,0,0,0)), dp_fv_ie,
      EVU<const Scalar[NP*NP][NUM_LEV]>(&q_g(ie,iq,0,0,0)),
      EVU<Scalar[NP*NP][NUM_LEV]>(rw1.data()), EVU<Scalar[NP*NP][NUM_LEV]>(rw2.data()),
      iq, EVU<Scalar***>(&q(ie,0,0,0), q.extent_int(1), q.extent_int(2), q.extent_int(3)));
  };
  Kokkos::parallel_for(m_tp_ne_qsize, feq);
}

void GllFvRemapImpl::
run_fv_phys_to_dyn (const int timeidx, const Real dt,
                    const CPhys2T& Ts, const CPhys3T& uvs, const CPhys3T& qs) {
  using Kokkos::parallel_for;

  const int np2 = GllFvRemapImpl::np2;
  const int nlevpk = num_lev_pack;
  const int nreal_per_slot1 = np2*max_num_lev_pack;
  const auto nf2 = m_data.nf2;
  const auto nelemd = m_data.nelemd;
  const auto qsize = m_data.qsize;

  assert(Ts.extent_int(0) >= nelemd && Ts.extent_int(1) >= nf2 && Ts.extent_int(2) % packn == 0);
  assert(uvs.extent_int(0) >= nelemd && uvs.extent_int(1) >= nf2 && uvs.extent_int(2) == 2 &&
         uvs.extent_int(3) % packn == 0);
  assert(qs.extent_int(0) >= nelemd && qs.extent_int(1) >= nf2 && qs.extent_int(2) >= qsize &&
         qs.extent_int(3) % packn == 0);

  const auto& sp = Context::singleton().get<SimulationParams>();
  const bool q_adjustment = sp.ftype != ForcingAlg::FORCING_0;

  CVPhys2T
    T(creal2pack(Ts), Ts.extent_int(0), Ts.extent_int(1), Ts.extent_int(2)/packn);
  CVPhys3T
    uv(creal2pack(uvs), uvs.extent_int(0), uvs.extent_int(1), uvs.extent_int(2),
       uvs.extent_int(3)/packn),
    q(creal2pack(qs), qs.extent_int(0), qs.extent_int(1), qs.extent_int(2),
      qs.extent_int(3)/packn);

  const auto dp_fv = m_derived.m_divdp_proj; // store dp_fv between kernels
  const auto ps_v = m_state.m_ps_v;
  const auto gll_metdet = m_geometry.m_metdet;
  const auto w_ff = m_data.w_ff;
  const auto fv_metdet = m_data.fv_metdet;
  const auto g2f_remapd = m_data.g2f_remapd;
  const auto f2g_remapd = m_data.f2g_remapd;
  const auto hvcoord = m_hvcoord;

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, m_tu_ne);
    const auto ie = kv.ie;

    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const EVU<Real*> rw1s(pack2real(rw1), nreal_per_slot1);
    
    const EVU<const Real*> fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);

    // ps and dp_fv
    const EVU<Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk); {
      const EVU<Real**> ps_v_ie(&ps_v(ie,timeidx,0,0), np2, 1), w1(rw1s.data(), np2, 1);
      Real* const w2 = rw1s.data() + np2;
      remapd(team, nf2, np2, 1, g2f_remapd,
             1, gll_metdet_ie, w_ff, fv_metdet_ie,
             ps_v_ie, w1, EVU<Real**>(w2, nf2, 1));
      calc_dp_fv(team, hvcoord, nf2, nlevpk, EVU<Real*>(w2, nf2), dp_fv_ie);
    }
    
  };
  parallel_for(m_tp_ne, fe);

  assert(q_adjustment); // for now
  const auto dp_g = m_state.m_dp3d;
  const auto q_g = m_tracers.Q;
  const auto fq = m_tracers.fq;
  const auto qlim = m_tracers.qlim;
  const auto feq = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, qsize, m_tu_ne);
    const auto ie = kv.ie, iq = kv.iq;
    const auto ttrf = Kokkos::TeamThreadRange(kv.team, nf2);
    const auto ttrg = Kokkos::TeamThreadRange(kv.team, np2);
    const auto tvr  = Kokkos::ThreadVectorRange(kv.team, nlevpk);
    const auto all = Kokkos::ALL();

    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const auto rw2 = Kokkos::subview(m_data.buf1[1], kv.team_idx, all, all, all);
    const auto r2w = Kokkos::subview(m_data.buf2[0], kv.team_idx, all, all, all, all);

    const EVU<const Real*> fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);
    const EVU<const Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk);

    // Get limiter bounds.
    const EVU<Scalar**> qf_ie(&r2w(1,0,0,0), nf2, nlevpk);
    parallel_for( ttrf, [&] (const int i) {
        parallel_for(tvr, [&] (const int k) { qf_ie(i,k) = q(ie,i,iq,k); }); });
    calc_extrema(kv, nf2, nlevpk, qf_ie,
                 EVU<Scalar*>(&qlim(ie,iq,0,0), nlevpk), EVU<Scalar*>(&qlim(ie,iq,1,0), nlevpk));
    // FV Q_ten
    //   GLL Q0 -> FV Q0
    const EVU<Scalar**> dqf_ie(&r2w(0,0,0,0), nf2, nlevpk);
    const EVU<const Scalar[NP*NP][NUM_LEV]>
      dp_g_ie(&dp_g(ie,timeidx,0,0,0)), qg_ie(&q_g(ie,iq,0,0,0));
    g2f_mixing_ratio(
      kv, np2, nf2, nlevpk, g2f_remapd, gll_metdet_ie,
      w_ff, fv_metdet_ie, dp_g_ie, dp_fv_ie, qg_ie,
      EVU<Scalar[NP*NP][NUM_LEV]>(rw1.data()), EVU<Scalar[NP*NP][NUM_LEV]>(rw2.data()),
      0, EVU<Scalar***>(dqf_ie.data(), nf2, 1, nlevpk));
    //   FV Q_ten = FV Q1 - FV Q0
    parallel_for( ttrf, [&] (const int i) {
      parallel_for(tvr, [&] (const int k) { dqf_ie(i,k) = qf_ie(i,k) - dqf_ie(i,k); }); });
    //   GLL Q_ten
    const EVU<Scalar[NP*NP][NUM_LEV]> dqg_ie(rw2.data());
    f2g_scalar_dp(kv, nf2, np2, nlevpk, f2g_remapd, fv_metdet_ie, gll_metdet_ie,
                  dp_fv_ie, dp_g_ie, dqf_ie, EVU<Scalar[NP*NP][NUM_LEV]>(rw1.data()), dqg_ie);
    //   GLL Q1
    const EVU<Scalar[NP*NP][NUM_LEV]> fq_ie(&fq(ie,iq,0,0,0));
    parallel_for( ttrg, [&] (const int i) {
      parallel_for(tvr, [&] (const int k) { fq_ie(i,k) = qg_ie(i,k) + dqg_ie(i,k); }); });
  };
  parallel_for(m_tp_ne_qsize, feq);

  // halo exchange

  const auto geq = KOKKOS_LAMBDA (const MT& team) {
    
  };
  parallel_for(m_tp_ne_qsize, geq);
}

} // namespace Homme
