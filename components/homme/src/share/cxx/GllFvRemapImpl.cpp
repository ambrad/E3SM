/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "GllFvRemapImpl.hpp"

namespace Homme {

typedef ExecViewUnmanaged<      Scalar[NP*NP][NUM_LEV]> evus_np2_nlev;
typedef ExecViewUnmanaged<const Scalar[NP*NP][NUM_LEV]> evucs_np2_nlev;
typedef ExecViewUnmanaged<      Scalar[2][NP*NP][NUM_LEV]> evus_2_np2_nlev;
typedef ExecViewUnmanaged<const Scalar[2][NP*NP][NUM_LEV]> evucs_2_np2_nlev;
typedef ExecViewUnmanaged<Scalar*  > evus1;
typedef ExecViewUnmanaged<Scalar** > evus2;
typedef ExecViewUnmanaged<Scalar***> evus3;
typedef ExecViewUnmanaged<Real*  > evur1;
typedef ExecViewUnmanaged<Real** > evur2;
typedef ExecViewUnmanaged<Real***> evur3;
typedef ExecViewUnmanaged<const Real*  > evucr1;
typedef ExecViewUnmanaged<const Real** > evucr2;

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
    m_forcing(Context::singleton().get<ElementsForcing>()),
    m_geometry(Context::singleton().get<ElementsGeometry>()),
    m_tracers(Context::singleton().get<Tracers>()),
    m_tp_ne(1,1,1), m_tu_ne(m_tp_ne), // throwaway settings
    m_tp_ne_qsize(1,1,1), m_tu_ne_qsize(m_tp_ne_qsize), // throwaway settings
    m_tp_ne_dss(1,1,1), m_tu_ne_dss(m_tp_ne_dss) // throwaway settings
{
  nslot = calc_nslot(m_geometry.num_elems());
}

void GllFvRemapImpl::reset (const SimulationParams& params) {
  const auto num_elems = Context::singleton().get<Connectivity>().get_num_local_elements();

  m_data.qsize = params.qsize;
  Errors::runtime_check(m_data.qsize > 0, "GllFvRemapImpl requires qsize > 0");
  m_data.nelemd = num_elems;
  m_data.n_dss_fld = m_data.qsize + 2 + 1;

  m_tp_ne = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd);
  m_tp_ne_qsize = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd * m_data.qsize);
  m_tp_ne_dss = Homme::get_default_team_policy<ExecSpace>(m_data.nelemd * m_data.n_dss_fld);
  m_tu_ne = TeamUtils<ExecSpace>(m_tp_ne);
  m_tu_ne_qsize = TeamUtils<ExecSpace>(m_tp_ne_qsize);
  m_tu_ne_dss = TeamUtils<ExecSpace>(m_tp_ne_dss);

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

  auto& c = Context::singleton();

  {
    auto bm_exchange = c.get<MpiBuffersManagerMap>()[MPI_EXCHANGE];
    m_dss_be = std::make_shared<BoundaryExchange>();
    m_dss_be->set_buffers_manager(bm_exchange);
    m_dss_be->set_num_fields(0, 0, m_data.n_dss_fld);
    m_dss_be->register_field(m_tracers.fq, m_data.qsize, 0);
    m_dss_be->register_field(m_forcing.m_fm, 2, 0); // omit vertical
    m_dss_be->register_field(m_forcing.m_ft);
    m_dss_be->registration_completed();
  }

  {
    auto bm_exchange_minmax = c.get<MpiBuffersManagerMap>()[MPI_EXCHANGE_MIN_MAX];
    m_extrema_be = std::make_shared<BoundaryExchange>();
    BoundaryExchange& be = *m_extrema_be;
    be.set_buffers_manager(bm_exchange_minmax);
    be.set_num_fields(m_data.qsize, 0, 0);
    be.register_min_max_fields(m_tracers.qlim, m_data.qsize, 0);
    be.registration_completed();
  }
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
    fDinv_f(Dinv_f_r, nf, 2, 2, d.nelemd);

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
            D   (ie,k,d0,d1) = cD   (ie,d1,d0,i,j); // reverse d0,d1 in my use case
            Dinv(ie,k,d0,d1) = cDinv(ie,d1,d0,i,j);
          }
        for (int k = 0; k < nf2; ++k) {
          D_f   (ie,k,d0,d1) = fD_f   (k,d0,d1,ie); // reverse d0,d1
          Dinv_f(ie,k,d0,d1) = fDinv_f(k,d0,d1,ie);
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

// Min/max of q in each level.
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

template <typename TA>
static KOKKOS_FUNCTION void
calc_extrema_real1 (const KernelVariables& kv, const int n, const TA& q,
                    Real& qmin, Real& qmax) {
  GllFvRemapImpl::team_parallel_for_with_linear_index(
    kv.team, 1, 
    [&] (const int k) {
      qmin = qmax = q(0);
      for (int i = 1; i < n; ++i) {
        const auto qi = q(i);
        qmin = min(qmin, qi);
        qmax = max(qmax, qi);
      }
    });  
}

// qmin,max are already initialized with values. Augment these using q.
template <typename TA, typename TE>
static KOKKOS_FUNCTION void
augment_extrema (const KernelVariables& kv, const int n, const int nlev,
                 const TA& q, const TE& qmin, const TE& qmax) {
  const int packn = GllFvRemapImpl::packn;
  GllFvRemapImpl::team_parallel_for_with_linear_index(
    kv.team, nlev, 
    [&] (const int k) {
      auto& qmink = qmin(k);
      auto& qmaxk = qmax(k);
      for (int i = 0; i < n; ++i) {
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
  g::loop_ik(ttrg, tvr, [&] (int i, int k) { w1(i,k) = dpg(i,k)*qg(i,k); });
  g::remapd(kv.team, nf2, np2, nlev, g2f_remap, geog, sf, geof, w1, w1, w2);
  g::loop_ik(ttrf, tvr, [&] (int i, int k) { w2(i,k) /= dpf(i,k); });

  // Compute extremal q values in element on GLL grid. Use qf as tmp space.
  const evus1 qmin(&qf(0,iqf,0), nlev), qmax(&qf(1,iqf,0), nlev);
  calc_extrema(kv, np2, nlev, qg, qmin, qmax);

  // Apply CAAS to w2, the provisional q_f values.
  g::limiter_clip_and_sum(kv.team, nf2, nlev, sf, geof, qmin, qmax, dpf, w1, w2);
  // Copy to qf array.
  g::loop_ik(ttrf, tvr, [&] (int i, int k) { qf(i,iqf,k) = w2(i,k); });
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

  g::loop_ik(ttrf, tvr, [&] (int i, int k) { w1(i,k) = dpf(i,k)*qf(i,k); });
  g::remapd(kv.team, np2, nf2, nlev, f2g_remap, geof, 1, geog, w1, w1, qg);
  g::loop_ik(ttrg, tvr, [&] (int i, int k) { qg(i,k) /= dpg(i,k); });
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
       uvs.extent_int(3)/packn),
    q(real2pack(qs), qs.extent_int(0), qs.extent_int(1), qs.extent_int(2),
      qs.extent_int(3)/packn);

  const auto dp_fv = m_derived.m_divdp_proj; // store dp_fv between kernels
  const auto ps_v = m_state.m_ps_v;
  const auto phis_g = m_geometry.m_phis;
  const auto gll_metdet = m_geometry.m_metdet;
  const auto fv_metdet = m_data.fv_metdet;
  const auto w_ff = m_data.w_ff;
  const auto g2f_remapd = m_data.g2f_remapd;
  const auto Dinv = m_data.Dinv;
  const auto D_f = m_data.D_f;
  const auto v = m_state.m_v;
  const auto hvcoord = m_hvcoord;

  const auto fe = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, m_tu_ne);
    const auto ie = kv.ie;
    
    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const auto r2w = Kokkos::subview(m_data.buf2[0], kv.team_idx, all, all, all, all);
    const EVU<Real*> rw1s(pack2real(rw1), nreal_per_slot1);
    
    const evucr1 fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);

    // ps and dp_fv
    const evus2 dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk); {
      const evur2 ps_v_ie(&ps_v(ie,timeidx,0,0), np2, 1), wrk(rw1s.data(), np2, 1);
      remapd(team, nf2, np2, 1, g2f_remapd, gll_metdet_ie, w_ff, fv_metdet_ie,
             ps_v_ie, wrk, evur2(&ps(ie,0), nf2, 1));
      calc_dp_fv(team, hvcoord, nf2, nlevpk, EVU<Real*>(&ps(ie,0), nf2), dp_fv_ie);
    }

    { // phis
      const evucr1 phis_g_ie(&phis_g(ie,0,0), np2);
      Real qmin, qmax;
      calc_extrema_real1(kv, np2, phis_g_ie, qmin, qmax);
      const evur2 wrk(rw1s.data(), np2, 1), phis_f_ie(&phis(ie,0), nf2, 1);
      remapd(team, nf2, np2, 1, g2f_remapd, gll_metdet_ie, w_ff, fv_metdet_ie,
             evucr2(phis_g_ie.data(), np2, 1), wrk, phis_f_ie);
      limiter_clip_and_sum_real1(team, nf2, w_ff, fv_metdet_ie, qmin, qmax,
                                 evur1(rw1s.data(), nf2), evur1(phis_f_ie.data(), nf2));
    }

    // T

    // (u,v)
    remapd<false>(team, nf2, np2, nlevpk, g2f_remapd,
                  evur3(&Dinv(ie,0,0,0), np2, 2, 2), w_ff, evur3(&D_f(ie,0,0,0), nf2, 2, 2),
                  evucs_2_np2_nlev(&v(ie,timeidx,0,0,0,0)), evus_2_np2_nlev(r2w.data()),
                  evus3(&uv(ie,0,0,0), uv.extent_int(1), uv.extent_int(2), uv.extent_int(3)));

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
    
    const evucr1 fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);
    const EVU<const Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk);
    
    // q
    g2f_mixing_ratio(
      kv, np2, nf2, nlevpk, g2f_remapd, gll_metdet_ie, w_ff, fv_metdet_ie,
      evucs_np2_nlev(&dp_g(ie,timeidx,0,0,0)), dp_fv_ie, evucs_np2_nlev(&q_g(ie,iq,0,0,0)),
      evus_np2_nlev(rw1.data()), evus_np2_nlev(rw2.data()), iq,
      evus3(&q(ie,0,0,0), q.extent_int(1), q.extent_int(2), q.extent_int(3)));
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
  const auto gll_spheremp = m_geometry.m_spheremp;
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
    
    const evucr1 fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);

    // ps and dp_fv
    const evus2 dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk); {
      const evur2 ps_v_ie(&ps_v(ie,timeidx,0,0), np2, 1), w1(rw1s.data(), np2, 1);
      Real* const w2 = rw1s.data() + np2;
      remapd(team, nf2, np2, 1, g2f_remapd, gll_metdet_ie, w_ff, fv_metdet_ie,
             ps_v_ie, w1, evur2(w2, nf2, 1));
      calc_dp_fv(team, hvcoord, nf2, nlevpk, EVU<Real*>(w2, nf2), dp_fv_ie);
    }

    // (u,v)

    // T

  };
  parallel_for(m_tp_ne, fe);

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
    const auto all  = Kokkos::ALL();

    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    const auto rw2 = Kokkos::subview(m_data.buf1[1], kv.team_idx, all, all, all);
    const auto r2w = Kokkos::subview(m_data.buf2[0], kv.team_idx, all, all, all, all);

    const evucr1 fv_metdet_ie(&fv_metdet(ie,0), nf2),
      gll_metdet_ie(&gll_metdet(ie,0,0), np2);
    const EVU<const Scalar**> dp_fv_ie(&dp_fv(ie,0,0,0), nf2, nlevpk);

    if (q_adjustment) {
      // Get limiter bounds.
      const evus2 qf_ie(&r2w(1,0,0,0), nf2, nlevpk);
      loop_ik(ttrg, tvr, [&] (int i, int k) { qf_ie(i,k) = q(ie,i,iq,k); });
      calc_extrema(kv, nf2, nlevpk, qf_ie,
                   evus1(&qlim(ie,iq,0,0), nlevpk), evus1(&qlim(ie,iq,1,0), nlevpk));
      // FV Q_ten
      //   GLL Q0 -> FV Q0
      const evus2 dqf_ie(&r2w(0,0,0,0), nf2, nlevpk);
      const evucs_np2_nlev dp_g_ie(&dp_g(ie,timeidx,0,0,0)), qg_ie(&q_g(ie,iq,0,0,0));
      g2f_mixing_ratio(
        kv, np2, nf2, nlevpk, g2f_remapd, gll_metdet_ie,
        w_ff, fv_metdet_ie, dp_g_ie, dp_fv_ie, qg_ie,
        evus_np2_nlev(rw1.data()), evus_np2_nlev(rw2.data()),
        0, evus3(dqf_ie.data(), nf2, 1, nlevpk));
      //   FV Q_ten = FV Q1 - FV Q0
      loop_ik(ttrg, tvr, [&] (int i, int k) { dqf_ie(i,k) = qf_ie(i,k) - dqf_ie(i,k); });
      // GLL Q_ten
      const evus_np2_nlev dqg_ie(rw2.data());
      f2g_scalar_dp(kv, nf2, np2, nlevpk, f2g_remapd, fv_metdet_ie, gll_metdet_ie,
                    dp_fv_ie, dp_g_ie, dqf_ie, evus_np2_nlev(rw1.data()), dqg_ie);
      // GLL Q1
      const evus_np2_nlev fq_ie(&fq(ie,iq,0,0,0));
      loop_ik(ttrg, tvr, [&] (int i, int k) { fq_ie(i,k) = qg_ie(i,k) + dqg_ie(i,k); });
    } else {
      // FV Q_ten
      const evus2 dqf_ie(&r2w(0,0,0,0), nf2, nlevpk);
      loop_ik(ttrf, tvr, [&] (int i, int k) { dqf_ie(i,k) = dt*q(ie,i,iq,k)/dp_fv_ie(i,k); });
      // GLL Q_ten
      const evucs_np2_nlev dp_g_ie(&dp_g(ie,timeidx,0,0,0));
      const evus_np2_nlev dqg_ie(rw2.data());
      f2g_scalar_dp(kv, nf2, np2, nlevpk, f2g_remapd, fv_metdet_ie, gll_metdet_ie,
                    dp_fv_ie, dp_g_ie, dqf_ie, evus_np2_nlev(rw1.data()), dqg_ie);
      // GLL Q1
      const evucs_np2_nlev qg_ie(&q_g(ie,iq,0,0,0));
      const evus_np2_nlev fq_ie(&fq(ie,iq,0,0,0));
      loop_ik(ttrg, tvr, [&] (int i, int k) { fq_ie(i,k) = qg_ie(i,k) + dqg_ie(i,k); });
      // GLL Q0 -> FV Q0
      const evus2 qf_ie(&r2w(1,0,0,0), nf2, nlevpk);
      g2f_mixing_ratio(
        kv, np2, nf2, nlevpk, g2f_remapd, gll_metdet_ie,
        w_ff, fv_metdet_ie, dp_g_ie, dp_fv_ie, qg_ie,
        evus_np2_nlev(rw1.data()), evus_np2_nlev(rw2.data()),
        0, evus3(qf_ie.data(), nf2, 1, nlevpk));
      // FV Q1
      loop_ik(ttrg, tvr, [&] (int i, int k) { qf_ie(i,k) += dqf_ie(i,k); });
      // Get limiter bounds.
      calc_extrema(kv, nf2, nlevpk, qf_ie,
                   evus1(&qlim(ie,iq,0,0), nlevpk), evus1(&qlim(ie,iq,1,0), nlevpk));
    }
  };
  parallel_for(m_tp_ne_qsize, feq);

  // Halo exchange extrema data.
  m_extrema_be->exchange_min_max();

  const auto geq = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, qsize, m_tu_ne);
    const auto ie = kv.ie, iq = kv.iq;
    const auto all = Kokkos::ALL();
    const auto rw1 = Kokkos::subview(m_data.buf1[0], kv.team_idx, all, all, all);
    // Augment bounds with GLL Q0 bounds. This assures that if the tendency is
    // 0, GLL Q1 = GLL Q0.
    const evucs_np2_nlev qg_ie(&q_g(ie,iq,0,0,0));
    const evus1 qmin(&qlim(ie,iq,0,0), nlevpk), qmax(&qlim(ie,iq,1,0), nlevpk);
    augment_extrema(kv, np2, nlevpk, qg_ie, qmin, qmax);
    // Final GLL Q1, except for DSS.
    const evucr1 gll_spheremp_ie(&gll_spheremp(ie,0,0), np2);
    const evucs_np2_nlev dp_g_ie(&dp_g(ie,timeidx,0,0,0));
    const evus_np2_nlev fq_ie(&fq(ie,iq,0,0,0));
    limiter_clip_and_sum(kv.team, np2, nlevpk, 1, gll_spheremp_ie, qmin, qmax, dp_g_ie,
                         evus_np2_nlev(rw1.data()), fq_ie);
    if ( ! q_adjustment) {
      const auto ttrg = Kokkos::TeamThreadRange(kv.team, np2);
      const auto tvr  = Kokkos::ThreadVectorRange(kv.team, nlevpk);
      loop_ik(ttrg, tvr, [&] (int i, int k) {
        fq_ie(i,k) = dp_g_ie(i,k)*(fq_ie(i,k) - qg_ie(i,k))/dt;
      });
    }
  };
  parallel_for(m_tp_ne_qsize, geq);
}

void GllFvRemapImpl::run_fv_phys_to_dyn_dss () {
  const int np2 = GllFvRemapImpl::np2;
  const int nlevpk = num_lev_pack;
  const int n_dss_fld = m_data.n_dss_fld;
  const int qsize = m_data.qsize;
  const auto gll_spheremp = m_geometry.m_spheremp;
  const auto fq = m_tracers.fq;
  const auto fm = m_forcing.m_fm;
  const auto ft = m_forcing.m_ft;
  const auto f = KOKKOS_LAMBDA (const MT& team) {
    const int ie  = team.league_rank() / n_dss_fld;
    const int idx = team.league_rank() % n_dss_fld;
    const auto ttrg = Kokkos::TeamThreadRange(team, np2);
    const auto tvr  = Kokkos::ThreadVectorRange(team, nlevpk);
    const evucr1 s(&gll_spheremp(ie,0,0), np2);
    const evus2 f_ie((idx  < qsize     ? &fq(ie,idx,      0,0,0) :
                      idx == qsize + 2 ? &ft(ie,          0,0,0) :
                      /**/               &fm(ie,idx-qsize,0,0,0)),
                     np2, nlevpk);
    loop_ik(ttrg, tvr, [&] (int i, int k) { f_ie(i,k) *= s(i); });
  };
  Kokkos::parallel_for(m_tp_ne_dss, f);
  m_dss_be->exchange(m_geometry.m_rspheremp);
}

} // namespace Homme
