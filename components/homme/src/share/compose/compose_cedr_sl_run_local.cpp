#include "compose_cedr.hpp"
#include "compose_cedr_cdr.hpp"
#include "compose_cedr_sl.hpp"

namespace homme {
namespace sl {

template <int np_,
          typename CV2, typename CV4, typename CV5, typename QV5,
          typename V4, typename V5>
KOKKOS_FUNCTION
void solve_local (const Int ie, const Int k, const Int q,
                  const Int np1, const Int n1_qdp, const Int np,
                  const bool scalar_bounds, const Int limiter_option,
                  const CV2& spheremp, const CV4& dp3d_c,
                  const QV5& q_min, const CV5& q_max,
                  const Real Qm, V5& qdp_c, V4& q_c) {
  cedr_assert(np == np_);
  static const Int np2 = np_*np_;

  Real wa[np2], qlo[np2], qhi[np2], y[np2], x[np2];
  Real rhom = 0;
  for (Int g = 0; g < np2; ++g) {
    const Real rhomij = dp3d_c(ie,np1,g,k) * spheremp(ie,g);
    rhom += rhomij;
    wa[g] = rhomij;
    y[g] = q_c(ie,q,g,k);
    x[g] = y[g];
  }

  //todo Replace with ReconstructSafely.
  if (scalar_bounds) {
    qlo[0] = idx_qext(q_min,ie,q,0,k);
    qhi[0] = idx_qext(q_max,ie,q,0,k);
    for (Int i = 1; i < np2; ++i) qlo[i] = qlo[0];
    for (Int i = 1; i < np2; ++i) qhi[i] = qhi[0];
    // We can use either 2-norm minimization or ClipAndAssuredSum for
    // the local filter. CAAS is the faster. It corresponds to limiter
    // = 0. 2-norm minimization is the same in spirit as limiter = 8,
    // but it assuredly achieves the first-order optimality conditions
    // whereas limiter 8 does not.
    if (limiter_option == 8)
      cedr::local::solve_1eq_bc_qp(np2, wa, wa, Qm, qlo, qhi, y, x);
    else {
      // We need to use *some* limiter; if 8 isn't chosen, default to
      // CAAS.
      cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x);
    }
  } else {
    for (Int g = 0; g < np2; ++g) {
      qlo[g] = idx_qext(q_min,ie,q,g,k);
      qhi[g] = idx_qext(q_max,ie,q,g,k);
    }
    for (Int trial = 0; trial < 3; ++trial) {
      int info;
      if (limiter_option == 8) {
        info = cedr::local::solve_1eq_bc_qp(
          np2, wa, wa, Qm, qlo, qhi, y, x);
        if (info == 1) info = 0;
      } else {
        info = 0;
        cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x, false /* clip */);
        // Clip for numerics against the cell extrema.
        Real qlo_s = qlo[0], qhi_s = qhi[0];
        for (Int i = 1; i < np2; ++i) {
          qlo_s = ko::min(qlo_s, qlo[i]);
          qhi_s = ko::max(qhi_s, qhi[i]);
        }
        for (Int i = 0; i < np2; ++i)
          x[i] = ko::max(qlo_s, cedr::impl::min(qhi_s, x[i]));
      }
      if (info == 0 || trial == 1) break;
      switch (trial) {
      case 0: {
        Real qlo_s = qlo[0], qhi_s = qhi[0];
        for (Int i = 1; i < np2; ++i) {
          qlo_s = ko::min(qlo_s, qlo[i]);
          qhi_s = ko::max(qhi_s, qhi[i]);
        }
        for (Int i = 0; i < np2; ++i) qlo[i] = qlo_s;
        for (Int i = 0; i < np2; ++i) qhi[i] = qhi_s;
      } break;
      case 1: {
        const Real q = Qm / rhom;
        for (Int i = 0; i < np2; ++i) qlo[i] = ko::min(qlo[i], q);
        for (Int i = 0; i < np2; ++i) qhi[i] = ko::max(qhi[i], q);                
      } break;
      }
    }
  }
        
  for (Int g = 0; g < np2; ++g) {
    q_c(ie,q,g,k) = x[g];
    qdp_c(ie,n1_qdp,q,g,k) = q_c(ie,q,g,k) * dp3d_c(ie,np1,g,k);
  }
}

KOKKOS_FUNCTION
Int vertical_caas_backup (const Int n, Real* rhom,
                          const Real q_min, const Real q_max,
                          Real Qmlo_tot, Real Qmhi_tot, const Real Qm_tot,
                          Real* Qmlo, Real* Qmhi, Real* Qm) {
  Int status = 0;
  if (Qm_tot < Qmlo_tot || Qm_tot > Qmhi_tot) {
    if (Qm_tot < Qmlo_tot) {
      status = -2;
      for (Int i = 0; i < n; ++i) Qmhi[i] = Qmlo[i];
      for (Int i = 0; i < n; ++i) Qmlo[i] = q_min*rhom[i];
      Qmlo_tot = 0;
      for (Int i = 0; i < n; ++i) Qmlo_tot += Qmlo[i];
      if (Qm_tot < Qmlo_tot) status = -4;
    } else {
      status = -1;
      for (Int i = 0; i < n; ++i) Qmlo[i] = Qmhi[i];
      for (Int i = 0; i < n; ++i) Qmhi[i] = q_max*rhom[i];
      Qmhi_tot = 0;
      for (Int i = 0; i < n; ++i) Qmhi_tot += Qmhi[i];
      if (Qm_tot > Qmhi_tot) status = -3;
    }
    if (status < -2) {
      Real rhom_tot = 0;
      for (Int i = 0; i < n; ++i) rhom_tot += rhom[i];
      const Real q = Qm_tot/rhom_tot;
      for (Int i = 0; i < n; ++i) Qm[i] = q*rhom[i];
      return status;
    }
  }
  for (Int i = 0; i < n; ++i) rhom[i] = 1;
  cedr::local::caas(n, rhom, Qm_tot, Qmlo, Qmhi, Qm, Qm, false);
  return status;
}

template <int np_, typename MT, typename CDRT>
void run_local (CDR<MT>& cdr, CDRT* cedr_cdr_p,
                const Data& d, Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete, const bool scalar_bounds,
                const Int limiter_option) {
  const auto& ta = *d.ta;
  cedr_assert(ta.np == np_);
  static const Int np = np_, np2 = np_*np_;
  const Int nlev = ta.nlev, qsize = ta.qsize, nlevwrem = cdr.nsuplev*cdr.nsublev;

#ifdef COMPOSE_PORT_DEV_VIEWS
  const auto q_min = ta.q_min;
  const auto q_max = ta.q_max;
#else
  const QExtremaH<ko::MachineTraits>
    q_min(q_min_r, ta.nelemd, ta.qsize, ta.nlev, np2);
  const QExtremaHConst<ko::MachineTraits>
    q_max(q_max_r, ta.nelemd, ta.qsize, ta.nlev, np2);
#endif
  const auto np1 = ta.np1;
  const auto n1_qdp = ta.n1_qdp;
  const auto spheremp = ta.spheremp;
  const auto dp3d_c = ta.dp3d;
  const auto qdp_c = ta.qdp;
  const auto q_c = ta.q;

  const Int nsublev = cdr.nsublev;
  const Int nsuplev = cdr.nsuplev;
  const auto cdr_over_super_levels = cdr.cdr_over_super_levels;
  const auto caas_in_suplev = cdr.caas_in_suplev;
  const auto ie2lci = cdr.ie2lci;
  const auto cedr_cdr = *cedr_cdr_p;
  const auto f = KOKKOS_LAMBDA (const Int& idx) {
    const Int ie = nets + idx/(nsuplev*qsize);
    const Int q = (idx / nsuplev) % qsize;
    const Int spli = idx % nsuplev;
    const Int k0 = nsublev*spli;
    const Int ti = cdr_over_super_levels ? q : spli*qsize + q;
    if (caas_in_suplev) {
      const auto ie_idx = (cdr_over_super_levels ?
                           nsuplev*ie + spli :
                           ie);
      const auto lci = ie2lci[ie_idx];
      const Real Qm_tot = cedr_cdr.get_Qm(lci, ti);
      Real Qm_min_tot = 0, Qm_max_tot = 0;
      Real rhom[CDR<MT>::nsublev_per_suplev], Qm[CDR<MT>::nsublev_per_suplev],
        Qm_min[CDR<MT>::nsublev_per_suplev], Qm_max[CDR<MT>::nsublev_per_suplev];
      // Redistribute mass in the vertical direction of the super level.
      Int n = nsublev;
      for (Int sbli = 0; sbli < nsublev; ++sbli) {
        const Int k = k0 + sbli;
        if (k >= nlev) {
          n = sbli;
          break;
        }
        rhom[sbli] = 0; Qm[sbli] = 0; Qm_min[sbli] = 0; Qm_max[sbli] = 0;
        for (Int g = 0; g < np2; ++g) {
          const Real rhomij = dp3d_c(ie,np1,g,k) * spheremp(ie,g);
          rhom[sbli] += rhomij;
          Qm[sbli] += q_c(ie,q,g,k) * rhomij;
          Qm_min[sbli] += idx_qext(q_min,ie,q,g,k) * rhomij;
          Qm_max[sbli] += idx_qext(q_max,ie,q,g,k) * rhomij;
        }
        Qm_min_tot += Qm_min[sbli];
        Qm_max_tot += Qm_max[sbli];
      }
      if (Qm_tot >= Qm_min_tot && Qm_tot <= Qm_max_tot) {
        for (Int i = 0; i < n; ++i) rhom[i] = 1;
        cedr::local::caas(n, rhom, Qm_tot, Qm_min, Qm_max, Qm, Qm, false);
      } else {
        Real q_min_s, q_max_s;
        bool first = true;
        for (Int sbli = 0; sbli < n; ++sbli) {
          const Int k = k0 + sbli;
          for (Int g = 0; g < np2; ++g) {
            if (first) {
              q_min_s = idx_qext(q_min,ie,q,g,k);
              q_max_s = idx_qext(q_max,ie,q,g,k);
              first = false;
            } else {
              q_min_s = ko::min(q_min_s, idx_qext(q_min,ie,q,g,k));
              q_max_s = ko::max(q_max_s, idx_qext(q_max,ie,q,g,k));
            }
          }
        }
        vertical_caas_backup(n, rhom, q_min_s, q_max_s,
                             Qm_min_tot, Qm_max_tot, Qm_tot,
                             Qm_min, Qm_max, Qm);
      }
      // Redistribute mass in the horizontal direction of each level.
      for (Int i = 0; i < n; ++i) {
        const Int k = k0 + i;
        solve_local<np_>(ie, k, q, np1, n1_qdp, np,
                         scalar_bounds, limiter_option,
                         spheremp, dp3d_c, q_min, q_max, Qm[i], qdp_c, q_c);
      }
    } else {
      for (Int sbli = 0; sbli < nsublev; ++sbli) {
        const Int k = k0 + sbli;
        if (k >= nlev) break;
        const auto ie_idx = (cdr_over_super_levels ?
                             nlevwrem*ie + k :
                             nsublev*ie + sbli);
        const auto lci = ie2lci[ie_idx];
        const Real Qm = cedr_cdr.get_Qm(lci, ti);
        solve_local<np_>(ie, k, q, np1, n1_qdp, np,
                         scalar_bounds, limiter_option,
                         spheremp, dp3d_c, q_min, q_max, Qm, qdp_c, q_c);
      }
    }
  };
  ko::fence();
  { Timer t("03_run_local");
    ko::parallel_for(ko::RangePolicy<typename MT::DES>(0, (nete - nets + 1)*nsuplev*qsize), f); }
}

template <typename MT>
void run_local (CDR<MT>& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete, const bool scalar_bounds,
                const Int limiter_option) {
  if (dynamic_cast<typename CDR<MT>::QLTT*>(cdr.cdr.get()))
    run_local<4, MT, typename CDR<MT>::QLTT>(
      cdr, dynamic_cast<typename CDR<MT>::QLTT*>(cdr.cdr.get()),
      d, q_min_r, q_max_r, nets, nete, scalar_bounds, limiter_option);
  else if (dynamic_cast<typename CDR<MT>::CAAST*>(cdr.cdr.get()))
    run_local<4, MT, typename CDR<MT>::CAAST>(
      cdr, dynamic_cast<typename CDR<MT>::CAAST*>(cdr.cdr.get()),
      d, q_min_r, q_max_r, nets, nete, scalar_bounds, limiter_option);
  else
    cedr_throw_if(true, "run_local: could not cast cdr.");
}

template void
run_local(CDR<ko::MachineTraits>& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
          const Int nets, const Int nete, const bool scalar_bounds,
          const Int limiter_option);

} // namespace sl
} // namespace homme