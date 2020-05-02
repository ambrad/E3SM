#include "compose_cedr_cdr.hpp"
#include "compose_cedr_sl.hpp"

namespace homme {
namespace sl {

void solve_local (const Int ie, const Int k, const Int q,
                  const Int tl_np1, const Int n1_qdp, const Int np, 
                  const bool scalar_bounds, const Int limiter_option,
                  const FA1<const Real>& spheremp, const FA3<const Real>& dp3d_c,
                  const FA4<const Real>& q_min, const FA4<const Real>& q_max,
                  const Real Qm, FA4<Real>& qdp_c, FA3<Real>& q_c) {
  static constexpr Int max_np = 4, max_np2 = max_np*max_np;
  const Int np2 = np*np;
  cedr_assert(np <= max_np);

  Real wa[max_np2], qlo[max_np2], qhi[max_np2], y[max_np2], x[max_np2];
  Real rhom = 0;
  for (Int g = 0; g < np2; ++g) {
    const Real rhomij = dp3d_c(g,k,tl_np1) * spheremp(g);
    rhom += rhomij;
    wa[g] = rhomij;
    y[g] = q_c(g,k,q);
    x[g] = y[g];
  }

  //todo Replace with ReconstructSafely.
  if (scalar_bounds) {
    qlo[0] = q_min(0,k,q,ie);
    qhi[0] = q_max(0,k,q,ie);
    const Int N = std::min(max_np2, np2);
    for (Int i = 1; i < N; ++i) qlo[i] = qlo[0];
    for (Int i = 1; i < N; ++i) qhi[i] = qhi[0];
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
    const Int N = std::min(max_np2, np2);
    for (Int g = 0; g < np2; ++g) {
      qlo[g] = q_min(g,k,q,ie);
      qhi[g] = q_max(g,k,q,ie);
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
        for (Int i = 1; i < N; ++i) {
          qlo_s = std::min(qlo_s, qlo[i]);
          qhi_s = std::max(qhi_s, qhi[i]);
        }
        for (Int i = 0; i < N; ++i)
          x[i] = cedr::impl::max(qlo_s, cedr::impl::min(qhi_s, x[i]));
      }
      if (info == 0 || trial == 1) break;
      switch (trial) {
      case 0: {
        Real qlo_s = qlo[0], qhi_s = qhi[0];
        for (Int i = 1; i < N; ++i) {
          qlo_s = std::min(qlo_s, qlo[i]);
          qhi_s = std::max(qhi_s, qhi[i]);
        }
        const Int N = std::min(max_np2, np2);
        for (Int i = 0; i < N; ++i) qlo[i] = qlo_s;
        for (Int i = 0; i < N; ++i) qhi[i] = qhi_s;
      } break;
      case 1: {
        const Real q = Qm / rhom;
        for (Int i = 0; i < N; ++i) qlo[i] = std::min(qlo[i], q);
        for (Int i = 0; i < N; ++i) qhi[i] = std::max(qhi[i], q);                
      } break;
      }
    }
  }
        
  for (Int g = 0; g < np2; ++g) {
    q_c(g,k,q) = x[g];
    qdp_c(g,k,q,n1_qdp) = q_c(g,k,q) * dp3d_c(g,k,tl_np1);
  }
}

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

void run_local (CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete, const bool scalar_bounds,
                const Int limiter_option) {
  const Int np = d.np, np2 = np*np, nlev = d.nlev, qsize = d.qsize,
    nlevwrem = cdr.nsuplev*cdr.nsublev;

  FA4<      Real> q_min(q_min_r, np2, nlev, qsize, nete+1);
  FA4<const Real> q_max(q_max_r, np2, nlev, qsize, nete+1);

  for (Int ie = nets; ie <= nete; ++ie) {
    FA1<const Real> spheremp(d.spheremp[ie], np2);
    FA4<      Real> qdp_c(d.qdp_pc[ie], np2, nlev, d.qsize_d, 2);
    FA3<const Real> dp3d_c(d.dp3d_c[ie], np2, nlev, d.timelevels);
    FA3<      Real> q_c(d.q_c[ie], np2, nlev, d.qsize_d);
#ifdef COMPOSE_COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const Int ti = cdr.cdr_over_super_levels ? q : spli*qsize + q;
        if (cdr.caas_in_suplev) {
          const auto ie_idx = cdr.cdr_over_super_levels ?
            cdr.nsuplev*ie + spli :
            ie;
          const auto lci = cdr.ie2lci[ie_idx];
          const Real Qm_tot = cdr.cdr->get_Qm(lci, ti);
          Real Qm_min_tot = 0, Qm_max_tot = 0;
          Real rhom[CDR::nsublev_per_suplev], Qm[CDR::nsublev_per_suplev],
            Qm_min[CDR::nsublev_per_suplev], Qm_max[CDR::nsublev_per_suplev];
          // Redistribute mass in the vertical direction of the super level.
          Int n = cdr.nsublev;
          for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
            const Int k = k0 + sbli;
            if (k >= nlev) {
              n = sbli;
              break;
            }
            rhom[sbli] = 0; Qm[sbli] = 0; Qm_min[sbli] = 0; Qm_max[sbli] = 0;
            for (Int g = 0; g < np2; ++g) {
              const Real rhomij = dp3d_c(g,k,d.tl_np1) * spheremp(g);
              rhom[sbli] += rhomij;
              Qm[sbli] += q_c(g,k,q) * rhomij;
              Qm_min[sbli] += q_min(g,k,q,ie) * rhomij;
              Qm_max[sbli] += q_max(g,k,q,ie) * rhomij;
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
                  q_min_s = q_min(g,k,q,ie);
                  q_max_s = q_max(g,k,q,ie);
                  first = false;
                } else {
                  q_min_s = std::min(q_min_s, q_min(g,k,q,ie));
                  q_max_s = std::max(q_max_s, q_max(g,k,q,ie));
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
            solve_local(ie, k, q, d.tl_np1, d.n1_qdp, np,
                        scalar_bounds, limiter_option,
                        spheremp, dp3d_c, q_min, q_max, Qm[i], qdp_c, q_c);
          }
        } else {
          for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
            const Int k = k0 + sbli;
            if (k >= nlev) break;
            const auto ie_idx = cdr.cdr_over_super_levels ?
              nlevwrem*ie + k :
              cdr.nsublev*ie + sbli;
            const auto lci = cdr.ie2lci[ie_idx];
            const Real Qm = cdr.cdr->get_Qm(lci, ti);
            solve_local(ie, k, q, d.tl_np1, d.n1_qdp, np,
                        scalar_bounds, limiter_option,
                        spheremp, dp3d_c, q_min, q_max, Qm, qdp_c, q_c);
          }
        }
      }
    }
  }
}

} // namespace sl
} // namespace homme
