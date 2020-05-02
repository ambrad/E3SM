#include "compose_cedr_cdr.hpp"
#include "compose_cedr_sl.hpp"

namespace homme {
namespace sl {

static void run_cdr (CDR& q) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
  q.cdr->run();
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

void run_global (CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                 const Int nets, const Int nete) {
  const Int np = d.np, np2 = np*np, nlev = d.nlev, qsize = d.qsize,
    nlevwrem = cdr.nsuplev*cdr.nsublev;
  cedr_assert(np <= 4);
  
  FA4<      Real> q_min(q_min_r, np2, nlev, qsize, nete+1);
  FA4<const Real> q_max(q_max_r, np2, nlev, qsize, nete+1);
  //const auto& dp3d_c = d.ta->dp3d;

  for (Int ie = nets; ie <= nete; ++ie) {
    FA1<const Real> spheremp(d.spheremp[ie], np2);
    FA4<const Real> qdp_p(d.qdp_pc[ie], np2, nlev, d.qsize_d, 2);
    FA3<const Real> dp3d_c(d.dp3d_c[ie], np2, nlev, d.timelevels);
    FA3<const Real> q_c(d.q_c[ie], np2, nlev, d.qsize_d);
#ifdef COMPOSE_COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const bool nonneg = cdr.nonneg[q];
        const Int ti = cdr.cdr_over_super_levels ? q : spli*qsize + q;
        Real Qm = 0, Qm_min = 0, Qm_max = 0, Qm_prev = 0, rhom = 0, volume = 0;
        Int ie_idx;
        if (cdr.caas_in_suplev)
          ie_idx = cdr.cdr_over_super_levels ?
            cdr.nsuplev*ie + spli :
            ie;
        for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
          const auto k = k0 + sbli;
          if ( ! cdr.caas_in_suplev)
            ie_idx = cdr.cdr_over_super_levels ?
              nlevwrem*ie + k :
              cdr.nsublev*ie + sbli;
          const auto lci = cdr.ie2lci[ie_idx];
          if ( ! cdr.caas_in_suplev) {
            Qm = 0; Qm_min = 0; Qm_max = 0; Qm_prev = 0;
            rhom = 0;
            volume = 0;
          }
          if (k < nlev) {
            for (Int g = 0; g < np2; ++g) {
              volume += spheremp(g); // * dp0[k];
              const Real rhomij = dp3d_c(g,k,d.tl_np1) * spheremp(g);
              rhom += rhomij;
              Qm += q_c(g,k,q) * rhomij;
              if (nonneg) q_min(g,k,q,ie) = std::max<Real>(q_min(g,k,q,ie), 0);
              Qm_min += q_min(g,k,q,ie) * rhomij;
              Qm_max += q_max(g,k,q,ie) * rhomij;
              Qm_prev += qdp_p(g,k,q,d.n0_qdp) * spheremp(g);
            }
          }
          const bool write = ! cdr.caas_in_suplev || sbli == cdr.nsublev-1;
          if (write) {
            // For now, handle just one rhom. For feasible global problems,
            // it's used only as a weight vector in QLT, so it's fine. In fact,
            // use just the cell geometry, rather than total density, since in QLT
            // this field is used as a weight vector.
            //todo Generalize to one rhom field per level. Until then, we're not
            // getting QLT's safety benefit.
            if (ti == 0) cdr.cdr->set_rhom(lci, 0, volume);
            cdr.cdr->set_Qm(lci, ti, Qm, Qm_min, Qm_max, Qm_prev);
            if (Qm_prev < -0.5) {
              static bool first = true;
              if (first) {
                first = false;
                std::stringstream ss;
                ss << "Qm_prev < -0.5: Qm_prev = " << Qm_prev
                   << " on rank " << cdr.p->rank()
                   << " at (ie,gid,spli,k0,q,ti,sbli,lci,k,n0_qdp,tl_np1) = ("
                   << ie << "," << cdr.ie2gci[ie] << "," << spli << "," << k0 << ","
                   << q << "," << ti << "," << sbli << "," << lci << "," << k << ","
                   << d.n0_qdp << "," << d.tl_np1 << ")\n";
                ss << "Qdp(:,:,k,q,n0_qdp) = [";
                for (Int g = 0; g < np2; ++g)
                  ss << " " << qdp_p(g,k,q,d.n0_qdp);
                ss << "]\n";
                ss << "dp3d(:,:,k,tl_np1) = [";
                for (Int g = 0; g < np2; ++g)
                  ss << " " << dp3d_c(g,k,d.tl_np1);
                ss << "]\n";
                pr(ss.str());
              }
            }
          }
        }
      }
    }
  }

  run_cdr(cdr);
}

} // namespace sl
} // namespace homme
