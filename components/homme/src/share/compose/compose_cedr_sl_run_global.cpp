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
  
  const auto& ta = *d.ta;
#ifdef COMPOSE_PORT_DEV_VIEWS
  const auto& q_min = ta.q_min;
  const auto& q_max = ta.q_max;
#else
  const QExtremaH<ko::MachineTraits> q_min(q_min_r, ta.nelemd, ta.qsize, ta.nlev, ta.np2);
  const QExtremaHConst<ko::MachineTraits> q_max(q_max_r, ta.nelemd, ta.qsize, ta.nlev, ta.np2);
#endif
  const auto& dp3d_c = ta.pdp3d;
  const auto np1 = ta.np1;
  const auto& qdp_p = ta.pqdp;
  const auto n0_qdp = ta.n0_qdp;
  const auto& q_c = ta.pq;

  for (Int ie = nets; ie <= nete; ++ie) {
    FA1<const Real> spheremp(d.spheremp[ie], np2);
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
              const Real rhomij = dp3d_c(ie,np1,g,k) * spheremp(g);
              rhom += rhomij;
              Qm += q_c(ie,q,g,k) * rhomij;
              if (nonneg) q_min(ie,q,k,g) = std::max<Real>(q_min(ie,q,k,g), 0);
              Qm_min += q_min(ie,q,k,g) * rhomij;
              Qm_max += q_max(ie,q,k,g) * rhomij;
              Qm_prev += qdp_p(ie,n0_qdp,q,g,k) * spheremp(g);
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
                  ss << " " << qdp_p(ie,n0_qdp,q,g,k);
                ss << "]\n";
                ss << "dp3d(:,:,k,tl_np1) = [";
                for (Int g = 0; g < np2; ++g)
                  ss << " " << dp3d_c(ie,np1,g,k);
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
