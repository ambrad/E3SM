#include "compose_cedr_cdr.hpp"
#include "compose_cedr_sl.hpp"

namespace homme {
namespace sl {

template <typename MT, typename Ie2gci, typename Qdp, typename Dp3d>
KOKKOS_FUNCTION ko::EnableIfNotOnGpu<MT> warn_on_Qm_prev_negative (
  Real Qm_prev, const cedr::mpi::Parallel::Ptr& p, Int ie, const Ie2gci& ie2gci,
  Int np2, Int spli, Int k0, Int q, Int ti, Int sbli, Int lci, Int k, Int n0_qdp, Int np1,
  const Qdp& qdp_p, const Dp3d& dp3d_c)
{
  static bool first = true;
  if (first) {
    first = false;
    std::stringstream ss;
    ss << "Qm_prev < -0.5: Qm_prev = " << Qm_prev
       << " on rank " << p->rank()
       << " at (ie,gid,spli,k0,q,ti,sbli,lci,k,n0_qdp,tl_np1) = ("
       << ie << "," << ie2gci[ie] << "," << spli << "," << k0 << ","
       << q << "," << ti << "," << sbli << "," << lci << "," << k << ","
       << n0_qdp << "," << np1 << ")\n";
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

template <typename MT, typename Ie2gci, typename Qdp, typename Dp3d>
KOKKOS_FUNCTION ko::EnableIfOnGpu<MT> warn_on_Qm_prev_negative (
  Real Qm_prev, const cedr::mpi::Parallel::Ptr& p, Int ie, const Ie2gci& ie2gci,
  Int np2, Int spli, Int k0, Int q, Int ti, Int sbli, Int lci, Int k, Int n0_qdp, Int np1,
  const Qdp& qdp_p, const Dp3d& dp3d_c)
{}

template <typename MT>
static void run_cdr (CDR<MT>& q) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
  q.cdr->run();
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

template <typename MT>
void run_global (CDR<MT>& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                 const Int nets, const Int nete) {
  const auto& ta = *d.ta;
  const Int np = ta.np, np2 = np*np, nlev = ta.nlev, qsize = ta.qsize,
    nlevwrem = cdr.nsuplev*cdr.nsublev;
  cedr_assert(np <= 4);
  
#ifdef COMPOSE_PORT_DEV_VIEWS
  const auto q_min = ta.q_min;
  const auto q_max = ta.q_max;
#else
  const QExtremaH<MT>      q_min(q_min_r, ta.nelemd, ta.qsize, ta.nlev, ta.np2);
  const QExtremaHConst<MT> q_max(q_max_r, ta.nelemd, ta.qsize, ta.nlev, ta.np2);
#endif
  const auto np1 = ta.np1;
  const auto n0_qdp = ta.n0_qdp;
  const auto spheremp = ta.spheremp;
  const auto dp3d_c = ta.dp3d;
  const auto qdp_p = ta.qdp;
  const auto q_c = ta.q;

  const Int nsublev = cdr.nsublev;
  const Int nsuplev = cdr.nsuplev;
  const auto& nonnegs = cdr.nonneg;
  const auto cdr_over_super_levels = cdr.cdr_over_super_levels;
  const auto caas_in_suplev = cdr.caas_in_suplev;
  const auto& ie2lci = cdr.ie2lci;
  const auto& ie2gci = cdr.ie2gci;
  const auto& p = cdr.p;
  const auto& cedr_cdr = cdr.cdr;
  const auto f = KOKKOS_LAMBDA (const Int& idx) {
    const Int ie = nets + idx/(nsuplev*qsize);
    const Int spli = (idx / qsize) % nsuplev;
    const Int q = idx % qsize;
    const Int k0 = nsublev*spli;
    const bool nonneg = nonnegs[q];
    const Int ti = cdr_over_super_levels ? q : spli*qsize + q;
    Real Qm = 0, Qm_min = 0, Qm_max = 0, Qm_prev = 0, rhom = 0, volume = 0;
    Int ie_idx;
    if (caas_in_suplev)
      ie_idx = cdr_over_super_levels ?
        nsuplev*ie + spli :
        ie;
    for (Int sbli = 0; sbli < nsublev; ++sbli) {
      const auto k = k0 + sbli;
      if ( ! caas_in_suplev)
        ie_idx = cdr_over_super_levels ?
          nlevwrem*ie + k :
          nsublev*ie + sbli;
      const auto lci = ie2lci[ie_idx];
      if ( ! caas_in_suplev) {
        Qm = 0; Qm_min = 0; Qm_max = 0; Qm_prev = 0;
        rhom = 0;
        volume = 0;
      }
      if (k < nlev) {
        for (Int g = 0; g < np2; ++g) {
          volume += spheremp(ie,g); // * dp0[k];
          const Real rhomij = dp3d_c(ie,np1,g,k) * spheremp(ie,g);
          rhom += rhomij;
          Qm += q_c(ie,q,g,k) * rhomij;
          if (nonneg) q_min(ie,q,k,g) = std::max<Real>(q_min(ie,q,k,g), 0);
          Qm_min += q_min(ie,q,k,g) * rhomij;
          Qm_max += q_max(ie,q,k,g) * rhomij;
          Qm_prev += qdp_p(ie,n0_qdp,q,g,k) * spheremp(ie,g);
        }
      }
      const bool write = ! caas_in_suplev || sbli == nsublev-1;
      if (write) {
        // For now, handle just one rhom. For feasible global problems,
        // it's used only as a weight vector in QLT, so it's fine. In fact,
        // use just the cell geometry, rather than total density, since in QLT
        // this field is used as a weight vector.
        //todo Generalize to one rhom field per level. Until then, we're not
        // getting QLT's safety benefit.
        if (ti == 0) cedr_cdr->set_rhom(lci, 0, volume);
        cedr_cdr->set_Qm(lci, ti, Qm, Qm_min, Qm_max, Qm_prev);
        if (Qm_prev < -0.5)
          warn_on_Qm_prev_negative<MT>(Qm_prev, p, ie, ie2gci, np2, spli, k0, q,
                                       ti, sbli, lci, k, n0_qdp, np1, qdp_p, dp3d_c);
      }
    }    
  };
  ko::parallel_for(ko::RangePolicy<typename MT::DES>(0, (nete - nets + 1)*nsuplev*qsize), f);

  run_cdr(cdr);
}

template void
run_global(CDR<ko::MachineTraits>& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
           const Int nets, const Int nete);

} // namespace sl
} // namespace homme
