#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

template <Int np, typename MT>
void calc_q_extrema (IslMpi<MT>& cm, const Int& nets, const Int& nete) {
#ifdef COMPOSE_PORT
  const auto qdp = cm.tracer_arrays.qdp;
  const auto dp = cm.tracer_arrays.dp;
  const auto q = cm.tracer_arrays.q;
  for (Int tci = nets; tci <= nete; ++tci) {
    auto& ed = cm.ed_d(tci);
    for (Int iq = 0; iq < cm.qsize; ++iq)
      for (Int lev = 0; lev < cm.nlev; ++lev) {
        Real q_min_s, q_max_s;
        q(tci,iq,0,lev) = qdp(tci,iq,0,lev)/dp(tci,0,lev);
        q_min_s = q_max_s = q(tci,iq,0,lev);
        for (Int k = 1; k < np*np; ++k) {
          q(tci,iq,k,lev) = qdp(tci,iq,k,lev)/dp(tci,k,lev);
          q_min_s = std::min(q_min_s, q(tci,iq,k,lev));
          q_max_s = std::max(q_max_s, q(tci,iq,k,lev));
        }
        ed.q_extrema(iq,lev,0) = q_min_s;
        ed.q_extrema(iq,lev,1) = q_max_s;
      }
  }
#else
  for (Int tci = nets; tci <= nete; ++tci) {
    auto& ed = cm.ed_d(tci);
    const FA2<const Real> dp(ed.dp, cm.np2, cm.nlev);
    const FA3<const Real> qdp(ed.qdp, cm.np2, cm.nlev, cm.qsize);
    const FA3<Real> q(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (Int iq = 0; iq < cm.qsize; ++iq)
      for (Int lev = 0; lev < cm.nlev; ++lev) {
        const Real* const dp0 = &dp(0,lev);
        const Real* const qdp0 = &qdp(0,lev,iq);
        Real* const q0 = &q(0,lev,iq);
        Real q_min_s, q_max_s;
        q0[0] = qdp0[0] / dp0[0];
        q_min_s = q_max_s = q0[0];
        for (Int k = 1; k < np*np; ++k) {
          q0[k] = qdp0[k] / dp0[k];
          q_min_s = std::min(q_min_s, q0[k]);
          q_max_s = std::max(q_max_s, q0[k]);
        }
        ed.q_extrema(iq,lev,0) = q_min_s;
        ed.q_extrema(iq,lev,1) = q_max_s;
      }
  }
#endif
}

template <typename MT>
void calc_q_extrema (IslMpi<MT>& cm, const Int& nets, const Int& nete) {
  switch (cm.np) {
  case 4: calc_q_extrema<4>(cm, nets, nete); break;
  default: slmm_throw_if(true, "np " << cm.np << "not supported");
  }
}

template void calc_q_extrema(IslMpi<slmm::MachineTraits>& cm, const Int& nets,
                             const Int& nete);

} // namespace islmpi
} // namespace homme