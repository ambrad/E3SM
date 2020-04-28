#ifndef INCLUDE_COMPOSE_CEDR_SL_HPP
#define INCLUDE_COMPOSE_CEDR_SL_HPP

#include "compose_homme.hpp"

namespace homme {
namespace sl {

using homme::Int;
using homme::Real;
using homme::FA2;
using homme::FA4;
using homme::FA5;

// Following are naming conventions in element_state and sl_advection:
//     elem(ie)%state%Q(:,:,k,q) is tracer mixing ratio.
//     elem(ie)%state%dp3d(:,:,k,tl%np1) is essentially total density.
//     elem(ie)%state%Qdp(:,:,k,q,n0_qdp) is Q*dp3d.
//     rho(:,:,k,ie) is spheremp*dp3d, essentially total mass at a GLL point.
//     Hence Q*rho = Q*spheremp*dp3d is tracer mass at a GLL point.
// We need to get pointers to some of these; elem can't be given the bind(C)
// attribute, so we can't take the elem array directly. We get these quantities
// at a mix of previous and current time steps.
//   In the code that follows, _p is previous and _c is current time step. Q is
// renamed to q, and Q is tracer mass at a GLL point.
struct Data {
  typedef std::shared_ptr<Data> Ptr;
  const Int np, nlev, qsize, qsize_d, timelevels;
  Int n0_qdp, n1_qdp, tl_np1;
  std::vector<const Real*> spheremp, dp3d_c;
  std::vector<Real*> q_c, qdp_pc;
  const Real* dp0;

  struct Check {
    Kokkos::View<Real**, Kokkos::Serial>
      mass_p, mass_c, mass_lo, mass_hi,
      q_lo, q_hi, q_min_l, q_max_l, qd_lo, qd_hi;
    Check (const Int nlev, const Int qsize)
      : mass_p("mass_p", nlev, qsize), mass_c("mass_c", nlev, qsize),
        mass_lo("mass_lo", nlev, qsize), mass_hi("mass_hi", nlev, qsize),
        q_lo("q_lo", nlev, qsize), q_hi("q_hi", nlev, qsize),
        q_min_l("q_min_l", nlev, qsize), q_max_l("q_max_l", nlev, qsize),
        qd_lo("qd_lo", nlev, qsize), qd_hi("qd_hi", nlev, qsize)
    {}
  };
  std::shared_ptr<Check> check;

  Data (Int lcl_ncell, Int np_, Int nlev_, Int qsize_, Int qsize_d_, Int timelevels_)
    : np(np_), nlev(nlev_), qsize(qsize_), qsize_d(qsize_d_), timelevels(timelevels_),
      spheremp(lcl_ncell, nullptr), dp3d_c(lcl_ncell, nullptr), q_c(lcl_ncell, nullptr),
      qdp_pc(lcl_ncell, nullptr)
  {}
};

void accum_values(const Int ie, const Int k, const Int q, const Int tl_np1,
                  const Int n0_qdp, const Int np, const bool nonneg,
                  const FA2<const Real>& spheremp, const FA4<const Real>& dp3d_c,
                  const FA5<Real>& q_min, const FA5<const Real>& q_max,
                  const FA5<const Real>& qdp_p, const FA4<const Real>& q_c,
                  Real& volume, Real& rhom, Real& Qm, Real& Qm_prev,
                  Real& Qm_min, Real& Qm_max);

void run_global(CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete);

void run_local(CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
               const Int nets, const Int nete, const bool scalar_bounds,
               const Int limiter_option);

void check(CDR& cdr, Data& d, const Real* q_min_r, const Real* q_max_r,
           const Int nets, const Int nete);

} // namespace sl
} // namespace homme

#endif
