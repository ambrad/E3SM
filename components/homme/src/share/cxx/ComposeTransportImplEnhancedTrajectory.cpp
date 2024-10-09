/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#ifdef HOMME_ENABLE_COMPOSE

#pragma message "CTET undef NDEBUG"
#undef NDEBUG
#include "/home/ac.ambradl/compy-goodies/util/dbg.hpp"

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

#include "compose_test.hpp"

#include <random>

namespace Homme {
namespace { // anon

using cti = ComposeTransportImpl;
using CSelNlev  = cti::CSNlev;
using CRelNlev  = cti::CRNlev;
using CSelNlevp = cti::CSNlevp;
using CRelNlevp = cti::CRNlevp;
using CS2elNlev = cti::CS2Nlev;
using SelNlev   = cti::SNlev;
using RelNlev   = cti::RNlev;
using SelNlevp  = cti::SNlevp;
using RelNlevp  = cti::RNlevp;
using S2elNlev  = cti::S2Nlev;
using R2elNlev  = cti::R2Nlev;
using S2elNlevp = cti::S2Nlevp;

using RelV = ExecViewUnmanaged<Real[NP][NP]>;
using CRelV = typename ViewConst<RelV>::type;

template <int N> using RelNV = ExecViewUnmanaged<Real[NP][NP][N]>;
template <int N> using CRelNV = typename ViewConst<RelNV<N>>::type;

template <int N> using RNV = ExecViewUnmanaged<Real[N]>;
template <int N> using CRNV = typename ViewConst<RNV<N>>::type;
using RNlevp = RNV<cti::num_phys_lev+1>;
using CRNlevp = CRNV<cti::num_phys_lev+1>;

using RnV = ExecViewUnmanaged<Real*>;
using CRnV = ExecViewUnmanaged<const Real*>;

KOKKOS_INLINE_FUNCTION int len (const  RnV& v) { return v.extent_int(0); }
KOKKOS_INLINE_FUNCTION int len (const CRnV& v) { return v.extent_int(0); }

using RelnV = ExecViewUnmanaged<Real***>;
using CRelnV = ExecViewUnmanaged<const Real***>;

KOKKOS_INLINE_FUNCTION
RnV getcol (const RelnV& a, const int i, const int j) {
  return Kokkos::subview(a,i,j,Kokkos::ALL);
}

KOKKOS_INLINE_FUNCTION
CRnV getcolc (const CRelnV& a, const int i, const int j) {
  return Kokkos::subview(a,i,j,Kokkos::ALL);
}

KOKKOS_INLINE_FUNCTION
void assert_eln (const CRelnV& a, const int nlev) {
  assert(a.extent_int(0) >= NP);
  assert(a.extent_int(1) >= NP);
  assert(a.extent_int(2) >= nlev);
}

// For sorted ascending x[0:n] and x in [x[0], x[n-1]] with hint xi_idx, return
// i such that x[i] <= xi <= x[i+1].
//   This function is meant for the case that x_idx is very close to the
// support. If that isn't true, then this method is inefficient; binary search
// should be used instead.
template <typename ConstRealArray>
KOKKOS_FUNCTION static
int find_support (const int n, const ConstRealArray& x, const int x_idx,
                  const Real xi) {
  assert(xi >= x[0] and xi <= x[n-1]);
  // Handle the most common case.
  if (x_idx < n-1 and xi >= x[x_idx  ] and xi <= x[x_idx+1]) return x_idx;
  if (x_idx > 0   and xi >= x[x_idx-1] and xi <= x[x_idx  ]) return x_idx-1;
  // Move on to less common ones.
  const int max_step = max(x_idx, n-1 - x_idx);
  for (int step = 1; step <= max_step; ++step) {
    if (x_idx < n-1-step and xi >= x[x_idx+step  ] and xi <= x[x_idx+step+1])
      return x_idx+step;
    if (x_idx > step     and xi >= x[x_idx-step-1] and xi <= x[x_idx-step  ])
      return x_idx-step-1;
  }
  assert(false);
  return -1;
}

// Linear interpolation core computation.
template <typename XT, typename YT>
KOKKOS_FUNCTION Real
linterp (const int n, const XT& x, const YT& y, const int x_idx, const Real xi) {
  const auto isup = find_support(n, x, x_idx, xi);
  const Real a = (xi - x[isup])/(x[isup+1] - x[isup]);
  return (1-a)*y[isup] + a*y[isup+1];
}

// Linear interpolation at the lowest team ||ism.
//   Range provides this ||ism over index 0 <= k < ni.
//   Interpolate y(x) to yi(xi).
//   x_idx_offset is added to k in the call to find_support.
//   Arrays should all have rank 1.
template <typename Range, typename XT, typename YT, typename XIT, typename YIT>
KOKKOS_FUNCTION void
linterp (const Range& range,
         const int n , const XT& x , const YT& y,
         const int ni, const XIT& xi, const YIT& yi,
         const int x_idx_offset = 0, const char* const caller = nullptr) {
#ifndef NDEBUG
  if (xi[0] < x[0] or xi[ni-1] > x[n-1]) {
    if (caller)
      printf("linterp: xi out of bounds: %s %1.15e %1.15e %1.15e %1.15e\n",
             caller, x[0], xi[0], xi[ni-1], x[n-1]);
    assert(false);
  }
#endif
  assert(range.start >= 0 );
  assert(range.end   <= ni);
  const auto f = [&] (const int k) {
    yi[k] = linterp(n, x, y, k + x_idx_offset, xi[k]);
  };
  Kokkos::parallel_for(range, f);
}

KOKKOS_FUNCTION void
eta_interp_eta (const KernelVariables& kv, const int nlev,
                const CRnV& hy_etai, const CRelnV& x, const CRnV& y,
                const RelnV& xwrk, const RnV& ywrk,
                const int ni, const CRnV& xi, const RelnV& yi) {
  const auto& xbdy = xwrk;
  const auto& ybdy = ywrk;
  assert(hy_etai.extent_int(0) >= nlev+1);
  assert_eln(x, nlev);
  assert(y.extent_int(0) >= nlev);
  assert_eln(xbdy, nlev+2);
  assert(ybdy.extent_int(0) >= nlev+2);
  assert(xi.extent_int(0) >= ni);
  assert_eln(yi, ni);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr_ni = Kokkos::ThreadVectorRange(kv.team, ni);
  const auto tvr_nlevp2 = Kokkos::ThreadVectorRange(kv.team, nlev+2);
  const auto f_y = [&] (const int k) {
    ybdy(k) = (k == 0      ? hy_etai(0) :
               k == nlev+1 ? hy_etai(nlev) :
               /**/          y(k-1));
  };
  Kokkos::parallel_for(Kokkos::TeamVectorRange(kv.team, nlev+2), f_y);
  kv.team_barrier();
  const auto f_x = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    const auto g = [&] (const int k) {
      xbdy(i,j,k) = (k == 0      ? hy_etai(0) :
                     k == nlev+1 ? hy_etai(nlev) :
                     /**/          x(i,j,k-1));
    };
    Kokkos::parallel_for(tvr_nlevp2, g);
  };
  Kokkos::parallel_for(ttr, f_x);
  kv.team_barrier();
  const auto f_linterp = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    linterp(tvr_ni,
            nlev+2, getcolc(xbdy,i,j), ybdy,
            ni,     xi,                getcol(yi,i,j),
            1, "eta_interp_eta");
  };
  Kokkos::parallel_for(ttr, f_linterp);
}

KOKKOS_FUNCTION void
eta_interp_horiz (const KernelVariables& kv, const int nlev,
                  const CRnV& hy_etai, const CRnV& x, const CRelnV& y,
                  const RnV& xwrk, const RelnV& ywrk,
                  const CRelnV& xi, const RelnV& yi) {
  const auto& xbdy = xwrk;
  const auto& ybdy = ywrk;
  assert(hy_etai.extent_int(0) >= nlev+1);
  assert(x.extent_int(0) >= nlev);
  assert_eln(y, nlev);
  assert(xbdy.extent_int(0) >= nlev+2);
  assert_eln(ybdy, nlev+2);
  assert_eln(xi, nlev);
  assert_eln(yi, nlev);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr_nlev = Kokkos::ThreadVectorRange(kv.team, nlev);
  const auto tvr_nlevp2 = Kokkos::ThreadVectorRange(kv.team, nlev+2);
  const auto f_x = [&] (const int k) {
    xbdy(k) = (k == 0      ? hy_etai(0) :
               k == nlev+1 ? hy_etai(nlev) :
               /**/          x(k-1));
  };
  Kokkos::parallel_for(Kokkos::TeamVectorRange(kv.team, nlev+2), f_x);
  kv.team_barrier();
  const auto f_y = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    const auto g = [&] (const int k) {
      // Constant interp outside of the etam support.
      ybdy(i,j,k) = (k == 0      ? y(i,j,0) :
                     k == nlev+1 ? y(i,j,nlev-1) :
                     /**/          y(i,j,k));
    };
    Kokkos::parallel_for(tvr_nlevp2, g);
  };
  Kokkos::parallel_for(ttr, f_y);
  kv.team_barrier();
  const auto f_linterp = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    linterp(tvr_nlev,
            nlev+2, xbdy,            getcolc(ybdy,i,j),
            nlev,   getcolc(xi,i,j), getcol(yi,i,j),
            1, "eta_interp_horiz");
  };
  Kokkos::parallel_for(ttr, f_linterp);
}

/* Compute level pressure thickness given eta at interfaces using the following
   approximation:
          e = A(e) + B(e)
       p(e) = A(e) p0 + B(e) ps
            = e p0 + B(e) (ps - p0)
           a= e p0 + I[Bi(eref)](e) (ps - p0).
   Then dp = diff(p).
*/
KOKKOS_FUNCTION void
eta_to_dp (const KernelVariables& kv, const int nlev,
           const Real hy_ps0, const CRnV& hy_bi, const CRnV& hy_etai,
           const CRelV& ps, const CRelnV& etai, const RelnV& wrk,
           const RelnV& dp) {
  const int nlevp = nlev + 1;
  assert(hy_bi.extent_int(0) >= nlevp);
  assert(hy_etai.extent_int(0) >= nlevp);
  assert_eln(etai, nlevp);
  assert_eln(wrk, nlevp);
  assert_eln(dp, nlev);
  const auto& bi = wrk;
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr_linterp = Kokkos::ThreadVectorRange(kv.team, nlevp);
  const auto f_linterp = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    linterp(tvr_linterp,
            nlevp, hy_etai, hy_bi,
            nlevp, getcolc(etai,i,j), getcol(bi,i,j),
            0, "eta_to_dp");
  };
  Kokkos::parallel_for(ttr, f_linterp);
  kv.team_barrier();
  const auto tvr = Kokkos::ThreadVectorRange(kv.team, nlev);
  const auto f = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    const auto dps = ps(i,j) - hy_ps0;
    const auto g = [&] (const int k) {
      dp(i,j,k) = ((etai(i,j,k+1) - etai(i,j,k))*hy_ps0 +
                   (bi(i,j,k+1) - bi(i,j,k))*dps);
    };
    Kokkos::parallel_for(tvr, g);
  };
  Kokkos::parallel_for(ttr, f);
}

/* Limit eta levels so their thicknesses, deta, are bounded below by 'low'.

   This method pulls mass only from intervals k that are larger than their
   reference value (deta(k) > deta_ref(k)), and only down to their reference
   value. This concentrates changes to intervals that, by having a lot more mass
   than usual, drive other levels negative, leaving all the other intervals
   unchanged.

   This selective use of mass provides enough to fulfill the needed mass.
   Inputs:
       m (deta): input mass
       r (deta_ref): level mass reference.
   Preconditions:
       (1) 0 <= low <= min r(i)
       (2) 1 = sum r(i) = sum(m(i)).
   Rewrite (2) as
       1 = sum_{m(i) >= r(i)} m(i) + sum_{m(i) < r(i)} m(i)
   and, thus,
       0 = sum_{m(i) >= r(i)} (m(i) - r(i)) + sum_{m(i) < r(i)} (m(i) - r(i)).
   Then
       sum_{m(i) >= r(i)} (m(i) - r(i))         (available mass to redistribute)
           = -sum_{m(i) < r(i)} (m(i) - r(i))
          >= -sum_{m(i) < lo  } (m(i) - r(i))
          >= -sum_{m(i) < lo  } (m(i) - lo  )   (mass to fill in).
   Thus, if the preconditions hold, then there's enough mass to redistribute.
 */
template <typename Range>
KOKKOS_FUNCTION void
deta_caas (const KernelVariables& kv, const Range& tvr_nlevp,
           const CRnV& deta_ref, const Real low, const RnV& w,
           const RnV& deta) {
  const auto g1 = [&] (const int k, Kokkos::Real2& sums) {
    Real wk;
    if (deta(k) < low) {
      sums.v[0] += deta(k) - low;
      deta(k) = low;
      wk = 0;
    } else {
      wk = (deta(k) > deta_ref(k) ?
            deta(k) - deta_ref(k) :
            0);
    }
    sums.v[1] += wk;
    w(k) = wk;
  };
  Kokkos::Real2 sums;
  Dispatch<>::parallel_reduce(kv.team, tvr_nlevp, g1, sums);
  const Real wneeded = sums.v[0];
  if (wneeded == 0) return;
  // Remove what is needed from the donors.
  const Real wavail = sums.v[1];
  const auto g2 = [&] (const int k) {
    deta(k) += wneeded*(w(k)/wavail);
  };
  Kokkos::parallel_for(tvr_nlevp, g2);
}

KOKKOS_FUNCTION void
deta_caas (const KernelVariables& kv, const int nlevp, const CRnV& deta_ref,
           const Real low, const RelnV& wrk, const RelnV& deta) {
  assert(deta_ref.extent_int(0) >= nlevp);
  assert_eln(wrk, nlevp);
  assert_eln(deta, nlevp);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr = Kokkos::ThreadVectorRange(kv.team, nlevp);
  const auto f = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    deta_caas(kv, tvr, deta_ref, low, getcol(wrk,i,j), getcol(deta,i,j));
  };
  Kokkos::parallel_for(ttr, f);
}

// Wrapper to deta_caas. On input and output, eta contains the midpoint eta
// values. On output, deta_caas has been applied, if necessary, to
// diff(eta(i,j,:)).
KOKKOS_FUNCTION void
limit_etam (const KernelVariables& kv, const int nlev, const CRnV& hy_etai,
            const CRnV& deta_ref, const Real deta_tol, const RelnV& wrk1,
            const RelnV& wrk2, const RelnV& eta) {
  assert(hy_etai.extent_int(0) >= nlev+1);
  assert(deta_ref.extent_int(0) >= nlev+1);
  const auto deta = wrk2;
  assert_eln(wrk1, nlev+1);
  assert_eln(deta, nlev+1);
  assert_eln(eta , nlev  );
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr = Kokkos::ThreadVectorRange(kv.team, nlev+1);
  // eta -> deta; limit deta if needed.
  const auto f1 = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    const auto  etaij = getcolc( eta,i,j);
    const auto detaij = getcol(deta,i,j);
    const auto g1 = [&] (const int k, int& nbad) {
      const auto d = (k == 0    ? etaij(0) - hy_etai(0) :
                      k == nlev ? hy_etai(nlev) - etaij(nlev-1) :
                      /**/        etaij(k) - etaij(k-1));
      const bool ok = d >= deta_tol;
      if (not ok) ++nbad;
      detaij(k) = d;
    };
    int nbad = 0;
    Dispatch<>::parallel_reduce(kv.team, tvr, g1, nbad);
    if (nbad == 0) {
      // Signal this column is fine.
      Kokkos::single(Kokkos::PerThread(kv.team), [&] () { detaij(0) = -1; });
      return;
    };
    deta_caas(kv, tvr, deta_ref, deta_tol, getcol(wrk1,i,j), detaij);
  };
  Kokkos::parallel_for(ttr, f1);
  kv.team_barrier();
  // deta -> eta; ignore columns where limiting wasn't needed.
  const auto f2 = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    const auto  etaij = getcol( eta,i,j);
    const auto detaij = getcol(deta,i,j);
    if (detaij(0) == -1) return;
    const auto g = [&] (const int k, Real& accum, const bool final) {
      assert(k != 0 || accum == 0);
      const Real d = k == 0 ? hy_etai(0) + detaij(0) : detaij(k);
      accum += d;
      if (final) etaij(k) = accum;
    };
    Dispatch<>::parallel_scan(kv.team, nlev, g);
  };
  Kokkos::parallel_for(ttr, f2);
}

} // namespace anon

// Public function.

void ComposeTransportImpl::calc_enhanced_trajectory (const int np1, const Real dt) {
}

// Testing.

namespace { // anon

template <typename V>
decltype(Kokkos::create_mirror_view(V())) cmvdc (const V& v) {
  const auto h = Kokkos::create_mirror_view(v);
  deep_copy(h, v);
  return h;
}

Kokkos::TeamPolicy<ExecSpace>
get_test_team_policy (const int nelem, const int nlev, const int ncol=NP*NP) {
  ThreadPreferences tp;
  tp.max_threads_usable = ncol;
  tp.max_vectors_usable = nlev;
  tp.prefer_threads = true;
  tp.prefer_larger_team = true;
  return Homme::get_default_team_policy<ExecSpace>(nelem, tp);
}

struct TestData {
  std::mt19937_64 engine;
  static const Real eps;

  TestData (const int seed) : engine(seed) {}

  Real urand (const Real lo = 0, const Real hi = 1) {
    std::uniform_real_distribution<Real> urb(lo, hi);
    return urb(engine);
  }
};

const Real TestData::eps = std::numeric_limits<Real>::epsilon();

int test_find_support (TestData&) {
  int ne = 0;
  const int n = 97;
  std::vector<Real> x(n);
  for (int i = 0; i < n; ++i) x[i] = -11.7 + (i*i)/n;
  const int ntest = 10000;
  for (int i = 0; i < ntest; ++i) {
    const Real xi = x[0] + (Real(i)/ntest)*(x[n-1] - x[0]);
    for (int x_idx : {0, 1, n/3, n/2, n-2, n-1}) {
      const int sup = find_support(n, x.data(), x_idx, xi);
      if (sup > n-2) ++ne;
      else if (xi < x[sup] or xi > x[sup+1]) ++ne;
    }
  }
  return ne;
}

void todev (const std::vector<Real>& h, const RnV& d) {
  assert(h.size() <= d.size());
  const auto m = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
}

void fillcols (const int n, const Real* const h, const RelnV::HostMirror& a) {
  assert(n <= a.extent_int(2));
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (size_t k = 0; k < n; ++k)
        a(i,j,k) = h[k];  
}

void todev (const int n, const Real* const h, const RelnV& d) {
  const auto m = Kokkos::create_mirror_view(d);
  fillcols(n, h, m)  ;
  Kokkos::deep_copy(d, m);
}

void todev (const std::vector<Real>& h, const RelnV& d) {
  todev(h.size(), h.data(), d);
}

void todev (const CRnV::HostMirror& h, const RelnV& d) {
  todev(h.extent_int(0), h.data(), d);
}

void tohost (const ExecView<const Real*>& d, std::vector<Real>& h) {
  assert(h.size() <= d.size());
  const auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  for (size_t i = 0; i < h.size(); ++i) h[i] = m(i);
}

void run_linterp (const std::vector<Real>& x, const std::vector<Real>& y,
                  std::vector<Real>& xi, std::vector<Real>& yi) {
  const auto n = x.size(), ni = xi.size();
  assert(y.size() == n); assert(yi.size() == ni);
  // input -> device (test different sizes >= n)
  ExecView<Real*> xv("xv", n), yv("yv", n+1), xiv("xiv", ni+2), yiv("yiv", ni+3);
  todev(x, xv);
  todev(y, yv);
  todev(xi, xiv);
  // call linterp
  const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
    const auto range = Kokkos::TeamVectorRange(team, ni);
    linterp(range, n, xv, yv, ni, xiv, yiv, 0, "unittest");
  };
  Homme::ThreadPreferences tp;
  tp.max_threads_usable = 1;
  tp.max_vectors_usable = ni;
  tp.prefer_threads = false;
  tp.prefer_larger_team = true;
  const auto policy = get_test_team_policy(1, n);
  Kokkos::parallel_for(policy, f);
  Kokkos::fence();
  // output -> host
  tohost(yiv, yi);
}

void make_random_sorted (TestData& td, const int n, const Real xlo, const Real xhi,
                         std::vector<Real>& x) {
  assert(n >= 2);
  x.resize(n);
  x[0] = xlo;
  for (int i = 1; i < n-1; ++i) x[i] = td.urand(xlo, xhi);
  x[n-1] = xhi;
  std::sort(x.begin(), x.end());
}

int test_linterp (TestData& td) {
  int nerr = 0;
  { // xi == x => yi == y.
    int ne = 0;
    const int n = 30;
    std::vector<Real> x(n), y(n), xi(n), yi(n);
    make_random_sorted(td, n, -0.1, 1.2, x);
    make_random_sorted(td, n, -3, -1, y);
    for (int i = 0; i < n; ++i) xi[i] = x[i];
    run_linterp(x, y, xi, yi);
    for (int i = 0; i < n; ++i)
      if (yi[i] != y[i])
        ++ne;
    nerr += ne;
  }
  { // Reconstruct a linear function exactly.
    int ne = 0;
    const int n = 56, ni = n-3;
    const Real xlo = -1.2, xhi = 3.1;
    const auto f = [&] (const Real x) { return -0.7 + 1.3*x; };
    std::vector<Real> x(n), y(n), xi(ni), yi(ni);
    for (int trial = 0; trial < 4; ++trial) {
      make_random_sorted(td, n, xlo, xhi, x);
      make_random_sorted(td, ni,
                         xlo + (trial == 1 or trial == 3 ?  0.1 : 0),
                         xhi + (trial == 2 or trial == 3 ? -0.5 : 0),
                         xi);
      for (int i = 0; i < n; ++i) y[i] = f(x[i]);
      run_linterp(x, y, xi, yi);
      for (int i = 0; i < ni; ++i)
        if (std::abs(yi[i] - f(xi[i])) > 100*td.eps)
          ++ne;
    }
    nerr += ne;
  }
  return nerr;
}

int make_random_deta (TestData& td, const Real deta_tol, const int nlev,
                      Real* const deta) {
  int nerr = 0;
  Real sum = 0;
  for (int k = 0; k < nlev; ++k) {
    deta[k] = td.urand(0, 1) + 0.1;
    sum += deta[k];
  }
  for (int k = 0; k < nlev; ++k) {
    deta[k] /= sum;
    if (deta[k] < deta_tol) ++nerr;
  }
  return nerr;
}

int make_random_deta (TestData& td, const Real deta_tol, const RnV& deta) {
  int nerr = 0;
  const int nlev = deta.extent_int(0);
  const auto m = Kokkos::create_mirror_view(deta);
  nerr = make_random_deta(td, deta_tol, nlev, &m(0));
  Kokkos::deep_copy(deta, m);
  return nerr;  
}

int make_random_deta (TestData& td, const Real deta_tol, const RelnV& deta) {
  int nerr = 0;
  const int nlev = deta.extent_int(2);
  const auto m = Kokkos::create_mirror_view(deta);
  for (int i = 0; i < NP; ++i)
    for (int j = 0; j < NP; ++j)
      nerr += make_random_deta(td, deta_tol, nlev, &m(i,j,0));
  Kokkos::deep_copy(deta, m);
  return nerr;
}

int test_deta_caas (TestData& td) {
  int nerr = 0;

  const int nlev = 77;
  const Real deta_tol = 10*td.eps/nlev;

  // nlev+1 deltas: deta = diff([0, etam, 1])
  ExecView<Real[nlev+1]> deta_ref("deta_ref");
  ExecView<Real[NP][NP][nlev+1]> deta("deta"), wrk("wrk");
  nerr += make_random_deta(td, deta_tol, deta_ref);

  const auto policy = get_test_team_policy(1, nlev);
  const auto run = [&] (const RelnV& deta) {
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      deta_caas(kv, nlev+1, deta_ref, deta_tol, wrk, deta);
    };
    Kokkos::parallel_for(policy, f);
    Kokkos::fence();
  };

  { // Test that if all is OK, the input is not altered.
    nerr += make_random_deta(td, deta_tol, deta);
    ExecView<Real[NP][NP][nlev+1]>::HostMirror copy("copy");
    Kokkos::deep_copy(copy, deta);
    run(deta);
    const auto m = cmvdc(deta);
    bool diff = false;
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j)
        for (int k = 0; k <= nlev; ++k)
          if (m(i,j,k) != copy(i,j,k))
            diff = true;
    if (diff) ++nerr;
  }

  { // Modify one etam and test that only adjacent intervals change beyond eps.
    // nlev midpoints
    ExecView<Real[nlev]> etam_ref("etam_ref");
    const auto her = Kokkos::create_mirror_view(etam_ref);
    const auto hder = cmvdc(deta_ref);
    {
      her(0) = hder(0);
      for (int k = 1; k < nlev; ++k)
        her(k) = her(k-1) + hder(k);
      Kokkos::deep_copy(etam_ref, her);
    }
    std::vector<Real> etam(nlev);
    const auto hde = Kokkos::create_mirror_view(deta);
    const auto get_idx = [&] (const int i, const int j) {
      const int idx = static_cast<int>(0.15*nlev);
      return std::max(1, std::min(nlev-2, idx+NP*i+j));
    };
    for (int trial = 0; trial < 2; ++trial) {
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j) {
          for (int k = 0; k < nlev; ++k) etam[k] = her(k);
          // Perturb one level.
          const int idx = get_idx(i,j);
          etam[idx] += trial == 0 ? 1.1 : -13.1;
          hde(i,j,0) = etam[0];
          for (int k = 1; k < nlev; ++k) hde(i,j,k) = etam[k] - etam[k-1];
          hde(i,j,nlev) = 1 - etam[nlev-1];
          // Make sure we have a meaningful test.
          Real minval = 1;
          for (int k = 0; k <= nlev; ++k) minval = std::min(minval, hde(i,j,k));
          if (minval >= deta_tol) ++nerr;
        }
      Kokkos::deep_copy(deta, hde);
      run(deta);
      Kokkos::deep_copy(hde, deta);
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j) {
          const int idx = get_idx(i,j);
          // Min val should be deta_tol.
          Real minval = 1;
          for (int k = 0; k <= nlev; ++k) minval = std::min(minval, hde(i,j,k));
          if (minval != deta_tol) ++nerr;
          // Sum of levels should be 1.
          Real sum = 0;
          for (int k = 0; k <= nlev; ++k) sum += hde(i,j,k);
          if (std::abs(sum - 1) > 100*td.eps) ++nerr;
          // Only two deltas should be affected.
          Real maxdiff = 0;
          for (int k = 0; k <= nlev; ++k) {
            const auto diff = std::abs(hde(i,j,k) - hder(k));
            if (k == idx || k == idx+1) {
              if (diff <= deta_tol) ++nerr;
            } else {
              maxdiff = std::max(maxdiff, diff);
            }
          }
          if (maxdiff > 100*td.eps) ++nerr;
        }
    }
  }

  { // Test generally (and highly) perturbed levels.
    const auto hde = Kokkos::create_mirror_view(deta);
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j) {
        Real sum = 0;
        for (int k = 0; k <= nlev; ++k) {
          hde(i,j,k) = td.urand(-0.5, 0.5);
          sum += hde(i,j,k);
        }
        // Make the column sum to 0.2 for safety in the next step.
        const Real colsum = 0.2;
        for (int k = 0; k <= nlev; ++k) hde(i,j,k) += (colsum - sum)/(nlev+1);
        for (int k = 0; k <= nlev; ++k) hde(i,j,k) /= colsum;
        sum = 0;
        for (int k = 0; k <= nlev; ++k) sum += hde(i,j,k);
        if (std::abs(sum - 1) > 100*td.eps) ++nerr;
      }
    Kokkos::deep_copy(deta, hde);
    run(deta);
    Kokkos::deep_copy(hde, deta);
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j) {
        Real sum = 0, minval = 1;
        for (int k = 0; k <= nlev; ++k) sum += hde(i,j,k);
        for (int k = 0; k <= nlev; ++k) minval = std::min(minval, hde(i,j,k));
        if (std::abs(sum - 1) > 1e3*td.eps) ++nerr;
        if (minval != deta_tol) ++nerr;
      }
  }
  
  return nerr;
}

struct HybridLevels {
  Real ps0;
  std::vector<Real> ai, bi, am, bm, etai, etam, deta_ref;
};

// Follow DCMIP2012 3D tracer transport specification for a, b, eta.
void fill (HybridLevels& h, const int n) {
  h.ai.resize(n+1); h.bi.resize(n+1);
  h.am.resize(n  ); h.bm.resize(n  );
  h.etai.resize(n+1); h.etam.resize(n);

  const auto Rd = PhysicalConstants::Rgas;
  const auto T0 = 300; // K
  const auto p0 = PhysicalConstants::p0;
  const auto g = PhysicalConstants::g;
  const Real ztop = 12e3; // m

  h.ps0 = p0;

  const auto calc_pressure = [&] (const Real z) {
    return p0*std::exp(-g*z/(Rd*T0));
  };

  const Real eta_top = calc_pressure(ztop)/p0;
  assert(eta_top > 0);
  for (int i = 0; i <= n; ++i) {
    const auto z = (Real(n - i)/n)*ztop;
    h.etai[i] = calc_pressure(z)/p0;
    h.bi[i] = i == 0 ? 0 : (h.etai[i] - eta_top)/(1 - eta_top);
    h.ai[i] = h.etai[i] - h.bi[i];
    assert(i == 0 || h.etai[i] > h.etai[i-1]);
  }
  assert(h.bi  [0] == 0); // Real(n - i)/n is exactly 1, so exact = holds
  assert(h.bi  [n] == 1); // exp(0) is exactly 0, so exact = holds
  assert(h.etai[n] == 1); // same

  const auto tomid = [&] (const std::vector<Real>& in, std::vector<Real>& mi) {
    for (int i = 0; i < n; ++i) mi[i] = (in[i] + in[i+1])/2;
  };
  tomid(h.ai, h.am);
  tomid(h.bi, h.bm);
  tomid(h.etai, h.etam);

  h.deta_ref.resize(n+1);
  h.deta_ref[0] = h.etam[0] - h.etai[0];
  for (int i = 1; i < n; ++i) h.deta_ref[i] = h.etam[i] - h.etam[i-1];
  h.deta_ref[n] = h.etai[n] - h.etam[n-1];
}

int test_limit_etam (TestData& td) {
  int nerr = 0;

  const int nlev = 92;
  const Real deta_tol = 1e5*td.eps/nlev;

  ExecView<Real[nlev+1]> hy_etai("hy_etai"), deta_ref("deta_ref");
  ExecView<Real[NP][NP][nlev+1]> wrk1("wrk1"), wrk2("wrk2");
  ExecView<Real[NP][NP][nlev]> etam("etam");

  HybridLevels h;
  fill(h, nlev);
  todev(h.etai, hy_etai);
  todev(h.deta_ref, deta_ref);

  const auto he = Kokkos::create_mirror_view(etam);

  const auto policy = get_test_team_policy(1, nlev);
  const auto run = [&] () {
    Kokkos::deep_copy(etam, he);
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      limit_etam(kv, nlev, hy_etai, deta_ref, deta_tol, wrk1, wrk2, etam);
    };
    Kokkos::parallel_for(policy, f);
    Kokkos::fence();
    Kokkos::deep_copy(he, etam);
  };

  fillcols(h.etam.size(), h.etam.data(), he);
  // Col 0 should be untouched. Cols 1 and 2 should have very specific changes.
  const int col1_idx = static_cast<int>(0.25*nlev);
  he(0,1,col1_idx) += 0.3;
  const int col2_idx = static_cast<int>(0.8*nlev);
  he(0,2,col2_idx) -= 5.3;
  // The rest of the columns get wild changes.
  for (int idx = 3; idx < NP*NP; ++idx) {
    const int i = idx / NP, j = idx % NP;
    for (int k = 0; k < nlev; ++k)
      he(i,j,k) += td.urand(-1, 1)*(h.etai[k+1] - h.etai[k]);
  }
  run();
  bool ok = true;
  for (int k = 0; k < nlev; ++k)
    if (he(0,0,k) != h.etam[k]) ok = false;
  for (int k = 0; k < nlev; ++k) {
    if (k == col1_idx) continue;
    if (std::abs(he(0,1,k) - h.etam[k]) > 100*td.eps) ok = false;
  }
  for (int k = 0; k < nlev; ++k) {
    if (k == col2_idx) continue;
    if (std::abs(he(0,2,k) - h.etam[k]) > 100*td.eps) ok = false;
  }
  Real mingap = 1;
  for (int i = 0; i < NP; ++i)
    for (int j = 0; j < NP; ++j) {
      mingap = std::min(mingap, he(i,j,0) - h.etai[0]);
      for (int k = 1; k < nlev; ++k)
        mingap = std::min(mingap, he(i,j,k) - he(i,j,k-1));
      mingap = std::min(mingap, h.etai[nlev] - he(i,j,nlev-1));
    }
  // Test minimum level delta, with room for numerical error.
  if (mingap < 0.8*deta_tol) ok = false;
  if (not ok) ++nerr;
  
  return nerr;
}

int test_eta_interp (TestData& td) {
  int nerr = 0;

  const int nlev = 56;
  HybridLevels h;
  fill(h, nlev);

  ExecView<Real[nlev+1]> hy_etai("hy_etai");
  ExecView<Real[NP][NP][nlev  ]> x("x"), y("y");
  ExecView<Real[NP][NP][nlev+1]> xi("xi"), yi("yi");
  ExecView<Real[NP][NP][nlev+2]> xwrk("xwrk"), ywrk("ywrk");

  todev(h.etai, hy_etai);

  const auto xh  = Kokkos::create_mirror_view(x );
  const auto yh  = Kokkos::create_mirror_view(y );
  const auto xih = Kokkos::create_mirror_view(xi);
  const auto yih = Kokkos::create_mirror_view(xi);

  const auto policy = get_test_team_policy(1, nlev);
  const auto run_eta = [&] (const int ni) {
    Kokkos::deep_copy(x, xh); Kokkos::deep_copy(y, yh);
    Kokkos::deep_copy(xi, xih);
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      eta_interp_eta(kv, nlev, hy_etai,
                     x, getcolc(y,0,0),
                     xwrk, getcol(ywrk,0,0),
                     ni, getcolc(xi,0,0), yi);
    };
    Kokkos::parallel_for(policy, f);
    Kokkos::fence();
    Kokkos::deep_copy(yih, yi);
  };
  const auto run_horiz = [&] () {
    Kokkos::deep_copy(x, xh); Kokkos::deep_copy(y, yh);
    Kokkos::deep_copy(xi, xih);
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      eta_interp_horiz(kv, nlev, hy_etai,
                       getcolc(x,0,0), y,
                       getcol(xwrk,0,0), ywrk,
                       xi, yi);
    };
    Kokkos::parallel_for(policy, f);
    Kokkos::fence();
    Kokkos::deep_copy(yih, yi);
  };

  std::vector<Real> v;
  const Real d = 1e-6, vlo = h.etai[0]+d, vhi = h.etai[nlev]-d;
  for (const int ni : {int(0.7*nlev), nlev-1, nlev, nlev+1}) {
    make_random_sorted(td, nlev, vlo, vhi, v);
    fillcols(nlev, v.data(), xh);
    fillcols(nlev, v.data(), yh);
    make_random_sorted(td, ni, vlo, vhi, v);
    fillcols(ni, v.data(), xih);
    run_eta(ni);
    bool ok = true;
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j)
        for (int k = 0; k < ni; ++k)
          if (std::abs(yih(i,j,k) - xih(i,j,k)) > 100*td.eps) ok = false;
    if (not ok) ++nerr;
  }
  
  return nerr;
}

int test_eta_to_dp (TestData& td) {
  int nerr = 0;

  const int nlev = 88;
  HybridLevels h;
  fill(h, nlev);

  ExecView<Real[nlev+1]> hy_bi("hy_bi"), hy_etai("hy_etai");
  ExecView<Real[NP][NP][nlev+1]> etai("etai"), wrk("wrk");
  ExecView<Real[NP][NP][nlev]> dp("dp");
  ExecView<Real[NP][NP]> ps("ps");
  const Real hy_ps0 = h.ps0;

  todev(h.bi, hy_bi);
  todev(h.etai, hy_etai);

  const auto psm = Kokkos::create_mirror_view(ps);
  HostView<Real[NP][NP][nlev]> dp1("dp1");
  Real dp1_max = 0;
  for (int i = 0; i < NP; ++i)
    for (int j = 0; j < NP; ++j)
      psm(i,j) = (1 + 0.1*td.urand(-1, 1))*h.ps0;
  Kokkos::deep_copy(ps, psm);

  const auto policy = get_test_team_policy(1, nlev);
  const auto run = [&] () {
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      eta_to_dp(kv, nlev, hy_ps0, hy_bi, hy_etai, ps, etai, wrk, dp);
    };
    Kokkos::parallel_for(policy, f);
    Kokkos::fence();
  };

  { // Test that for etai_ref we get the same as the usual formula.
    todev(h.etai, etai);
    const auto psm = Kokkos::create_mirror_view(ps);
    HostView<Real[NP][NP][nlev]> dp1("dp1");
    Real dp1_max = 0;
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j)
        for (int k = 0; k < nlev; ++k) {
          dp1(i,j,k) = ((h.ai[k+1] - h.ai[k])*h.ps0 +
                        (h.bi[k+1] - h.bi[k])*psm(i,j));
          dp1_max = std::max(dp1_max, std::abs(dp1(i,j,k)));
        }
    run();
    const auto dph = cmvdc(dp);
    Real err_max = 0;
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j)
        for (int k = 0; k < nlev; ++k)
          err_max = std::max(err_max, std::abs(dph(i,j,k) - dp1(i,j,k)));
    if (err_max > 100*td.eps*dp1_max) ++nerr;
  }

  { // Test that sum(dp) = ps for random input etai.
    std::vector<Real> etai_r;
    make_random_sorted(td, nlev+1, h.etai[0], h.etai[nlev], etai_r);
    todev(etai_r, etai);
    run();
    const auto dph1 = cmvdc(dp);
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j) {
        Real ps = h.ai[0]*h.ps0;
        for (int k = 0; k < nlev; ++k)
          ps += dph1(i,j,k);
        if (std::abs(ps - psm(i,j)) > 100*td.eps*psm(i,j)) ++nerr;
      }    
    // Test that values on input don't affect solution.
    Kokkos::deep_copy(wrk, 0);
    Kokkos::deep_copy(dp, 0);
    run();
    const auto dph2 = cmvdc(dp);
    bool alleq = true;
    for (int i = 0; i < NP; ++i)
      for (int j = 0; j < NP; ++j)
        for (int k = 0; k < nlev; ++k)
          if (dph2(i,j,k) != dph1(i,j,k))
            alleq = false;
    if (not alleq) ++nerr;
  }

  return nerr;
}

int test_init_velocity_record (TestData& td) {
  int nerr = 0;
  pr("test_init_velocity_record");
  return nerr;
}

} // namespace anon

#define comunittest(f) do {                     \
    ne = f(td);                                 \
    if (ne) printf(#f " ne %d\n", ne);          \
    nerr += ne;                                 \
  } while (0)

int ComposeTransportImpl::run_enhanced_trajectory_unit_tests () {
  int nerr = 0, ne;
  TestData td(1);
  comunittest(test_find_support);
  comunittest(test_linterp);
  comunittest(test_eta_interp);
  comunittest(test_eta_to_dp);
  comunittest(test_deta_caas);
  comunittest(test_limit_etam);
  comunittest(test_init_velocity_record);
  return nerr;
}

#undef comunittest

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
