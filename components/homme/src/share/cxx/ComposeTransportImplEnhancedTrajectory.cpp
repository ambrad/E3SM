/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#ifdef HOMME_ENABLE_COMPOSE

#pragma message "CTET undef NDEBUG"
#undef NDEBUG

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

#include "compose_test.hpp"

#include <random>

namespace Homme {
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
template <typename ConstRealArray>
KOKKOS_FUNCTION static Real
linterp (const int n, const ConstRealArray& x, const ConstRealArray& y,
         const int x_idx, const Real xi) {
  const auto isup = find_support(n, x, x_idx, xi);
  const Real a = (xi - x[isup])/(x[isup+1] - x[isup]);
  return (1-a)*y[isup] + a*y[isup+1];
}

// Linear interpolation at the lowest team ||ism.
//   Range provides this ||ism over index 0 <= k < ni.
//   Interpolate y(x) to yi(xi).
//   x_idx_offset is added to k in the call to find_support.
template <typename Range, typename ConstRealArray, typename RealArray>
KOKKOS_FUNCTION static void
linterp (const Range& range,
         const int n , const ConstRealArray& x , const ConstRealArray& y,
         const int ni, const ConstRealArray& xi, const RealArray& yi,
         const int x_idx_offset = 0, const char* const caller = nullptr) {
#ifndef NDEBUG
  if (xi[0] < x[0] or xi[ni-1] > x[n-1]) {
    if (caller)
      printf("linterp: xi out of bounds: %s\n", caller);
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

template <int N>
KOKKOS_FUNCTION static void
linterp (const KernelVariables& kv,
         const int n, const CRelNV<N>& x, const CRelNV<N>& y,
         const int ni, const CRelNV<N>& xi, const RelNV<N>& yi,
         const int x_idx_offset = 0,
         const char* const caller = nullptr) {
  assert(n <= N);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr = Kokkos::ThreadVectorRange(kv.team, ni);
  const auto f = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    linterp(tvr,
            n , Homme::subview(x ,i,j), Homme::subview(y ,i,j),
            ni, Homme::subview(xi,i,j), Homme::subview(yi,i,j),
            x_idx_offset, caller);
  };
  Kokkos::parallel_for(ttr, f);
}

/*
  Compute level pressure thickness given eta at interfaces using the following
  approximation:
         e = A(e) + B(e)
      p(e) = A(e) p0 + B(e) ps
           = e p0 + B(e) (ps - p0)
          a= e p0 + I[Bi(eref)](e) (ps - p0).
  Then dp = diff(p).
*/
template <int Nlev>
KOKKOS_FUNCTION static void
eta_to_dp (const KernelVariables& kv,
           const Real hy_ps0, const CRNV<Nlev+1>& hy_bi, const CRNV<Nlev+1>& hy_etai,
           const CRelV& ps, const CRelNV<Nlev+1>& etai, const RelNV<Nlev+1>& wrk,
           const RelNV<Nlev>& dp) {
  const int nlev = cti::num_phys_lev, nlevp = nlev + 1;
  const auto& bi = wrk;
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr_linterp = Kokkos::ThreadVectorRange(kv.team, nlevp);
  const auto f_linterp = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    linterp(tvr_linterp,
            nlevp, hy_etai, hy_bi,
            nlevp, Homme::subview(etai,i,j), Homme::subview(bi,i,j),
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

// Public function.

void ComposeTransportImpl::calc_enhanced_trajectory (const int np1, const Real dt) {
}

// Testing.

namespace {
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
} // namespace

static int test_find_support (TestData&) {
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

static void
run_linterp (const std::vector<Real>& x, const std::vector<Real>& y,
             std::vector<Real>& xi, std::vector<Real>& yi) {
  const auto n = x.size(), ni = xi.size();
  assert(y.size() == n); assert(yi.size() == ni);
  // input -> device (test different sizes >= n)
  ExecView<Real*> xv("xv", n), yv("yv", n+1), xiv("xiv", ni+2), yiv("yiv", ni+3);
  const auto xm = Kokkos::create_mirror_view(xv);
  const auto ym = Kokkos::create_mirror_view(yv);
  const auto xim = Kokkos::create_mirror_view(xiv);
  const auto yim = Kokkos::create_mirror_view(yiv);
  for (size_t i = 0; i < n; ++i) xm(i) = x[i];
  for (size_t i = 0; i < n; ++i) ym(i) = y[i];
  for (size_t i = 0; i < ni; ++i) xim(i) = xi[i];
  Kokkos::deep_copy(xv, xm);
  Kokkos::deep_copy(yv, ym);
  Kokkos::deep_copy(xiv, xim);
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
  const auto policy = Homme::get_default_team_policy<ExecSpace>(1, tp);
  Kokkos::parallel_for(policy, f);
  Kokkos::fence();
  // output -> host
  Kokkos::deep_copy(yim, yiv);
  for (size_t i = 0; i < ni; ++i) yi[i] = yim(i);
}

static void
make_random_sorted (TestData& td, const int n, const Real xlo, const Real xhi,
                    std::vector<Real>& x) {
  assert(n >= 2);
  x.resize(n);
  x[0] = xlo;
  for (int i = 1; i < n-1; ++i) x[i] = td.urand(xlo, xhi);
  x[n-1] = xhi;
  std::sort(x.begin(), x.end());
}

static int test_linterp (TestData& td) {
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

struct HybridLevels {
  std::vector<Real> ai, bi, am, bm, etai, etam;
};

// Follow DCMIP2012 3D tracer transport specification for a, b, eta.
static void fill (HybridLevels& h, const int n) {
  h.ai.resize(n+1); h.bi.resize(n+1);
  h.am.resize(n  ); h.bm.resize(n  );
  h.etai.resize(n+1); h.etam.resize(n);

  const auto Rd = PhysicalConstants::Rgas;
  const auto T0 = 300; // K
  const auto p0 = PhysicalConstants::p0;
  const auto g = PhysicalConstants::g;
  const Real ztop = 12e3; // m

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
}

static int test_eta_to_dp (TestData& td) {
  int nerr = 0;
  const int nlev = 88;
  HybridLevels h;
  fill(h, nlev);
  return nerr;
}

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
  comunittest(test_eta_to_dp);
  return nerr;
}

#undef comunittest

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
