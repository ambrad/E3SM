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
CRnV getcol (const CRelnV& a, const int i, const int j) {
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
template <typename ConstRealArray>
KOKKOS_FUNCTION Real
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
KOKKOS_FUNCTION void
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

/*
  Compute level pressure thickness given eta at interfaces using the following
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
            nlevp, getcol(etai,i,j), getcol(bi,i,j),
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
deta_caas (const KernelVariables& kv, const Range& tvr,
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
  Dispatch<>::parallel_reduce(kv.team, tvr, g1, sums);
  const Real wneeded = sums.v[0];
  if (wneeded == 0) return;
  // Remove what is needed from the donors.
  const Real wavail = sums.v[1];
  const auto g2 = [&] (const int k) {
    deta(k) += wneeded*(w(k)/wavail);
  };
  Kokkos::parallel_for(tvr, g2);
}

KOKKOS_FUNCTION void
deta_caas (const KernelVariables& kv, const int nlev, const CRnV& deta_ref,
           const Real low, const RelnV& wrk, const RelnV& deta) {
  assert(deta_ref.extent_int(0) >= nlev);
  assert_eln(wrk, nlev);
  assert_eln(deta, nlev);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr = Kokkos::ThreadVectorRange(kv.team, nlev);
  const auto f = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    deta_caas(kv, tvr, deta_ref, low, getcol(wrk,i,j), getcol(deta,i,j));
  };
  Kokkos::parallel_for(ttr, f);
}

template <typename Range>
KOKKOS_FUNCTION void
limit_deta (const KernelVariables& kv, const Range& tvr_nlev, const Range& tvr_nlevp,
            const CRnV& hy_etai, const CRnV& deta_ref, const Real deta_tol,
            const RnV& eta) {
  
}

KOKKOS_FUNCTION void
limit_deta (const KernelVariables& kv, const int nlev, const CRnV& hy_etai,
            const CRnV& deta_ref, const Real deta_tol, const RelnV& eta) {
  assert(hy_etai.extent_int(0) >= nlev+1);
  assert(deta_ref.extent_int(0) >= nlev+1);
  assert_eln(eta, nlev);
  const auto ttr = Kokkos::TeamThreadRange(kv.team, NP*NP);
  const auto tvr_nlev  = Kokkos::ThreadVectorRange(kv.team, nlev);
  const auto tvr_nlevp = Kokkos::ThreadVectorRange(kv.team, nlev+1);
  const auto f = [&] (const int idx) {
    const int i = idx / NP, j = idx % NP;
    limit_deta(kv, tvr_nlev, tvr_nlevp, hy_etai, deta_ref, deta_tol,
               getcol(eta,i,j));
  };
  Kokkos::parallel_for(ttr, f);
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

void todev (const int n, const Real* const h, const RelnV& d) {
  assert(n <= d.extent_int(2));
  const int nlev = static_cast<int>(n);
  const auto m = Kokkos::create_mirror_view(d);
  for (int i = 0; i < m.extent_int(0); ++i)
    for (int j = 0; j < m.extent_int(1); ++j)
      for (size_t k = 0; k < nlev; ++k)
        m(i,j,k) = h[k];
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
  const auto policy = Homme::get_default_team_policy<ExecSpace>(1, tp);
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

int
make_random_deta (TestData& td, const Real deta_tol, const RelnV& deta) {
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

  const auto policy = Homme::get_default_team_policy<ExecSpace>(1);
  const auto run = [&] (const RelnV& deta) {
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      deta_caas(kv, nlev+1, deta_ref, deta_tol, wrk, deta);
    };
    Kokkos::parallel_for(policy, f);
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
  std::vector<Real> ai, bi, am, bm, etai, etam;
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
}

int test_limit_deta (TestData& td) {
  int nerr = 0;
  pr("test_limit_deta");
  return nerr;
}

int test_eta_interp_eta (TestData& td) {
  int nerr = 0;
  pr("test_eta_interp_eta");
  return nerr;
}

int test_eta_interp_horiz (TestData& td) {
  int nerr = 0;
  pr("test_eta_interp_horiz");
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

  const auto policy = Homme::get_default_team_policy<ExecSpace>(1);
  const auto run = [&] () {
    const auto f = KOKKOS_LAMBDA(const cti::MT& team) {
      KernelVariables kv(team);
      eta_to_dp(kv, nlev, hy_ps0, hy_bi, hy_etai, ps, etai, wrk, dp);
    };
    Kokkos::parallel_for(policy, f);
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
  comunittest(test_eta_interp_eta);
  comunittest(test_eta_interp_horiz);
  comunittest(test_eta_to_dp);
  comunittest(test_deta_caas);
  comunittest(test_limit_deta);
  comunittest(test_init_velocity_record);
  return nerr;
}

#undef comunittest

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
