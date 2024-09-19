/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#ifdef HOMME_ENABLE_COMPOSE

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

#include "compose_test.hpp"

namespace Homme {
using cti = ComposeTransportImpl;
using CSNlev  = cti::CSNlev;
using CRNlev  = cti::CRNlev;
using CSNlevp = cti::CSNlevp;
using CRNlevp = cti::CRNlevp;
using CS2Nlev = cti::CS2Nlev;
using SNlev   = cti::SNlev;
using RNlev   = cti::RNlev;
using SNlevp  = cti::SNlevp;
using RNlevp  = cti::RNlevp;
using S2Nlev  = cti::S2Nlev;
using R2Nlev  = cti::R2Nlev;
using S2Nlevp = cti::S2Nlevp;

template <int N> using RNV = ExecViewUnmanaged<Real[NP][NP][N]>;
template <int N> using CRNV = typename ViewConst<RNV<N>>::type;

// For sorted ascending x[0:n] and x in [x[0], x[n-1]] with hint xi_idx, return
// i such that x[i] <= xi <= x[i+1].
//   This function is meant for the case that x_idx is very close to the
// support. If that isn't true, then this method is inefficient; binary search
// should be used instead.
template <typename ConstRealArray>
KOKKOS_FUNCTION static
int find_support (const int n, ConstRealArray x, const int x_idx,
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

template <int N>
KOKKOS_FUNCTION static void
linterp (const KernelVariables& kv, const int n, const CRNV<N>& x, const CRNV<N>& y,
         const RNV<N>& xi, const RNV<N>& yi, const char* const caller = nullptr) {
  assert(n <= N);
  
}

void ComposeTransportImpl::calc_enhanced_trajectory (const int np1, const Real dt) {
}

static int test_find_support () {
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

#define comunittest(f) do {                     \
    ne = f();                                   \
    if (ne) printf(#f " ne %d\n", ne);          \
    nerr += ne;                                 \
  } while (0)

int ComposeTransportImpl::run_enhanced_trajectory_unit_tests () {
  int nerr = 0, ne;
  comunittest(test_find_support);
  return nerr;
}

#undef comunittest

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
