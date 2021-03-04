/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "Context.hpp"
#include "VerticalRemapManager.hpp"
#include "RemapFunctor.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

void ComposeTransportImpl
::remap_v (const ExecViewUnmanaged<const Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>& dp3d,
           const int np1, const ExecViewUnmanaged<const Scalar*[NP][NP][NUM_LEV]>& dp,
           const ExecViewUnmanaged<Scalar*[2][NP][NP][NUM_LEV]>& v) {
  using Kokkos::parallel_for;
  const auto& vrm = Context::singleton().get<VerticalRemapManager>();
  const auto r = vrm.get_remapper();
  const auto policy = Kokkos::RangePolicy<ExecSpace>(0, dp3d.extent_int(0)*NP*NP*NUM_LEV*2);
  const auto pre = KOKKOS_LAMBDA (const int idx) {
    int ie, q, i, j, k;
    cti::idx_ie_q_ij_nlev<NUM_LEV>(2, idx, ie, q, i, j, k);
    v(ie,q,i,j,k) *= dp3d(ie,np1,i,j,k);
  };
  parallel_for(policy, pre);
  r->remap1(dp3d, np1, dp, v);
  const auto post = KOKKOS_LAMBDA (const int idx) {
    int ie, q, i, j, k;
    cti::idx_ie_q_ij_nlev<NUM_LEV>(2, idx, ie, q, i, j, k);
    v(ie,q,i,j,k) /= dp(ie,i,j,k);
  };
  parallel_for(policy, post);
}

void ComposeTransportImpl::remap_q (const TimeLevel& tl, const Real dt) {
  GPTLstart("compose_vertical_remap");

  GPTLstop("compose_vertical_remap");
}

} // namespace Homme
