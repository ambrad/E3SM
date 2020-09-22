/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"

namespace Homme {

void ComposeTransportImpl::advance_hypervis_scalar (const Real dt_q) {
  const auto dt = dt_q / m_data.hv_q;
  for (int it = 0; it < m_data.hv_q; ++it) {
    const auto qsize = m_data.qsize;
    const auto Qtens = m_tracers.qtens_biharmonic;
    const auto Q = m_tracers.Q;
    // Qtens = Q
    const auto f1 = KOKKOS_LAMBDA (const int idx) {
      int ie, q, i, j, lev;
      idx_ie_q_ij_nlev<num_lev_pack>(qsize, idx, ie, q, i, j, lev);
      Qtens(ie,q,i,j,lev) = Q(ie,q,i,j,lev);
    };
    launch_ie_q_ij_nlev<num_lev_pack>(f1);
    // biharmonic_wk_scalar

    // Compute Q = Q spheremp - dt nu_q Qtens. N.B. spheremp is already in Qtens
    // from divergence_sphere_wk.

    // Halo exchange Q and apply rspheremp.

  }
}

} // namespace Homme
