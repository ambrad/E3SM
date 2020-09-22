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
    { // Qtens = Q
      const auto f = KOKKOS_LAMBDA (const int idx) {
        int ie, q, i, j, lev;
        idx_ie_q_ij_nlev<num_lev_pack>(qsize, idx, ie, q, i, j, lev);
        Qtens(ie,q,i,j,lev) = Q(ie,q,i,j,lev);
      };
      launch_ie_q_ij_nlev<num_lev_pack>(f);
    }
    // biharmonic_wk_scalar
    const auto laplace_simple_Qtens = [&] () {
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
        const auto Qtens_ie = Homme::subview(Qtens, kv.ie, kv.iq);
        m_sphere_ops.laplace_simple(kv, Qtens_ie, Qtens_ie);
      };
      Kokkos::parallel_for(m_tp_ne_qsize, f);
    };
    laplace_simple_Qtens();
    m_hv_dss_be[0]->exchange(m_elements.m_geometry.m_rspheremp);
    if (m_data.hv_scaling == 0) {
      laplace_simple_Qtens();
    } else {
      const auto tensorvisc = m_elements.m_geometry.m_tensorvisc;
      const auto f = KOKKOS_LAMBDA (const MT& team) {
        KernelVariables kv(team, m_data.qsize, m_tu_ne_qsize);
        const auto Qtens_ie = Homme::subview(Qtens, kv.ie, kv.iq);
        m_sphere_ops.laplace_tensor(kv, Homme::subview(tensorvisc, kv.ie),
                                    Qtens_ie, Qtens_ie);
      };
      Kokkos::parallel_for(m_tp_ne_qsize, f);
    }
    // Compute Q = Q spheremp - dt nu_q Qtens. N.B. spheremp is already in Qtens
    // from divergence_sphere_wk.
    
    // Halo exchange Q and apply rspheremp.
    m_hv_dss_be[1]->exchange(m_elements.m_geometry.m_rspheremp);
  }
}

} // namespace Homme
