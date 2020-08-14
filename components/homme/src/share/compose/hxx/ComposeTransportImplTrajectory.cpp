/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

KOKKOS_FUNCTION
void ugradv_sphere (
  const SphereOperators& sphere_ops, KernelVariables& kv,
  const typename ViewConst<ExecViewUnmanaged<Real[2][3][NP][NP]> >::type& vec_sphere2cart,
  // velocity, latlon
  const typename ViewConst<ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> >::type& u,
  const typename ViewConst<ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]> >::type& v,
  const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>& v_cart,
  const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>& ugradv_cart,
  // [u dot grad] v, latlon
  const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>& ugradv)
{
  for (int d_cart = 0; d_cart < 3; ++d_cart) {
    const auto f1 = [&] (const int i, const int j, const int k) {
      v_cart(i,j,k) = (vec_sphere2cart(0,d_cart,i,j) * v(0,i,j,k) +
                       vec_sphere2cart(1,d_cart,i,j) * v(1,i,j,k));      
    };
    cti::loop_ijk<NUM_LEV>(kv, f1);
    kv.team_barrier();

    sphere_ops.gradient_sphere<NUM_LEV>(kv, v_cart, ugradv_cart);

    const auto f2 = [&] (const int i, const int j, const int k) {
      if (d_cart == 0) ugradv(0,i,j,k) = ugradv(1,i,j,k) = 0;
      for (int d_latlon = 0; d_latlon < 2; ++d_latlon)
        ugradv(d_latlon,i,j,k) +=
          vec_sphere2cart(d_latlon,d_cart,i,j)*
          (u(0,i,j,k) * ugradv_cart(0,i,j,k) + u(1,i,j,k) * ugradv_cart(1,i,j,k));
    };
    cti::loop_ijk<NUM_LEV>(kv, f2);
  }
}

void ComposeTransportImpl::calc_trajectory () {
  assert( ! m_data.independent_time_steps); // until impl'ed

  const auto sphere_ops = m_sphere_ops;
  const auto geo = m_elements.m_geometry;
  const auto toplevel = KOKKOS_LAMBDA (const MT& team) {
    KernelVariables kv(team, m_tu_ne);
    const auto ie = kv.ie;

    const auto ugradv = Homme::subview(m_buf2[1], kv.team_idx);
    ugradv_sphere(sphere_ops, kv, Homme::subview(geo.m_vec_sph2cart, ie),
                  Homme::subview(m_derived.m_vn0, ie), Homme::subview(m_derived.m_vstar, ie),
                  Homme::subview(m_buf1, kv.team_idx), Homme::subview(m_buf2[0], kv.team_idx),
                  ugradv);
  };
  Kokkos::parallel_for(m_tp_ne, toplevel);
}

ComposeTransport::TestDepView::HostMirror ComposeTransportImpl::
test_trajectory(Real t0, Real t1, bool independent_time_steps) {
  ComposeTransport::TestDepView dep("dep", m_data.nelemd, num_phys_lev, np2, 3);
  calc_trajectory();
  const auto deph = Kokkos::create_mirror_view(dep);
  for (int ie = 0; ie < m_data.nelemd; ++ie)
    for (int lev = 0; lev < num_phys_lev; ++lev)
      for (int k = 0; k < np2; ++k)
          for (int d = 0; d < 3; ++d)
            deph(ie,lev,k,d) = ie*(lev + k) + d;
  return deph;
}

} // namespace Homme
