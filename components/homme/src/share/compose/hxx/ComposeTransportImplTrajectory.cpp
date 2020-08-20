/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

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

/* Calculate the trajecotry at second order using Taylor series expansion. Also
   DSS the vertical velocity data if running the 3D algorithm.

   Derivation:
       p is position, v velocity
       p0 = p1 - dt/2 (v(p0,t0) + v(p1,t1)) + O(dt^3)
       O((p1 - p0)^2) = O(dt^2)
       p0 - p1 = -dt v(p1,t1) + O(dt^2)
       v(p0,t0) = v(p1,t0) + grad v(p1,t0) (p0 - p1) + O((p0 - p1)^2)
                = v(p1,t0) + grad v(p1,t0) (p0 - p1) + O(dt^2)
                = v(p1,t0) - dt grad v(p1,t0) v(p1,t1) + O(dt^2)
       p0 = p1 - dt/2 (v(p0,t0) + v(p1,t1)) + O(dt^3)
          = p1 - dt/2 (v(p1,t0) - dt grad v(p1,t0) v(p1,t1) + v(p1,t1)) + O(dt^3)
          = p1 - dt/2 (v(p1,t0) + v(p1,t1) - dt grad v(p1,t0) v(p1,t1)) + O(dt^3)
   In the code, v(p1,t0) = vstar, v(p1,t1) is vn0.
 */
void ComposeTransportImpl::calc_trajectory (const Real dt) {
  assert( ! m_data.independent_time_steps); // until impl'ed
  const auto sphere_ops = m_sphere_ops;
  const auto geo = m_elements.m_geometry;
  const auto m_vec_sph2cart = geo.m_vec_sph2cart;
  const auto m_vstar = m_derived.m_vstar;
  { // Calculate midpoint velocity.
    const auto m_spheremp = geo.m_spheremp;
    const auto m_rspheremp = geo.m_rspheremp;
    const auto m_vn0 = m_derived.m_vn0;
    const auto buf1 = m_data.buf1;
    const auto buf2a = m_data.buf2[0];
    const auto buf2b = m_data.buf2[1];
    const auto calc_midpoint_velocity = KOKKOS_LAMBDA (const MT& team) {
      KernelVariables kv(team, m_tu_ne);
      const auto ie = kv.ie;

      const auto vn0 = Homme::subview(m_vn0, ie);
      const auto vstar = Homme::subview(m_vstar, ie);
      const auto ugradv = Homme::subview(buf2b, kv.team_idx);
      ugradv_sphere(sphere_ops, kv, Homme::subview(m_vec_sph2cart, ie),
                    vn0, vstar,
                    Homme::subview(buf1, kv.team_idx),
                    Homme::subview(buf2a, kv.team_idx),
                    ugradv);

      // Write the midpoint velocity to vstar.
      const auto spheremp = Homme::subview(m_spheremp, ie);
      const auto rspheremp = Homme::subview(m_rspheremp, ie);
      const auto f = [&] (const int i, const int j, const int k) {
        for (int d = 0; d < 2; ++d)
          vstar(d,i,j,k) = (((vn0(d,i,j,k) + vstar(d,i,j,k))/2 - dt*ugradv(d,i,j,k)/2)*
                            spheremp(i,j)*rspheremp(i,j));
      };
      cti::loop_ijk<num_lev_pack>(kv, f);
    };
    Kokkos::parallel_for(m_tp_ne, calc_midpoint_velocity);
  }
  { // DSS velocity.
    Kokkos::fence();
    const auto be = m_data.independent_time_steps ? m_v_dss_be[1] : m_v_dss_be[0];
    be->exchange();
  }
  { // Calculate departure point.
    const auto m_sphere_cart = geo.m_sphere_cart;
    const auto rearth = PhysicalConstants::rearth;
    const auto m_dep_pts = m_data.dep_pts;
    const auto calc_departure_point = KOKKOS_LAMBDA (const MT& team) {
      KernelVariables kv(team, m_tu_ne);
      const auto ie = kv.ie;

      const auto vstar = Homme::subview(m_vstar, ie);
      const auto vec_sphere2cart = Homme::subview(m_vec_sph2cart, ie);
      const auto sphere_cart = Homme::subview(m_sphere_cart, ie);
      const auto dep_pts = Homme::subview(m_dep_pts, ie);
      const auto f = [&] (const int i, const int j, const int k) {
        // dp = p1 - dt v/rearth
        Scalar dp[3], r = 0;
        for (int d = 0; d < 3; ++d) {
          const auto vel_cart = (vec_sphere2cart(0,d,i,j)*vstar(0,i,j,k) +
                                 vec_sphere2cart(1,d,i,j)*vstar(1,i,j,k));
          dp[d] = sphere_cart(i,j,d) - dt*vel_cart/rearth;
        }
        const auto r2 = square(dp[0]) + square(dp[1]) + square(dp[2]);
        // Pack -> scalar storage.
        const auto os = packn*k;
        for (int s = 0; s < packn; ++s) {
          if (num_phys_lev % packn != 0 && // compile out this conditional when possible
              os+s >= num_phys_lev) break;
          // No vec call for sqrt.
          const auto r = std::sqrt(r2[s]);
          for (int d = 0; d < 3; ++d)
            dep_pts(i,j,os+s,d) = dp[d][s]/r;
        }
      };
      cti::loop_ijk<num_lev_pack>(kv, f);
    };
    Kokkos::parallel_for(m_tp_ne, calc_departure_point);
  }
}

ComposeTransport::TestDepView::HostMirror ComposeTransportImpl::
test_trajectory(Real t0, Real t1, bool independent_time_steps) {
  ComposeTransport::TestDepView dep("dep", m_data.nelemd, num_phys_lev, np2, 3);
  calc_trajectory(t1 - t0);
  const auto deph = Kokkos::create_mirror_view(dep);
  for (int ie = 0; ie < m_data.nelemd; ++ie)
    for (int lev = 0; lev < num_phys_lev; ++lev)
      for (int k = 0; k < np2; ++k)
          for (int d = 0; d < 3; ++d)
            deph(ie,lev,k,d) = ie*(lev + k) + d;
  return deph;
}

} // namespace Homme
