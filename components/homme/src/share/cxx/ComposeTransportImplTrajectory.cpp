/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

#include "compose_test.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

KOKKOS_FUNCTION
static void ugradv_sphere (
  const SphereOperators& sphere_ops, const KernelVariables& kv,
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

typedef typename ViewConst<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV_P]> >::type CSNlevp;
typedef typename ViewConst<ExecViewUnmanaged<Real[NP][NP][NUM_LEV_P*VECTOR_SIZE]> >::type CRNlevp;
typedef ExecViewUnmanaged<Real[NP][NP][NUM_LEV*VECTOR_SIZE]> RNlev;

/* Form a 3rd-degree Lagrange polynomial over (x(k-1:k+1), y(k-1:k+1)) and set
   yi(k) to its derivative at x(k).
 */
KOKKOS_FUNCTION static void approx_derivative (
  const KernelVariables& kv, const CSNlevp& xs, const CSNlevp& ys,
  const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>& yps) // yp(:,:,0) is undefined
{
  CRNlevp x(cti::cpack2real(xs));
  CRNlevp y(cti::cpack2real(ys));
  RNlev yp(cti::pack2real(yps));
  const auto f = [&] (const int i, const int j, const int k) {
    if (k == 0) return;
    const auto& xkm1 = x(i,j,k-1);
    const auto& xk   = x(i,j,k  ); // also the interpolation point
    const auto& xkp1 = x(i,j,k+1);
    yp(i,j,k) = (y(i,j,k-1)*((         1 /(xkm1 - xk  ))*((xk - xkp1)/(xkm1 - xkp1))) +
                 y(i,j,k  )*((         1 /(xk   - xkm1))*((xk - xkp1)/(xk   - xkp1)) +
                             ((xk - xkm1)/(xk   - xkm1))*(         1 /(xk   - xkp1))) +
                 y(i,j,k+1)*(((xk - xkm1)/(xkp1 - xkm1))*(         1 /(xkp1 - xk  ))));
  };
  cti::loop_ijk<cti::num_phys_lev>(kv, f);
}

/* Calculate the trajectory at second order using Taylor series expansion. Also
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
  const auto geo = m_geometry;
  const auto m_vec_sph2cart = geo.m_vec_sph2cart;
  const auto m_vstar = m_derived.m_vstar;
  { // Calculate midpoint velocity.
    const auto m_spheremp = geo.m_spheremp;
    const auto m_rspheremp = geo.m_rspheremp;
    const auto m_v = m_state.m_v;
    const auto m_vn0 = m_derived.m_vn0;
    const auto buf1 = m_data.buf1;
    const auto buf2a = m_data.buf2[0];
    const auto buf2b = m_data.buf2[1];
    const auto np1 = m_data.np1;
    const auto independent_time_steps = m_data.independent_time_steps;
    const auto calc_midpoint_velocity = KOKKOS_LAMBDA (const MT& team) {
      KernelVariables kv(team, m_tu_ne);
      const auto ie = kv.ie;

      //todo independent_time_steps code here

      const auto vn0 = (independent_time_steps ?
                        Homme::subview(m_vn0, ie) :
                        Homme::subview(m_v, ie, np1));
      const auto vstar = Homme::subview(m_vstar, ie);
      const auto ugradv = Homme::subview(buf2b, kv.team_idx);
      ugradv_sphere(sphere_ops, kv, Homme::subview(m_vec_sph2cart, ie), vn0, vstar,
                    Homme::subview(buf1, kv.team_idx), Homme::subview(buf2a, kv.team_idx),
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
    const int packn = this->packn;
    const int num_phys_lev = this->num_phys_lev;
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
          const auto oss = os + s;
          if (num_phys_lev % packn != 0 && // compile out this conditional when possible
              oss >= num_phys_lev) break;
          // No vec call for sqrt.
          const auto r = std::sqrt(r2[s]);
          for (int d = 0; d < 3; ++d)
            dep_pts(oss,i,j,d) = dp[d][s]/r;
        }
      };
      cti::loop_ijk<num_lev_pack>(kv, f);
    };
    Kokkos::parallel_for(m_tp_ne, calc_departure_point);
  }
}

static int test_approx_derivative () {
  const Real a = 1.5, b = -0.7, c = 0.2;
  int nerr = 0;
  ExecView<Scalar[NP][NP][NUM_LEV_P]> xp("xp"), yp("yp");
  ExecView<Real[NP][NP][NUM_LEV_P*VECTOR_SIZE]>
    xs(cti::pack2real(xp)), ys(cti::pack2real(yp));
  const auto policy = Homme::get_default_team_policy<ExecSpace>(1);
  TeamUtils<ExecSpace> tu(policy);
  { // Fill xs and ys with manufactured coordinates and function.
    const auto f = KOKKOS_LAMBDA (const cti::MT& team) {
      KernelVariables kv(team, tu);
      const auto f = [&] (const int i, const int j, const int k) {
        const Real x = 1.7*(k + 1e-1*i*k + 1e-2*j*k*k)/cti::num_phys_lev;
        xs(i,j,k) = x;
        ys(i,j,k) = (a*x + b)*x + c;
      };
      cti::loop_ijk<cti::num_phys_lev+1>(kv, f);
    };
    Kokkos::parallel_for(policy, f);
  }
  ExecView<Scalar[NP][NP][NUM_LEV]> yip("yp");
  { // Run approx_derivative.
    const auto f = KOKKOS_LAMBDA (const cti::MT& team) {
      KernelVariables kv(team, tu);
      approx_derivative(kv, xp, yp, yip);
    };
    Kokkos::fence();
    Kokkos::parallel_for(policy, f);
  }
  { // Check answer.
    Kokkos::fence();
    const auto xsh = cti::cmvdc(xs);
    ExecView<Real[NP][NP][NUM_LEV*VECTOR_SIZE]> yis(cti::pack2real(yip));
    const auto yish = cti::cmvdc(yis);
    for (int i = 0; i < cti::np; ++i)
      for (int j = 0; j < cti::np; ++j)
        for (int k = 1; // k = 0 is not written
             k < cti::num_phys_lev; ++k) {
          const Real x = xsh(i,j,k);
          const Real ypp = 2*a*x + b;
          const Real err = std::abs(yish(i,j,k) - ypp);
          if (err > 50*std::numeric_limits<Real>::epsilon()) {
            ++nerr;
            printf("%2d %d %d %1.2f %6.2f %6.2f %9.2e\n",
                   i, j, k, x, ypp, yish(i,j,k), err);
          }
        }
  }
  return nerr;
}

int ComposeTransportImpl::run_trajectory_unit_tests () {
  return test_approx_derivative();
}

ComposeTransport::TestDepView::HostMirror ComposeTransportImpl::
test_trajectory(Real t0, Real t1, const bool independent_time_steps) {
  m_data.np1 = 0;
  const auto vstar = Kokkos::create_mirror_view(m_derived.m_vstar);
  const auto v = Kokkos::create_mirror_view(m_state.m_v);
  const auto pll = cmvdc(m_geometry.m_sphere_latlon);
  const auto np1 = m_data.np1;
  const int packn = this->packn;
  const compose::test::NonDivergentWindField wf;
  // On host b/c trig isn't BFB between host and device.
  const auto f = [&] (const int ie, const int lev, const int i, const int j) {
    Real latlon[] = {pll(ie,i,j,0), pll(ie,i,j,1)};
    compose::test::offset_latlon(num_phys_lev, lev, latlon[0], latlon[1]);
    Real uv[2];
    wf.eval(t0, latlon, uv);
    const int p = lev/packn, s = lev%packn;
    for (int d = 0; d < 2; ++d) vstar(ie,d,i,j,p)[s] = uv[d];
    wf.eval(t1, latlon, uv);
    for (int d = 0; d < 2; ++d) v(ie,np1,d,i,j,p)[s] = uv[d];
  };
  loop_host_ie_plev_ij(f);
  Kokkos::deep_copy(m_derived.m_vstar, vstar);
  Kokkos::deep_copy(m_state.m_v, v);

  calc_trajectory(t1 - t0);
  Kokkos::fence();

  const auto deph = cti::cmvdc(m_data.dep_pts);
  return deph;
}

} // namespace Homme
