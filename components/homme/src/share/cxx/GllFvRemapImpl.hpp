/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_GLLFVREMAP_IMPL_HPP
#define HOMMEXX_GLLFVREMAP_IMPL_HPP

#include "GllFvRemap.hpp"

#include "Context.hpp"
#include "Elements.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsDerivedState.hpp"
#include "FunctorsBuffersManager.hpp"
#include "ErrorDefs.hpp"
#include "HommexxEnums.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "SphereOperators.hpp"
#include "Tracers.hpp"
#include "TimeLevel.hpp"
#include "profiling.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "mpi/Connectivity.hpp"

#include <cassert>

namespace Homme {

struct GllFvRemapImpl {
  enum : int { np = NP };
  enum : int { packn = VECTOR_SIZE };
  enum : int { np2 = NP*NP };
  enum : int { num_lev_pack = NUM_LEV };
  enum : int { max_num_lev_pack = NUM_LEV_P };
  enum : int { num_lev_aligned = max_num_lev_pack*packn };
  enum : int { num_phys_lev = NUM_PHYSICAL_LEV };
  enum : int { num_work = 12 };

  typedef GllFvRemap::Phys0T Phys0T;
  typedef GllFvRemap::Phys1T Phys1T;
  typedef GllFvRemap::Phys2T Phys2T;
  typedef GllFvRemap::CPhys1T CPhys1T;
  typedef GllFvRemap::CPhys2T CPhys2T;
  typedef ExecViewUnmanaged<Scalar**>  VPhys1T;
  typedef ExecViewUnmanaged<Scalar***> VPhys2T;
  typedef VPhys1T::const_type CVPhys1T;
  typedef VPhys2T::const_type CVPhys2T;

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MT = typename TeamPolicy::member_type;

  using Buf1 = ExecViewUnmanaged<Scalar*[NP][NP][NUM_LEV_P]>;
  using Buf2 = ExecViewUnmanaged<Scalar*[2][NP][NP][NUM_LEV_P]>;

  struct Data {
    int nelemd, qsize, nf2;

    Buf1 buf1[3];
    Buf2 buf2[2];

    Data ()
      : nelemd(-1), qsize(-1), nf2(-1)
    {}
  };

  const HybridVCoord m_hvcoord;
  const Elements m_elements;
  const ElementsState m_state;
  const ElementsDerivedState m_derived;
  const ElementsGeometry m_geometry;
  const Tracers m_tracers;
  SphereOperators m_sphere_ops;
  int nslot;
  Data m_data;

  TeamPolicy m_tp_ne, m_tp_ne_qsize;
  TeamUtils<ExecSpace> m_tu_ne, m_tu_ne_qsize;

  std::shared_ptr<BoundaryExchange>
    m_qdp_dss_be[Q_NUM_TIME_LEVELS], m_v_dss_be[2], m_hv_dss_be[2];

  GllFvRemapImpl();

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size (const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }

  void reset(const SimulationParams& params);
  int requested_buffer_size() const;
  void init_buffers(const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  void run_dyn_to_fv(const int time_idx, const Phys0T& ps, const Phys0T& phis,
                     const Phys1T& T, const Phys1T& omega, const Phys2T& uv,
                     const Phys2T& q);
  void run_fv_to_dyn(const int time_idx, const Real dt, const CPhys1T& T,
                     const CPhys2T& uv, const CPhys2T& q);

  template <typename CR1, typename V1, typename CV2, typename V2>
  static KOKKOS_FUNCTION void
  limiter_clip_and_sum (const int nlev, const int n, const CR1& spheremp,
                        const V1& qmin, const V1& qmax, const CV2& dp, const V2& q) {
    
  }

  /* Compute
         y(1:m,k) = (A (d1 x(1:n,k)))/d2, k = 1:nlev
     Sizes are min; a dim can have larger size.
         A m by n, d1 n, d2 m
         x n by nlev, w n by nlev, y m by nlev
     Permitted aliases:
         w = x
   */
  template <typename AT, typename D1T, typename D2T, typename XT, typename WT, typename YT>
  static KOKKOS_FUNCTION void
  matvec (const MT& team,
          const int m, const int n, const int nlev, // range of x,y fastest dim
          const AT& A, const D1T& d1, const D2T& d2,
          const XT& x, const YT& w, const WT& y) {
    assert(A.extent_int(0) >= m && A.extent_int(1) >= n);
    assert(d1.extent_int(0) >= n); assert(d2.extent_int(0) >= m);
    assert(x.extent_int(0) >= n && x.extent_int(1) >= nlev);
    assert(y.extent_int(0) >= m && y.extent_int(1) >= nlev);
    using Kokkos::parallel_for;
    const auto ttrn = Kokkos::TeamThreadRange(team, n);
    const auto ttrm = Kokkos::TeamThreadRange(team, m);
    const auto tvr = Kokkos::ThreadVectorRange(team, nlev);
    parallel_for( ttrn,   [&] (const int i) {
      parallel_for(tvr,   [&] (const int k) { w(i,k) = x(i,k) * d1(i); }); });
    team.team_barrier();
    parallel_for( ttrm,   [&] (const int i) {
      parallel_for(tvr,   [&] (const int k) { y(i,k) = 0; });
      for (int j = 0; j < n; ++j)
        parallel_for(tvr, [&] (const int k) { y(i,k) += A(i,j) * w(j,k); });
      parallel_for(tvr,   [&] (const int k) { y(i,k) /= d2(i); }); });
  }

  /* Compute
         xt(i,d,k) = Dinv(i,:,:) x(i,:,k), i = 1:n
         yt(1:m,d,k) = (A (d1 (xt(1:n,d,k))))/d2
         y(i,d,k) = D(i,:,:) yt(i,:,k),    i = 1:n
     for k = 1:nlev. Sizes are min; a dim can have larger size.
         A m by n, d1 n, d2 m
         Dinv n by 2 by 2, D m by 2 by 2
         x (n,2) by nlev, w (n,2) by nlev with x indexing, y (m,2) by nlev
     Permitted aliases:
         w = x
     If x_idx_dof_d, then x is indexed as x(i,d,k) and y as y(d,i,k); else the
     opposite.
   */
  // Handle (dof,d) vs (d,dof) index ordering.
  template <bool idx_dof_d> static KOKKOS_INLINE_FUNCTION void
  matvec_idx_order (const int& dof, const int& d, int& i1, int& i2)
  { if (idx_dof_d) { i1 = dof; i2 = d; } else { i1 = d; i2 = dof; } }
  template <bool idx_dof_d> static KOKKOS_INLINE_FUNCTION int
  matvec_idx_dof (             const int idx) { return idx_dof_d ? idx / 2 : idx % 2; }
  template <bool idx_dof_d> static KOKKOS_INLINE_FUNCTION int
  matvec_idx_d   (const int n, const int idx) { return idx_dof_d ? idx % n : idx / n; }
  template <bool x_idx_dof_d,
            typename AT, typename D1T, typename D2T, typename DinvT, typename DT,
            typename XT, typename WT, typename YT>
  static KOKKOS_FUNCTION void
  matvec (const MT& team,
          const int m, const int n, const int nlev,
          const AT& A, const D1T& d1, const D2T& d2,
          const DinvT& Dinv, const DT& D,
          const XT& x, const WT& w, const YT& y) {
    using Kokkos::parallel_for;
    const auto ttrn  = Kokkos::TeamThreadRange(team,   n);
    const auto ttrm  = Kokkos::TeamThreadRange(team,   m);
    const auto ttr2m = Kokkos::TeamThreadRange(team, 2*m);
    const auto tvr = Kokkos::ThreadVectorRange(team, nlev);
    parallel_for(ttrn, [&] (const int i) {
      // This impl permits w to alias x. The alternative is to use twice as many
      // threads but w can't alias x.
      int i11, i12; matvec_idx_order<x_idx_dof_d>(i, 0, i11, i12);
      int i21, i22; matvec_idx_order<x_idx_dof_d>(i, 1, i21, i22);
      parallel_for(tvr, [&] (const int k) {
        const auto x1 = x(i11,i12,k), x2 = x(i21,i22,k);
        w(i11,i12,k) = (Dinv(i11,i12,1)*x1 + Dinv(i11,i12,2)*x2) * d1(i);
        w(i21,i22,k) = (Dinv(i21,i22,1)*x1 + Dinv(i21,i22,2)*x2) * d1(i);
      });
    });
    team.team_barrier();
    parallel_for(ttr2m, [&] (const int idx) {
      const int i = matvec_idx_dof<!x_idx_dof_d>(idx);
      const int d = matvec_idx_d<!x_idx_dof_d>(m, idx);
      int yi1, yi2; matvec_idx_order<!x_idx_dof_d>(i, d, yi1, yi2);
      parallel_for(tvr, [&] (const int k) { y(yi1,yi2,k) = 0; });
      for (int j = 0; j < n; ++j) {
        int xj1, xj2; matvec_idx_order<x_idx_dof_d>(j, d, xj1, xj2);
        parallel_for(tvr, [&] (const int k) { y(yi1,yi2,k) += A(i,j) * w(xj1,xj2,k); });
      }
      parallel_for(tvr, [&] (const int k) { y(yi1,yi2,k) /= d2(i); });
    });
    team.team_barrier();
    parallel_for(ttrm, [&] (const int i) {
      // This impl avoids a work slot having y's structure; the alternative
      // using twice as many threads requires an extra work slot.
      int i11, i12; matvec_idx_order<!x_idx_dof_d>(i, 0, i11, i12);
      int i21, i22; matvec_idx_order<!x_idx_dof_d>(i, 1, i21, i22);
      parallel_for(tvr, [&] (const int k) {
        const auto y1 = y(i11,i12,k), y2 = y(i21,i22,k);
        y(i11,i12,k) = (D(i11,i12,1)*y1 + D(i11,i12,2)*y2);
        y(i21,i22,k) = (D(i21,i22,1)*y1 + D(i21,i22,2)*y2);
      });
    });
  }

  template <typename View> static KOKKOS_INLINE_FUNCTION
  Real* pack2real (const View& v) { return &(*v.data())[0]; }
  template <typename View> static KOKKOS_INLINE_FUNCTION
  const Real* cpack2real (const View& v) { return &(*v.data())[0]; }
  template <typename View> static KOKKOS_INLINE_FUNCTION
  Scalar* real2pack (const View& v) { return reinterpret_cast<Scalar*>(v.data()); }
  template <typename View> static KOKKOS_INLINE_FUNCTION
  const Scalar* creal2pack (const View& v) { return reinterpret_cast<const Scalar*>(v.data()); }
};

} // namespace Homme

#endif // HOMMEXX_GLLFVREMAP_IMPL_HPP
