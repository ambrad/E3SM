/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP
#define HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP

#include "ComposeTransport.hpp"

#include "Context.hpp"
#include "Elements.hpp"
#include "ElementsGeometry.hpp"
#include "ElementsDerivedState.hpp"
#include "FunctorsBuffersManager.hpp"
#include "ErrorDefs.hpp"
#include "EulerStepFunctor.hpp"
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

struct ComposeTransportImpl {
  enum : int { np = NP };
  enum : int { packn = VECTOR_SIZE };
  enum : int { np2 = NP*NP };
  enum : int { num_lev_pack = NUM_LEV };
  enum : int { max_num_lev_pack = NUM_LEV_P };
  enum : int { num_lev_aligned = max_num_lev_pack*packn };
  enum : int { num_phys_lev = NUM_PHYSICAL_LEV };
  enum : int { num_work = 12 };

  static_assert(num_lev_aligned >= 3,
                "We use wrk(0:2,:) and so need num_lev_aligned >= 3");

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MT = typename TeamPolicy::member_type;

  using Buf1 = ExecViewUnmanaged<Scalar*[NP][NP][NUM_LEV]>;
  using Buf2 = ExecViewUnmanaged<Scalar*[2][NP][NP][NUM_LEV]>;

  using DeparturePoints = ExecViewManaged<Real*[NUM_PHYSICAL_LEV][NP][NP][3]>;

  struct Data {
    int nelemd, qsize, hv_q, np1, np1_qdp, limiter_option;
    bool independent_time_steps;

    int nslot;
    Buf1 buf1;
    Buf2 buf2[2];

    DeparturePoints dep_pts;

    Data () : nelemd(-1), qsize(-1), hv_q(1), np1_qdp(-1), independent_time_steps(false) {}
  };

  const HybridVCoord m_hvcoord;
  const Elements m_elements;
  const ElementsState m_state;
  const ElementsDerivedState m_derived;
  const Tracers m_tracers;
  SphereOperators m_sphere_ops;
  Data m_data;

  TeamPolicy m_tp_ne, m_tp_ne_qsize;
  TeamUtils<ExecSpace> m_tu_ne, m_tu_ne_qsize;

  std::shared_ptr<BoundaryExchange>
    m_qdp_dss_be[Q_NUM_TIME_LEVELS], m_v_dss_be[2], m_Q_dss_be;

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size (const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }

  ComposeTransportImpl ()
    : m_hvcoord(Context::singleton().get<HybridVCoord>()),
      m_elements(Context::singleton().get<Elements>()),
      m_derived(m_elements.m_derived),
      m_state(m_elements.m_state),
      m_tracers(Context::singleton().get<Tracers>()),
      m_sphere_ops(Context::singleton().get<SphereOperators>()),
      m_tp_ne(1,1,1), m_tu_ne(m_tp_ne), // throwaway settings
      m_tp_ne_qsize(1,1,1), m_tu_ne_qsize(m_tp_ne_qsize) // throwaway settings
  {}

  void reset(const SimulationParams& params);
  int requested_buffer_size() const;
  void init_buffers(const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  void run(const TimeLevel& tl, const Real dt);

  void calc_trajectory(const Real dt);

  ComposeTransport::TestDepView::HostMirror
  test_trajectory(Real t0, Real t1, bool independent_time_steps);

  void test_2d(const int nstep, std::vector<Real>& eval);

  template <int KLIM, typename Fn>
  KOKKOS_INLINE_FUNCTION
  static void loop_ijk (const KernelVariables& kv, const Fn& h) {
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    if (OnGpu<ExecSpace>::value) {
      const auto ttr = TeamThreadRange(kv.team, KLIM);
      const auto tvr = ThreadVectorRange(kv.team, NP*NP);
      const auto f = [&] (const int idx) {
        const int i = idx / NP, j = idx % NP;
        const auto g = [&] (const int k) { h(i,j,k); };
        parallel_for(tvr, g);
      };
      parallel_for(ttr, f);
    } else if (kv.team.team_size() == 1) {
      for (int i = 0; i < NP; ++i)
        for (int j = 0; j < NP; ++j)
          for (int k = 0; k < KLIM; ++k)
            h(i,j,k);
    } else {
      const auto tr = TeamThreadRange(kv.team, KLIM);
      const auto f = [&] (const int k) {
        for (int i = 0; i < NP; ++i)
          for (int j = 0; j < NP; ++j)
            h(i,j,k);
      };
      parallel_for(tr, f);
    }
  }

  template <typename Fn>
  void loop_host_ie_plev_ij (const Fn& f) const {
    for (int ie = 0; ie < m_data.nelemd; ++ie)
      for (int lev = 0; lev < num_phys_lev; ++lev)
        for (int i = 0; i < np; ++i)
          for (int j = 0; j < np; ++j)
            f(ie, lev, i, j);
  }

  template <int nlev> KOKKOS_INLINE_FUNCTION
  static void idx_ie_nlev_ij (const int idx, int& ie, int& lev, int& i, int& j) {
    ie = idx / (nlev*np*np);
    lev = (idx / (np*np)) % nlev;
    i = (idx / np) % np;
    j = idx % np;
  }

  KOKKOS_INLINE_FUNCTION
  static void idx_ie_physlev_ij (const int idx, int& ie, int& lev, int& i, int& j) {
    return idx_ie_nlev_ij<num_phys_lev>(idx, ie, lev, i, j);
  }

  template <typename Fn>
  void launch_ie_physlev_ij (Fn& f) const {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, m_data.nelemd*np*np*num_phys_lev), f);
  }

  KOKKOS_INLINE_FUNCTION
  static void idx_ie_packlev_ij (const int idx, int& ie, int& lev, int& i, int& j) {
    return idx_ie_nlev_ij<num_lev_pack>(idx, ie, lev, i, j);
  }

  template <typename Fn>
  void launch_ie_packlev_ij (Fn& f) const {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, m_data.nelemd*np*np*num_lev_pack), f);
  }

  template <int nlev> KOKKOS_INLINE_FUNCTION
  static void idx_ie_ij_nlev (const int idx, int& ie, int& i, int& j, int& lev) {
    ie = idx / (np*np*nlev);
    i = (idx / (np*nlev)) % np;
    j = (idx / nlev) % np;
    lev = idx % nlev;
  }

  template <int nlev, typename Fn>
  void launch_ie_ij_nlev (Fn& f) const {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, m_data.nelemd*np*np*nlev), f);
  }

  template <int nlev> KOKKOS_INLINE_FUNCTION
  static void idx_ie_q_ij_nlev (const int qsize, const int idx,
                                int& ie, int& q, int& i, int& j, int& lev) {
    ie = idx / (qsize*np*np*nlev);
    q = (idx / (np*np*nlev)) % qsize;
    i = (idx / (np*nlev)) % np;
    j = (idx / nlev) % np;
    lev = idx % nlev;
  }

  template <int nlev, typename Fn>
  void launch_ie_q_ij_nlev (Fn& f) const {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, m_data.nelemd*m_data.qsize*np*np*nlev), f);
  }
};

} // namespace Homme

#endif // HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP