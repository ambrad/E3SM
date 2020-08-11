/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP
#define HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP

#include "compose_kokkos.hpp"
#include "compose_cedr_cdr.hpp"
#include "compose_slmm_islmpi.hpp"
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
#include "profiling.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "mpi/Connectivity.hpp"

#include <cassert>

#include "/home/ambrad/repo/sik/hommexx/dbg.hpp"

namespace Homme {

struct ComposeTransportImpl {
  enum : int { np = NP };
  enum : int { packn = VECTOR_SIZE };
  enum : int { np2 = NP*NP };
  enum : int { max_num_lev_pack = NUM_LEV_P };
  enum : int { num_lev_aligned = max_num_lev_pack*packn };
  enum : int { num_phys_lev = NUM_PHYSICAL_LEV };
  enum : int { num_work = 12 };

  static_assert(num_lev_aligned >= 3,
                "We use wrk(0:2,:) and so need num_lev_aligned >= 3");

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MT = typename TeamPolicy::member_type;

  struct Data {
    int nelemd, qsize, hv_q, np1_qdp;
    bool independent_time_steps;

    Data () : nelemd(-1), qsize(-1), hv_q(1), np1_qdp(-1), independent_time_steps(false) {}
  };

  homme::islmpi::IslMpi<ko::MachineTraits>::Ptr islet;
  homme::CDR<ko::MachineTraits>::Ptr cdr;

  const HybridVCoord m_hvcoord;
  const Elements m_elements;
  const ElementsState m_state;
  const ElementsDerivedState m_derived;
  const Tracers m_tracers;
  Data m_data;

  TeamPolicy m_policy;
  TeamUtils<ExecSpace> m_tu;
  int nslot;

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
      m_policy(1,1,1), m_tu(m_policy) // throwaway settings
  {}

  void reset(const SimulationParams& params);

  int requested_buffer_size () const {
    // FunctorsBuffersManager wants the size in terms of sizeof(Real).
    return 0;
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
  }

  void init_boundary_exchanges();

  void run();

  void calc_trajectory();

  ComposeTransport::TestDepView::HostMirror
  test_trajectory(Real t0, Real t1, bool independent_time_steps);
};

} // namespace Homme

#endif // HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP
