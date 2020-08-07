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
  enum : int { packn = VECTOR_SIZE };
  enum : int { scaln = NP*NP };
  enum : int { npack = (scaln + packn - 1)/packn };
  enum : int { max_num_lev_pack = NUM_LEV_P };
  enum : int { num_lev_aligned = max_num_lev_pack*packn };
  enum : int { num_phys_lev = NUM_PHYSICAL_LEV };
  enum : int { num_work = 12 };

  static_assert(num_lev_aligned >= 3,
                "We use wrk(0:2,:) and so need num_lev_aligned >= 3");

  using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
  using MT = typename TeamPolicy::member_type;

  using Work
    = Kokkos::View<Scalar*[num_work][num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace>;
  using WorkSlot
    = Kokkos::View<Scalar           [num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using ConstWorkSlot
    = Kokkos::View<const Scalar     [num_lev_aligned][npack],
                   Kokkos::LayoutRight, ExecSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  struct Data {
    int nelemd, qsize, hv_q, np1_qdp;
    bool independent_time_steps;

    Data () : nelemd(-1), qsize(-1), hv_q(1), np1_qdp(-1), independent_time_steps(true) {}
  };

  homme::islmpi::IslMpi<ko::MachineTraits>::Ptr islet;
  homme::CDR<ko::MachineTraits>::Ptr cdr;

  const HybridVCoord m_hvcoord;
  const Elements m_elements;
  const ElementsState m_state;
  const ElementsDerivedState m_derived;
  const Tracers m_tracers;
  Data m_data;

  Work m_work;
  TeamPolicy m_policy;
  TeamUtils<ExecSpace> m_tu;
  int nslot;

  std::shared_ptr<BoundaryExchange> m_qdp_dss_be[Q_NUM_TIME_LEVELS], m_v_dss_be, m_Q_dss_be;

  KOKKOS_INLINE_FUNCTION
  static WorkSlot get_work_slot (const Work& w, const int& wi, const int& si) {
    using Kokkos::subview;
    using Kokkos::ALL;
    const auto a = ALL();
    return subview(w, wi, si, a, a);
  }

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
    return Work::shmem_size(nslot);
  }

  void init_buffers (const FunctorsBuffersManager& fbm) {
    Scalar* mem = reinterpret_cast<Scalar*>(fbm.get_memory());
    m_work = Work(mem, nslot);
    mem += Work::shmem_size(nslot)/sizeof(Scalar);
  }

  template <typename Fn>
  KOKKOS_INLINE_FUNCTION
  static void loop_ki (const KernelVariables& kv, const int klim,
                       const int ilim, const Fn& g) {
    using Kokkos::parallel_for;
    using Kokkos::TeamThreadRange;
    using Kokkos::ThreadVectorRange;

    if (OnGpu<ExecSpace>::value) {
      const auto tr = TeamThreadRange  (kv.team, klim);
      const auto vr = ThreadVectorRange(kv.team, ilim);
      const auto f = [&] (const int k) {
        const auto h = [&] (const int i) { g(k,i); };
        parallel_for(vr, h);
      };
      parallel_for(tr, f);
    } else if (kv.team.team_size() == 1) {
      for (int k = 0; k < klim; ++k)
        for (int i = 0; i < ilim; ++i)
          g(k,i);
    } else {
      const auto tr = TeamThreadRange  (kv.team, klim);
      const auto f = [&] (const int k) {
        for (int i = 0; i < ilim; ++i)
          g(k,i);
      };
      parallel_for(tr, f);
    }
  }

  // Format of rest of Hxx -> DIRK Newton iteration format.
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void transpose (const KernelVariables& kv, const int nlev,
                         const View& src, const WorkSlot& dst,
                         typename std::enable_if<View::rank == 3>::type* = 0) {
    assert(src.extent_int(2)*packn >= nlev);
    assert(src.extent_int(0) == NP && src.extent_int(1) == NP);
    const auto f = [&] (const int k) {
      const auto
      pk = k / packn,
      sk = k % packn;
      const auto g = [&] (const int i) {
        const auto gk0 = packn*i;
        for (int s = 0; s < packn; ++s) {
          const auto
          gk = gk0 + s,
          gi = gk / NP,
          gj = gk % NP;
          if (scaln % packn != 0 && // try to compile out this conditional when possible
              gk >= scaln) break;
          dst(k,i)[s] = src(gi,gj,pk)[sk];
        }
      };
      const int n = npack;
      const auto p = Kokkos::ThreadVectorRange(kv.team, n);
      Kokkos::parallel_for(p, g);
    };    
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, nlev), f);
  }

  // DIRK Newton iteration format -> format of rest of Hxx.
  template <typename View>
  KOKKOS_INLINE_FUNCTION
  static void transpose (const KernelVariables& kv, const int nlev,
                         const WorkSlot& src, const View& dst,
                         typename std::enable_if<View::rank == 3>::type* = 0) {
    assert(dst.extent_int(2)*packn >= nlev);
    assert(dst.extent_int(0) == NP && dst.extent_int(1) == NP);
    const auto f = [&] (const int idx) {
      const auto
      gi = idx / NP,
      gj = idx % NP,
      pi = idx / packn,
      si = idx % packn;
      const auto g = [&] (const int pk) {
        const auto k0 = pk*packn;
        // If there is a remainder at the end, we nonetheless transfer these
        // unused data to avoid a conditional and runtime loop limit.
        for (int sk = 0; sk < packn; ++sk)
          dst(gi,gj,pk)[sk] = src(k0+sk,pi)[si];
      };
      const auto p = Kokkos::ThreadVectorRange(kv.team, dst.extent_int(2));
      Kokkos::parallel_for(p, g);
    };    
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP*NP), f);
  }

  void init_boundary_exchanges();

  void run();

  void calc_trajectory();
};

} // namespace Homme

#endif // HOMMEXX_COMPOSE_TRANSPORT_IMPL_HPP
