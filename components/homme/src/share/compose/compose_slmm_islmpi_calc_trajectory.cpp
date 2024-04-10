#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

template <typename T> using CA4 = ko::View<T****,  ko::LayoutRight, ko::HostSpace>;
template <typename T> using CA5 = ko::View<T*****, ko::LayoutRight, ko::HostSpace>;

struct Trajectory {
  Real dtsub;
  CA5<Real> v01;
  const CA4<const Real> v1gradv0;
};

template <typename MT>
void traj_calc_rmt_next_step (IslMpi<MT>& cm, Trajectory& t) {
  calc_rmt_q_pass1(cm, true);
}

template <typename MT>
void calc_trajectory (IslMpi<MT>& cm, const Int nets, const Int nete,
                      const Int step, const Real dtsub, Real* v01_r,
                      const Real* v1gradv0_r, Real* dep_points_r)
{
  using slmm::Timer;

  slmm_assert(cm.np == 4);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

#ifdef COMPOSE_PORT
  auto& dep_points = cm.tracer_arrays->dep_points;
#else
  DepPointsH<MT> dep_points(dep_points_r, cm.nelemd, cm.nlev, cm.np2);
#endif

  CA5<Real> v01(v01_r, cm.nelemd, 2, 2, cm.np2, cm.nlev);
  CA4<const Real> v1gradv0(v1gradv0_r, cm.nelemd, 2, cm.np2, cm.nlev);

  if (step == 0) {
    for (int ie = nets; ie <= nete; ++ie)
      for (int d = 0; d < 2; ++d)
        for (int k = 0; k < cm.np2; ++k)
          for (int lev = 0; lev < cm.nlev; ++lev)
            v01(ie,0,d,k,lev) = ((v01(ie,0,d,k,lev) + v01(ie,1,d,k,lev))/2 -
                                 (dtsub/2)*v1gradv0(ie,d,k,lev));
    return;
  }

  // See comments in homme::islmpi::step for details. Each substep follows
  // essentially the same pattern.
  Trajectory t{dtsub, v01, v1gradv0};
  if (cm.mylid_with_comm_tid_ptr_h.capacity() == 0)
    init_mylid_with_comm_threaded(cm, nets, nete);
  setup_irecv(cm);
  analyze_dep_points(cm, nets, nete, dep_points);
  pack_dep_points_sendbuf_pass1(cm, true /* trajectory */);
  pack_dep_points_sendbuf_pass2(cm, dep_points, true /* trajectory */);
  isend(cm);
  recv_and_wait_on_send(cm);
  traj_calc_rmt_next_step(cm, t);
  isend(cm, true /* want_req */, true /* skip_if_empty */);
  setup_irecv(cm, true /* skip_if_empty */);
  //traj_calc_own_next_step(cm, nets, nete, t);
  recv(cm, true /* skip_if_empty */);
  //traj_copy_dep_points(cm, dep_points);
  wait_on_send(cm, true /* skip_if_empty */);
}

template void calc_trajectory(IslMpi<ko::MachineTraits>&, const Int, const Int,
                              const Int, const Real, Real*, const Real*, Real*);

} // namespace islmpi
} // namespace homme
