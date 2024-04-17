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
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp for
#endif
  for (Int it = 0; it < cm.nrmt_xs; ++it) {
    const Int
      ri = cm.rmt_xs_h(5*it), lid = cm.rmt_xs_h(5*it + 1), lev = cm.rmt_xs_h(5*it + 2),
      xos = cm.rmt_xs_h(5*it + 3), vos = 3*cm.rmt_xs_h(5*it + 4);
    const auto&& xs = cm.recvbuf(ri);
    auto&& v = cm.sendbuf(ri);
    //calc_velocity<np>(cm, lid, lev, &xs(xos), &v(vos));
  }
}

template <typename MT>
void traj_calc_own_next_step (IslMpi<MT>& cm, const Int nets, const Int nete,
                              const DepPoints<MT>& dep_points, Trajectory& t) {
}

template <typename MT>
void traj_copy_next_step (IslMpi<MT>& cm, Trajectory& t) {
}

template <typename MT>
void calc_trajectory (IslMpi<MT>& cm, const Int nets, const Int nete,
                      const Int step, const Real dtsub, Real* v01_r,
                      const Real* v1gradv0_r, const Real* dep_points_r)
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

  CA5<Real> v01(v01_r, cm.nelemd, 2, 3, cm.np2, cm.nlev);
  CA4<const Real> v1gradv0(v1gradv0_r, cm.nelemd, 3, cm.np2, cm.nlev);

  if (step == 0) {
    for (int ie = nets; ie <= nete; ++ie)
      for (int d = 0; d < 3; ++d)
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
  traj_calc_own_next_step(cm, nets, nete, dep_points, t);
  recv(cm, true /* skip_if_empty */);
  traj_copy_next_step(cm, t);
  wait_on_send(cm, true /* skip_if_empty */);
}

template void calc_trajectory(IslMpi<ko::MachineTraits>&, const Int, const Int,
                              const Int, const Real, Real*, const Real*, const Real*);

} // namespace islmpi
} // namespace homme
