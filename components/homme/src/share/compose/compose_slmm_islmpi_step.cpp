#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

// dep_points is const in principle, but if lev <=
// semi_lagrange_nearest_point_lev, a departure point may be altered if the
// winds take it outside of the comm halo.
template <typename MT>
void step (
  IslMpi<MT>& cm, const Int nets, const Int nete,
  Real* dep_points_r,           // dep_points(1:3, 1:np, 1:np)
  Real* q_min_r, Real* q_max_r) // q_{min,max}(1:np, 1:np, lev, 1:qsize, ie-nets+1)
{
  slmm_assert(cm.np == 4);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

  const DepPointsH<MT> dep_points_h(dep_points_r, cm.nelemd, cm.nlev, cm.np2, 3);
#ifdef COMPOSE_PORT_DEV_VIEWS
  const DepPoints<MT> dep_points("dep_points", cm.nelemd, cm.nlev, cm.np2, 3);
  ko::deep_copy(dep_points, dep_points_h);
#else
  const DepPoints<MT> dep_points = dep_points_h;
#endif

  const Int nelem = nete - nets + 1;
  const FA4<Real>
    q_min(q_min_r, cm.np2, cm.nlev, cm.qsize, nelem),
    q_max(q_max_r, cm.np2, cm.nlev, cm.qsize, nelem);

  // Partition my elements that communicate with remotes among threads, if I
  // haven't done that yet.
  if (cm.mylid_with_comm_tid_ptr_h.capacity() == 0)
    init_mylid_with_comm_threaded(cm, nets, nete);
  // Set up to receive departure point requests from remotes.
  setup_irecv(cm);
  // Determine where my departure points are, and set up requests to remotes as
  // well as to myself to fulfill these.
  analyze_dep_points(cm, nets, nete, dep_points);
  pack_dep_points_sendbuf_pass1(cm);
  pack_dep_points_sendbuf_pass2(cm, dep_points);
  // Send requests.
  isend(cm);
  // While waiting, compute q extrema in each of my elements.
  calc_q_extrema(cm, nets, nete);
  // Wait for the departure point requests. Since this requires a thread
  // barrier, at the same time make sure the send buffer is free for use.
  recv_and_wait_on_send(cm);
  // Compute the requested q for departure points from remotes.
  calc_rmt_q(cm);
  // Send q data.
  isend(cm, true /* want_req */, true /* skip_if_empty */);
  // Set up to receive q for each of my departure point requests sent to
  // remotes. We can't do this until the OpenMP barrier in isend assures that
  // all threads are done with the receive buffer's departure points.
  setup_irecv(cm, true /* skip_if_empty */);
  // While waiting to get my data from remotes, compute q for departure points
  // that have remained in my elements.
  calc_own_q(cm, nets, nete, dep_points, q_min, q_max);
  // Receive remote q data and use this to fill in the rest of my fields.
  recv(cm, true /* skip_if_empty */);
  copy_q(cm, nets, q_min, q_max);
  // Wait on send buffer so it's free to be used by others.
  wait_on_send(cm, true /* skip_if_empty */);

#ifdef COMPOSE_PORT_DEV_VIEWS
  ko::deep_copy(dep_points_h, dep_points);
#endif
}

template void step(IslMpi<slmm::MachineTraits>&, const Int, const Int, Real*, Real*, Real*);

} // namespace islmpi
} // namespace homme
