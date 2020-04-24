#include "compose_slmm_islmpi.hpp"

#include "gptl.h"

struct Timer {
  Timer (const std::string& name_) : name("SLMM_csl_" + name_) { GPTLstart(name.c_str()); }
  ~Timer () { GPTLstop(name.c_str()); }
private:
  const std::string name;
};

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

  const DepPointsH<MT> dep_points_h(dep_points_r, cm.nelemd, cm.nlev, cm.np2);
  const QExtremaH<MT>
    q_min_h(q_min_r, cm.nelemd, cm.qsize, cm.nlev, cm.np2),
    q_max_h(q_max_r, cm.nelemd, cm.qsize, cm.nlev, cm.np2);
#ifdef COMPOSE_PORT_DEV_VIEWS
  const DepPoints<MT> dep_points("dep_points", cm.nelemd, cm.nlev, cm.np2);
  const QExtrema<MT>
    q_min("q_min", cm.nelemd, cm.qsize, cm.nlev, cm.np2),
    q_max("q_max", cm.nelemd, cm.qsize, cm.nlev, cm.np2);
  ko::deep_copy(dep_points, dep_points_h);
#else
  const auto dep_points = dep_points_h;
  const auto q_min = q_min_h;
  const auto q_max = q_max_h;
#endif

  // Partition my elements that communicate with remotes among threads, if I
  // haven't done that yet.
  { Timer t("01_mylid");
    if (cm.mylid_with_comm_tid_ptr_h.capacity() == 0)
      init_mylid_with_comm_threaded(cm, nets, nete); }
  // Set up to receive departure point requests from remotes.
  { Timer t("02_setup_irecv");
    setup_irecv(cm); }
  // Determine where my departure points are, and set up requests to remotes as
  // well as to myself to fulfill these.
  { Timer t("03_adp");
    analyze_dep_points(cm, nets, nete, dep_points); }
  { Timer t("04_pack_pass1");
    pack_dep_points_sendbuf_pass1(cm); }
  { Timer t("05_pack_pass2");
    pack_dep_points_sendbuf_pass2(cm, dep_points); }
  // Send requests.
  { Timer t("06_isend");
    isend(cm); }
  // While waiting, compute q extrema in each of my elements.
  { Timer t("07_q_extrema");
    calc_q_extrema(cm, nets, nete); }
  // Wait for the departure point requests. Since this requires a thread
  // barrier, at the same time make sure the send buffer is free for use.
  { Timer t("08_recv_and_wait");
    recv_and_wait_on_send(cm); }
  // Compute the requested q for departure points from remotes.
  { Timer t("09_rmt_q");
    calc_rmt_q(cm); }
  // Send q data.
  { Timer t("10_isend");
    isend(cm, true /* want_req */, true /* skip_if_empty */); }
  // Set up to receive q for each of my departure point requests sent to
  // remotes. We can't do this until the OpenMP barrier in isend assures that
  // all threads are done with the receive buffer's departure points.
  { Timer t("11_setup_irecv");
    setup_irecv(cm, true /* skip_if_empty */); }
  // While waiting to get my data from remotes, compute q for departure points
  // that have remained in my elements.
  { Timer t("12_own_q");
    calc_own_q(cm, nets, nete, dep_points, q_min, q_max); }
  // Receive remote q data and use this to fill in the rest of my fields.
  { Timer t("13_recv");
    recv(cm, true /* skip_if_empty */); }
  { Timer t("14_copy_q");
    copy_q(cm, nets, q_min, q_max); }
  // Wait on send buffer so it's free to be used by others.
  { Timer t("15_wait_on_send");
    wait_on_send(cm, true /* skip_if_empty */); }

#ifdef COMPOSE_PORT_DEV_VIEWS
  ko::fence();
  ko::deep_copy(dep_points_h, dep_points);
  ko::deep_copy(q_min_h, q_min);
  ko::deep_copy(q_max_h, q_max);
#endif
}

template void step(IslMpi<slmm::MachineTraits>&, const Int, const Int, Real*, Real*, Real*);

} // namespace islmpi
} // namespace homme
