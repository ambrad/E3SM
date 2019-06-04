#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {
// mylid_with_comm(rankidx) is a list of element LIDs that have relations with
// other elements on other ranks. For horizontal threading, need to find the
// subsets that fit within the usual horizontal-threading nets:nete ranges.
template <typename MT>
void init_mylid_with_comm_threaded (IslMpi<MT>& cm, const Int& nets, const Int& nete) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    const int nthr = get_num_threads();
    cm.rwork = typename IslMpi<MT>::template Array<Real**>("rwork", nthr, cm.qsize);
    cm.mylid_with_comm_tid_ptr.reset_capacity(nthr+1, true);
    cm.horiz_openmp = get_num_threads() > 1;
  }
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
  const int tid = get_tid();
  const auto& beg = std::lower_bound(cm.mylid_with_comm.begin(),
                                     cm.mylid_with_comm.end(), nets);
  slmm_assert(cm.p->size() == 1 || beg != cm.mylid_with_comm.end());
  cm.mylid_with_comm_tid_ptr(tid) = beg - cm.mylid_with_comm.begin();
  if (tid == cm.mylid_with_comm_tid_ptr.n() - 2) {
    const auto& end = std::lower_bound(cm.mylid_with_comm.begin(),
                                       cm.mylid_with_comm.end(), nete+1);
    cm.mylid_with_comm_tid_ptr(tid+1) = end - cm.mylid_with_comm.begin();
  }
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

template <typename MT>
void setup_irecv (IslMpi<MT>& cm, const bool skip_if_empty) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp master
#endif
  {
    const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
    cm.recvreq.clear();
    for (Int ri = 0; ri < nrmtrank; ++ri) {
      if (skip_if_empty && cm.nx_in_rank(ri) == 0) continue;
      auto&& recvbuf = cm.recvbuf(ri);
      // The count is just the number of slots available, which can be larger
      // than what is actually being received.
      cm.recvreq.inc();
      mpi::irecv(*cm.p, recvbuf.data(), recvbuf.n(), cm.ranks(ri), 42,
                 &cm.recvreq.back());
    }
  }
}

template <typename MT>
void isend (IslMpi<MT>& cm, const bool want_req , const bool skip_if_empty) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    slmm_assert( ! (skip_if_empty && want_req));
    const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
    for (Int ri = 0; ri < nrmtrank; ++ri) {
      if (skip_if_empty && cm.sendcount(ri) == 0) continue;
      mpi::isend(*cm.p, cm.sendbuf(ri).data(), cm.sendcount(ri),
                 cm.ranks(ri), 42, want_req ? &cm.sendreq(ri) : nullptr);
    }
  }
}

template <typename MT>
void recv_and_wait_on_send (IslMpi<MT>& cm) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp master
#endif
  {
    mpi::waitall(cm.sendreq.n(), cm.sendreq.data());
    mpi::waitall(cm.recvreq.n(), cm.recvreq.data());
  }
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

template <typename MT>
void recv (IslMpi<MT>& cm, const bool skip_if_empty) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp master
#endif
  {
    mpi::waitall(cm.recvreq.n(), cm.recvreq.data());
  }
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

template void init_mylid_with_comm_threaded(
  IslMpi<slmm::MachineTraits>& cm, const Int& nets, const Int& nete);
template void setup_irecv(IslMpi<slmm::MachineTraits>& cm, const bool skip_if_empty);
template void isend(IslMpi<slmm::MachineTraits>& cm, const bool want_req,
                    const bool skip_if_empty);
template void recv_and_wait_on_send(IslMpi<slmm::MachineTraits>& cm);
template void recv(IslMpi<slmm::MachineTraits>& cm, const bool skip_if_empty);

} // namespace islmpi
} // namespace homme
