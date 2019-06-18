#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

/* Pack the departure points (x). We use two passes. We also set up the q
   metadata. Two passes lets us do some efficient tricks that are not available
   with one pass. Departure point and q messages are formatted as follows:
    xs: (#x-in-rank    int
         pad           i
         (lid-on-rank  i     only packed if #x in lid > 0
          #x-in-lid    i     > 0
          (lev         i     only packed if #x in (lid,lev) > 0
           #x          i     > 0
           x         3 real
            *#x) *#lev) *#lid) *#rank
    qs: (q-extrema    2 qsize r    (min, max) packed together
         q              qsize r
          *#x) *#lev *#lid *#rank
 */
template <typename MT>
void pack_dep_points_sendbuf_pass1 (IslMpi<MT>& cm) {
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp for
#endif
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    auto&& sendbuf = cm.sendbuf(ri);
    const auto&& lid_on_rank = cm.lid_on_rank(ri);
    Int xos = 0, qos = 0;
    xos += setbuf(sendbuf, xos, cm.nx_in_rank(ri), 0 /* empty space for alignment */);
    if (cm.nx_in_rank(ri) == 0) {
      cm.sendcount(ri) = xos;
      continue;
    }
    auto&& bla = cm.bla(ri);
    for (Int lidi = 0, lidn = cm.lid_on_rank(ri).n(); lidi < lidn; ++lidi) {
      auto nx_in_lid = cm.nx_in_lid(ri,lidi);
      if (nx_in_lid == 0) continue;
      xos += setbuf(sendbuf, xos, lid_on_rank(lidi), nx_in_lid);
      for (Int lev = 0; lev < cm.nlev; ++lev) {
        auto& t = bla(lidi,lev);
        t.qptr = qos;
        slmm_assert_high(t.cnt == 0);
        const Int nx = t.xptr;
        if (nx == 0) {
          t.xptr = -1;
          continue;
        }
        slmm_assert_high(nx > 0);
        const auto dos = setbuf(sendbuf, xos, lev, nx);
        t.xptr = xos + dos;
        xos += dos + 3*nx;
        qos += 2 + nx;
        nx_in_lid -= nx;
      }
      slmm_assert(nx_in_lid == 0);
    }
    cm.sendcount(ri) = xos;
  }
}

template <typename MT>
void pack_dep_points_sendbuf_pass2 (IslMpi<MT>& cm, const FA4<const Real>& dep_points) {
  const auto myrank = cm.p->rank();
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr_h(tid),
           end = cm.mylid_with_comm_tid_ptr_h(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm_h(ptr);
    auto& ed = cm.ed_d(tci);
    ed.rmt.clear();
    for (Int lev = 0; lev < cm.nlev; ++lev) {
      for (Int k = 0; k < cm.np2; ++k) {
        const Int sci = ed.src(lev,k);
        const auto& nbr = ed.nbrs(sci);
        if (nbr.rank == myrank) continue;
        const Int ri = nbr.rank_idx;
        const Int lidi = nbr.lid_on_rank_idx;
        auto&& sb = cm.sendbuf(ri);
#ifdef COMPOSE_HORIZ_OPENMP
        omp_lock_t* lock;
        if (cm.horiz_openmp) {
          lock = &cm.ri_lidi_locks(ri,lidi);
          omp_set_lock(lock);
        }
#endif
        Int xptr, qptr, cnt; {
          auto& t = cm.bla(ri,lidi,lev);
#ifdef COMPOSE_PORT
          cnt = ko::atomic_fetch_add(static_cast<volatile Int*>(&t.cnt), 1);
#else
          cnt = t.cnt;
          ++t.cnt;
#endif
          qptr = t.qptr;
          xptr = t.xptr + 3*cnt;
        }
#ifdef COMPOSE_HORIZ_OPENMP
        if (cm.horiz_openmp) omp_unset_lock(lock);
#endif
        slmm_assert_high(xptr > 0);
        for (Int i = 0; i < 3; ++i)
          sb(xptr + i) = dep_points(i,k,lev,tci);
        auto& item = ed.rmt.atomic_inc_and_return_next();
        item.q_extrema_ptr = cm.qsize * qptr;
        item.q_ptr = item.q_extrema_ptr + cm.qsize*(2 + cnt);
        item.lev = lev;
        item.k = k;
      }
    }
  }
}

template void pack_dep_points_sendbuf_pass1(IslMpi<slmm::MachineTraits>& cm);
template void pack_dep_points_sendbuf_pass2(IslMpi<slmm::MachineTraits>& cm,
                                            const FA4<const Real>& dep_points);

} // namespace islmpi
} // namespace homme
