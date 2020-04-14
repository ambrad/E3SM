#include "compose_slmm_islmpi.hpp"

namespace homme {
namespace islmpi {

/* Pack the departure points (x). We use two passes. We also set up the q
   metadata. Two passes let us do some efficient tricks that are not available
   with one pass. Departure point and q messages are formatted as follows:
    xs: (#x-in-rank         int                                     <-
         x-bulk-data-offset i                                        |
         (lid-on-rank       i     only packed if #x in lid > 0       |
          #x-in-lid         i     > 0                                |- meta data
          (lev              i     only packed if #x in (lid,lev) > 0 |
           #x)              i     > 0                                |
              *#lev) *#lid                                          <-
         x                  3 real                                  <-- bulk data
          *#x-in-rank) *#rank
    qs: (q-extrema    2 qsize r   (min, max) packed together
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
    auto&& sendbuf = cm.sendbuf_meta_h(ri);
    const auto&& lid_on_rank = cm.lid_on_rank(ri);
    // metadata offset, x bulk data offset, q bulk data offset
    Int mos = 0, xos = 0, qos = 0, sendcount = 0, cnt;
    cnt = setbuf(sendbuf, mos, 0, 0); // empty space for later
    mos += cnt;
    sendcount += cnt;
    if (cm.nx_in_rank(ri) == 0) {
      setbuf(sendbuf, 0, 0, mos);
      cm.x_bulkdata_offset_h(ri) = mos;
      cm.sendcount_h(ri) = sendcount;
      continue;
    }
    auto&& bla = cm.bla(ri);
    for (Int lidi = 0, lidn = cm.lid_on_rank(ri).n(); lidi < lidn; ++lidi) {
      auto nx_in_lid = cm.nx_in_lid(ri,lidi);
      if (nx_in_lid == 0) continue;
      cnt = setbuf(sendbuf, mos, lid_on_rank(lidi), nx_in_lid);
      mos += cnt;
      sendcount += cnt;
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
        const auto dos = setbuf(sendbuf, mos, lev, nx);
        mos += dos;
        sendcount += dos + 3*nx;
        t.xptr = xos;
        xos += 3*nx;
        qos += 2 + nx;
        nx_in_lid -= nx;
      }
      slmm_assert(nx_in_lid == 0);
    }
    setbuf(sendbuf, 0, cm.nx_in_rank(ri), mos /* offset to x bulk data */);
    cm.x_bulkdata_offset_h(ri) = mos;
    cm.sendcount_h(ri) = sendcount;
  }
#ifdef COMPOSE_PORT_SEPARATE_VIEWS
  // Copy metadata chunks to device sendbuf.
  deep_copy(cm.x_bulkdata_offset, cm.x_bulkdata_offset_h);
  deep_copy(cm.sendcount, cm.sendcount_h);
  assert(cm.sendbuf.n() == nrmtrank);
  Int os = 0;
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto n = cm.x_bulkdata_offset_h(ri);
    assert(n <= cm.sendmetasz[ri]);
    if (n > 0)
      ko::deep_copy(ko::View<Real*, typename MT::DES>(cm.sendbuf.data() + os, n),
                    ko::View<Real*, typename MT::HES>(cm.sendbuf_meta_h(ri).data(), n));
    os += cm.sendsz[ri];
  }
#endif
}

template <typename MT>
void pack_dep_points_sendbuf_pass2 (IslMpi<MT>& cm, const DepPoints<MT>& dep_points) {
  const auto myrank = cm.p->rank();
#ifdef COMPOSE_PORT
  const Int start = 0, end = cm.mylid_with_comm_d.n();
#else
  const int tid = get_tid();
  const Int
    start = cm.mylid_with_comm_tid_ptr_h(tid),
    end = cm.mylid_with_comm_tid_ptr_h(tid+1);
#endif
  {
    auto ed = cm.ed_d;
    ko::parallel_for(
      ko::RangePolicy<typename MT::DES>(start, end),
      KOKKOS_LAMBDA (const Int& ptr) {
        const Int tci = cm.mylid_with_comm_d(ptr);
        ed(tci).rmt.clear();
      });
  }
  {
    const Int np2 = cm.np2, nlev = cm.nlev, qsize = cm.qsize;
    const auto ed_d = cm.ed_d;
    const auto mylid_with_comm_d = cm.mylid_with_comm_d;
    const auto sendbuf = cm.sendbuf;
    const auto x_bulkdata_offset = cm.x_bulkdata_offset;
    const auto bla = cm.bla;
    const auto f = KOKKOS_LAMBDA (const Int& ki) {
      const Int ptr = start + ki/(nlev*np2);
      const Int lev = (ki/np2) % nlev;
      const Int k = ki % np2;
      const Int tci = mylid_with_comm_d(ptr);
      auto& ed = ed_d(tci);
      const Int sci = ed.src(lev,k);
      const auto& nbr = ed.nbrs(sci);
      if (nbr.rank == myrank) return;
      const Int ri = nbr.rank_idx;
      const Int lidi = nbr.lid_on_rank_idx;
      auto&& sb = sendbuf(ri);
#ifdef COMPOSE_HORIZ_OPENMP
      omp_lock_t* lock;
      if (cm.horiz_openmp) {
        lock = &cm.ri_lidi_locks(ri,lidi);
        omp_set_lock(lock);
      }
#endif
      Int xptr, qptr, cnt; {
        auto& t = bla(ri,lidi,lev);
#ifdef COMPOSE_PORT
        cnt = ko::atomic_fetch_add(static_cast<volatile Int*>(&t.cnt), 1);
#else
        cnt = t.cnt;
        ++t.cnt;
#endif
        qptr = t.qptr;
        xptr = x_bulkdata_offset(ri) + t.xptr + 3*cnt;
      }
#ifdef COMPOSE_HORIZ_OPENMP
      if (cm.horiz_openmp) omp_unset_lock(lock);
#endif
      slmm_kernel_assert_high(xptr > 0);
      for (Int i = 0; i < 3; ++i)
        sb(xptr + i) = dep_points(tci,lev,k,i);
      auto& item = ed.rmt.atomic_inc_and_return_next();
      item.q_extrema_ptr = qsize * qptr;
      item.q_ptr = item.q_extrema_ptr + qsize*(2 + cnt);
      item.lev = lev;
      item.k = k;
    };
    ko::parallel_for(
      ko::RangePolicy<typename MT::DES>(0, (end - start)*nlev*np2), f);
  }
}

template void pack_dep_points_sendbuf_pass1(IslMpi<slmm::MachineTraits>& cm);
template void pack_dep_points_sendbuf_pass2(IslMpi<slmm::MachineTraits>& cm,
                                            const DepPoints<slmm::MachineTraits>& dep_points);

} // namespace islmpi
} // namespace homme
