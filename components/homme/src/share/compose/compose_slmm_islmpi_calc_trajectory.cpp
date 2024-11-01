#include "compose_slmm_islmpi.hpp"
#include "compose_slmm_islmpi_interpolate.hpp"
#include "compose_slmm_islmpi_buf.hpp"

namespace homme {
namespace islmpi {

template <typename T> using CA4 = ko::View<T****, ko::LayoutRight, ko::HostSpace>;

// vnode and vdep are indexed as (ie,lev,k,dim), On entry, vnode contains nodal
// velocity data. These data are used to provide updates at departure points for
// both own and remote departure points, writing to vdep. dim = 0:2 is for the
// 3D Cartesian representation of the horizontal velocity; dim = 3 is for
// eta_dot.
struct Trajectory {
  CA4<const Real> vnode;
  CA4<Real> vdep;
};

template <Int np, typename MT> SLMM_KIF
void calc_v (const IslMpi<MT>& cm, const Trajectory& t,
             const Int src_lid, const Int lev,
             const Real* const dep_point, Real* const v_tgt) {
  Real ref_coord[2]; {
    const auto& m = cm.advecter->local_mesh(src_lid);
    cm.advecter->s2r().calc_sphere_to_ref(src_lid, m, dep_point,
                                          ref_coord[0], ref_coord[1]);
  }

  Real rx[np], ry[np];
  interpolate<MT>(cm.advecter->alg(), ref_coord, rx, ry);

  if (not cm.traj_3d) {
    slmm_assert(cm.dep_points_ndim == 3);
    for (int d = 0; d < cm.dep_points_ndim; ++d) {
      Real vel_nodes[np*np];
      for (int k = 0; k < np*np; ++k)
        vel_nodes[k] = t.vnode(src_lid,lev,k,d);
      v_tgt[d] = calc_q_tgt(rx, ry, vel_nodes);
    }
    return;
  }

  slmm_assert(cm.dep_points_ndim == 4);
  slmm_assert(dep_point[3] > 0 && dep_point[3] < 1);

  // Search for the eta midpoint values that support the departure point's eta
  // value.
  const auto eta_dep = dep_point[3];
  Int lev_dep = lev;
  if (eta_dep != cm.etam(lev)) {
    if (eta_dep < cm.etam(lev)) {
      for (lev_dep = lev-1; lev_dep >= 0; --lev_dep)
        if (eta_dep >= cm.etam(lev_dep))
          break;
    } else {
      for (lev_dep = lev; lev_dep < cm.nlev-1; ++lev_dep)
        if (eta_dep < cm.etam(lev_dep+1))
          break;
    }
  }
  slmm_assert(lev_dep >= -1 && lev_dep < cm.nlev);
  slmm_assert(lev_dep == -1 || eta_dep >= cm.etam(lev_dep));
  Real a;
  bool bdy = false;
  if (lev_dep == -1) {
    lev_dep = 0;
    a = 0;
    bdy = true;
  } else if (lev_dep == cm.nlev-1) {
    a = 0;
    bdy = true;
  } else {
    a = ((eta_dep - cm.etam(lev_dep)) /
         (cm.etam(lev_dep+1) - cm.etam(lev_dep)));
  }
  // Linear interp coefficients.
  const Real alpha[] = {1-a, a};

  for (int d = 0; d < 4; ++d)
    v_tgt[d] = 0;
  for (int i = 0; i < 2; ++i) {
    if (alpha[i] == 0) continue;
    for (int d = 0; d < 4; ++d) {
      Real vel_nodes[np*np];
      for (int k = 0; k < np*np; ++k)
        vel_nodes[k] = t.vnode(src_lid,lev_dep+i,k,d);
      v_tgt[d] += alpha[i]*calc_q_tgt(rx, ry, vel_nodes);
    }
  }
  // Treat eta_dot specially since eta_dot goes to 0 at the boundaries.
  if (bdy) {
    if (lev_dep == 0)
      v_tgt[3] *= eta_dep/cm.etam(0);
    else
      v_tgt[3] *= (1 - eta_dep)/(1 - cm.etam(cm.nlev-1));
  }
}

template <int np, typename MT>
void traj_calc_rmt_next_step (IslMpi<MT>& cm, Trajectory& t) {
  calc_rmt_q_pass1(cm, true);
  const auto ndim = cm.dep_points_ndim;
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp for
#endif
  for (Int it = 0; it < cm.nrmt_xs; ++it) {
    const Int
      ri = cm.rmt_xs_h(5*it), lid = cm.rmt_xs_h(5*it + 1), lev = cm.rmt_xs_h(5*it + 2),
      xos = cm.rmt_xs_h(5*it + 3), vos = ndim*cm.rmt_xs_h(5*it + 4);
    const auto&& xs = cm.recvbuf(ri);
    auto&& v = cm.sendbuf(ri);
    calc_v<np>(cm, t, lid, lev, &xs(xos), &v(vos));
  }
}

template <int np, typename MT>
void traj_calc_own_next_step (IslMpi<MT>& cm, const DepPoints<MT>& dep_points,
                              Trajectory& t) {
  const auto ndim = cm.dep_points_ndim;
#ifdef COMPOSE_PORT
  const auto& ed_d = cm.ed_d;
  const auto& own_dep_list = cm.own_dep_list;
  const auto f = COMPOSE_LAMBDA (const Int& it) {
    const Int tci = own_dep_list(it,0);
    const Int tgt_lev = own_dep_list(it,1);
    const Int tgt_k = own_dep_list(it,2);
    const auto& ed = ed_d(tci);
    const Int slid = ed.nbrs(ed.src(tgt_lev, tgt_k)).lid_on_rank;
    Real v_tgt[4];
    calc_v<np>(cm, t, slid, tgt_lev, &dep_points(tci,tgt_lev,tgt_k,0), v_tgt);
    for (int d = 0; d < ndim; ++d)
      t.vdep(tci,tgt_lev,tgt_k,d) = v_tgt[d];
  };
  ko::parallel_for(
    ko::RangePolicy<typename MT::DES>(0, cm.own_dep_list_len), f);
#else
  const int tid = get_tid();
  for (Int tci = 0; tci < cm.nelemd; ++tci) {
    auto& ed = cm.ed_d(tci);
    const Int ned = ed.own.n();
#ifdef COMPOSE_HORIZ_OPENMP
#   pragma omp for
#endif
    for (Int idx = 0; idx < ned; ++idx) {
      const auto& e = ed.own(idx);
      const Int slid = ed.nbrs(ed.src(e.lev, e.k)).lid_on_rank;
      Real v_tgt[4];
      calc_v<np>(cm, t, slid, e.lev, &dep_points(tci,e.lev,e.k,0), v_tgt);
      for (int d = 0; d < ndim; ++d)
        t.vdep(tci,e.lev,e.k,d) = v_tgt[d];
    }
  }
#endif
}

template <typename MT>
void traj_copy_next_step (IslMpi<MT>& cm, Trajectory& t) {
  const auto myrank = cm.p->rank();
  const auto ndim = cm.dep_points_ndim;
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr_h(tid),
           end = cm.mylid_with_comm_tid_ptr_h(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm_d(ptr);
    auto& ed = cm.ed_d(tci);
    for (const auto& e: ed.rmt) {
      slmm_assert(ed.nbrs(ed.src(e.lev, e.k)).rank != myrank);
      const Int ri = ed.nbrs(ed.src(e.lev, e.k)).rank_idx;
      const auto&& recvbuf = cm.recvbuf(ri);
      for (int d = 0; d < ndim; ++d)
        t.vdep(tci,e.lev,e.k,d) = recvbuf(e.q_ptr + d);
    }
  }
}

template <typename MT> void
calc_v_departure (IslMpi<MT>& cm, const Int nets, const Int nete,
                  const Int step, const Real dtsub,
                  Real* dep_points_r, const Real* vnode_r, Real* vdep_r)
{
  const int np = 4;

  slmm_assert(cm.np == np);
#ifdef COMPOSE_PORT
  slmm_assert(nets == 0 && nete+1 == cm.nelemd);
#endif

  const auto ndim = cm.dep_points_ndim;

#ifdef COMPOSE_PORT_TODO
#else
# pragma message "TODO"
  CA4<const Real> vnode(vnode_r, cm.nelemd, cm.nlev, cm.np2, ndim);
  CA4<      Real> vdep (vdep_r , cm.nelemd, cm.nlev, cm.np2, ndim);
#endif

  if (step == 0) {
    // The departure points are at the nodes. No interpolation is needed.
    for (Int ie = nets; ie <= nete; ++ie)
      for (Int lev = 0; lev < cm.nlev; ++lev)
        for (Int k = 0; k < cm.np2; ++k)
          for (Int d = 0; d < ndim; ++d)
            vdep(ie,lev,k,d) = vnode(ie,lev,k,d);
    return;
  }

#ifdef COMPOSE_PORT
  auto& dep_points = cm.tracer_arrays->dep_points;
#else
  DepPointsH<MT> dep_points(dep_points_r, cm.nelemd, cm.nlev, cm.np2, ndim);
#endif
  slmm_assert(dep_points.extent_int(3) == ndim);

  // See comments in homme::islmpi::step for details. Each substep follows
  // essentially the same pattern.
  Trajectory t{vnode, vdep};
  if (cm.mylid_with_comm_tid_ptr_h.capacity() == 0)
    init_mylid_with_comm_threaded(cm, nets, nete);
  setup_irecv(cm);
  analyze_dep_points(cm, nets, nete, dep_points);
  pack_dep_points_sendbuf_pass1(cm, true /* trajectory */);
  pack_dep_points_sendbuf_pass2(cm, dep_points, true /* trajectory */);
  isend(cm);
  recv_and_wait_on_send(cm);
  traj_calc_rmt_next_step<np>(cm, t);
  isend(cm, true /* want_req */, true /* skip_if_empty */);
  setup_irecv(cm, true /* skip_if_empty */);
  traj_calc_own_next_step<np>(cm, dep_points, t);
  recv(cm, true /* skip_if_empty */);
  traj_copy_next_step(cm, t);
  wait_on_send(cm, true /* skip_if_empty */);
}

template void calc_v_departure(
  IslMpi<ko::MachineTraits>&, const Int, const Int, const Int, const Real,
  Real*, const Real*, Real*);

} // namespace islmpi
} // namespace homme
