#include "compose_slmm_islmpi.hpp"

namespace slmm {
static Int test_gll () {
  Int nerr = 0;
  const Real tol = 1e2*std::numeric_limits<Real>::epsilon();
  GLL gll;
  const Real* x, * wt;
  for (Int np = 2; np <= 4; ++np) {
    for (Int monotone_type = 0; monotone_type <= 1; ++monotone_type) {
      const Basis b(np, monotone_type);
      gll.get_coef(b, x, wt);
      Real sum = 0;
      for (Int i = 0; i < b.np; ++i)
        sum += wt[i];
      if (std::abs(2 - sum) > tol) {
        std::cerr << "test_gll " << np << ", " << monotone_type
                  << ": 2 - sum = " << 2 - sum << "\n";
        ++nerr;
      }
      for (Int j = 0; j < b.np; ++j) {
        Real gj[GLL::np_max];
        gll.eval(b, x[j], gj);
        for (Int i = 0; i < b.np; ++i) {
          if (j == i) continue;
          if (std::abs(gj[i]) > tol) {
            std::cerr << "test_gll " << np << ", " << monotone_type << ": gj["
                      << i << "] = " << gj[i] << "\n";
            ++nerr;
          }
        }
      }
    }
  }
  for (Int np = 2; np <= 4; ++np) {
    const Basis b(np, 0);
    Real a[] = {-0.9, -0.7, -0.3, 0.1, 0.2, 0.4, 0.6, 0.8};
    const Real delta = std::sqrt(std::numeric_limits<Real>::epsilon());
    for (size_t ia = 0; ia < sizeof(a)/sizeof(Real); ++ia) {
      Real gj[GLL::np_max], gjp[GLL::np_max], gjm[GLL::np_max];
      gll.eval_derivative(b, a[ia], gj);
      gll.eval(b, a[ia] + delta, gjp);
      gll.eval(b, a[ia] - delta, gjm);
      for (Int i = 0; i < b.np; ++i) {
        const Real fd = (gjp[i] - gjm[i])/(2*delta);
        if (std::abs(fd - gj[i]) >= delta*std::abs(gjp[i]))
          ++nerr;
      }
    }
  }
  return nerr;
}

int unittest () {
  int nerr = 0;
  nerr += test_gll();
  return nerr;
}

static const Real sqrt5 = std::sqrt(5.0);
static const Real oosqrt5 = 1.0 / sqrt5;

void gll_np4_eval (const Real x, Real y[4]) {
  static constexpr Real oo8 = 1.0/8.0;
  const Real x2 = x*x;
  y[0] = (1.0 - x)*(5.0*x2 - 1.0)*oo8;
  y[1] = -sqrt5*oo8*(sqrt5 - 5.0*x)*(x2 - 1.0);
  y[2] = -sqrt5*oo8*(sqrt5 + 5.0*x)*(x2 - 1.0);
  y[3] = (1.0 + x)*(5.0*x2 - 1.0)*oo8;
}

// Linear interp in each region.
void gll_np4_subgrid_eval (const Real& x, Real y[4]) {
  if (x > 0) {
    gll_np4_subgrid_eval(-x, y);
    std::swap(y[0], y[3]);
    std::swap(y[1], y[2]);    
    return;
  }
  if (x < -oosqrt5) {
    const Real alpha = (x + 1)/(1 - oosqrt5);
    y[0] = 1 - alpha;
    y[1] = alpha;
    y[2] = 0;
    y[3] = 0;
  } else {
    const Real alpha = (x + oosqrt5)/(2*oosqrt5);
    y[0] = 0;
    y[1] = 1 - alpha;
    y[2] = alpha;
    y[3] = 0;
  }
}

// Quadratic interpolant across nodes 1,2,3 -- i.e., excluding node 0 -- of the
// np=4 reference element.
void outer_eval (const Real& x, Real v[4]) {
  static const Real
    xbar = (2*oosqrt5) / (1 + oosqrt5),
    ooxbar = 1 / xbar,
    ybar = 1 / (xbar - 1);
  const Real xn = (x + oosqrt5) / (1 + oosqrt5);
  v[0] = 0;
  v[1] = 1 + ybar*xn*((1 - ooxbar)*xn + ooxbar - xbar);
  v[2] = ybar*ooxbar*xn*(xn - 1);
  v[3] = ybar*xn*(xbar - xn);
}

// In the middle region, use the standard GLL np=4 interpolant; in the two outer
// regions, use an order-reduced interpolant that stabilizes the method.
void gll_np4_subgrid_exp_eval (const Real& x, Real y[4]) {
  static constexpr Real
    alpha = 0.5527864045000416708,
    v = 0.427*(1 + alpha),
    x2 = 0.4472135954999579277,
    x3 = 1 - x2,
    det = x2*x3*(x2 - x3),
    y2 = alpha,
    y3 = v,
    c1 = (x3*y2 - x2*y3)/det,
    c2 = (-x3*x3*y2 + x2*x2*y3)/det;
  if (x < -oosqrt5 || x > oosqrt5) {
    if (x < -oosqrt5) {
      outer_eval(-x, y);
      std::swap(y[0], y[3]);
      std::swap(y[1], y[2]);
    } else
      outer_eval(x, y);
    Real y4[4];
    gll_np4_eval(x, y4);
    const Real x0 = 1 - std::abs(x);
    const Real a = (c1*x0 + c2)*x0;
    for (int i = 0; i < 4; ++i)
      y[i] = a*y[i] + (1 - a)*y4[i];
  } else
    gll_np4_eval(x, y);
}
} // namespace slmm

namespace homme {
namespace islmpi {

template <Int np, typename MT>
void calc_q (const IslMpi<MT>& cm, const Int& src_lid, const Int& lev,
             const Real* const dep_point, Real* const q_tgt, const bool use_q) {
  Real ref_coord[2]; {
    const auto& m = cm.advecter->local_mesh(src_lid);
    cm.advecter->s2r().calc_sphere_to_ref(src_lid, m, dep_point,
                                          ref_coord[0], ref_coord[1]);
  }

  // Interpolate.
  Real rx[4], ry[4];
  typedef typename IslMpi<MT>::Advecter::Alg Alg;
  switch (cm.advecter->alg()) {
  case Alg::csl_gll:
    slmm::gll_np4_eval(ref_coord[0], rx);
    slmm::gll_np4_eval(ref_coord[1], ry);
    break;
  case Alg::csl_gll_subgrid:
    slmm::gll_np4_subgrid_eval(ref_coord[0], rx);
    slmm::gll_np4_subgrid_eval(ref_coord[1], ry);
    break;
  case Alg::csl_gll_exp:
    slmm::gll_np4_subgrid_exp_eval(ref_coord[0], rx);
    slmm::gll_np4_subgrid_exp_eval(ref_coord[1], ry);
    break;
  default:
    slmm_assert(0);
  }

  const auto& ed = cm.ed_d(src_lid);
  const Int levos = np*np*lev;
  const Int np2nlev = np*np*cm.nlev;
  if (use_q) {
    // We can use q from calc_q_extrema.
    const Real* const qs0 = ed.q + levos;
#   pragma ivdep
    for (Int iq = 0; iq < cm.qsize; ++iq) {
      const Real* const qs = qs0 + iq*np2nlev;
      q_tgt[iq] =
        (ry[0]*(rx[0]*qs[ 0] + rx[1]*qs[ 1] + rx[2]*qs[ 2] + rx[3]*qs[ 3]) +
         ry[1]*(rx[0]*qs[ 4] + rx[1]*qs[ 5] + rx[2]*qs[ 6] + rx[3]*qs[ 7]) +
         ry[2]*(rx[0]*qs[ 8] + rx[1]*qs[ 9] + rx[2]*qs[10] + rx[3]*qs[11]) +
         ry[3]*(rx[0]*qs[12] + rx[1]*qs[13] + rx[2]*qs[14] + rx[3]*qs[15]));
    }
  } else {
    // q from calc_q_extrema is being overwritten, so have to use qdp/dp.
    const Real* const dp = ed.dp + levos;
    const Real* const qdp0 = ed.qdp + levos;
#   pragma ivdep
    for (Int iq = 0; iq < cm.qsize; ++iq) {
      const Real* const qdp = qdp0 + iq*np2nlev;
      q_tgt[iq] = (ry[0]*(rx[0]*(qdp[ 0]/dp[ 0]) + rx[1]*(qdp[ 1]/dp[ 1])  +
                          rx[2]*(qdp[ 2]/dp[ 2]) + rx[3]*(qdp[ 3]/dp[ 3])) +
                   ry[1]*(rx[0]*(qdp[ 4]/dp[ 4]) + rx[1]*(qdp[ 5]/dp[ 5])  +
                          rx[2]*(qdp[ 6]/dp[ 6]) + rx[3]*(qdp[ 7]/dp[ 7])) +
                   ry[2]*(rx[0]*(qdp[ 8]/dp[ 8]) + rx[1]*(qdp[ 9]/dp[ 9])  +
                          rx[2]*(qdp[10]/dp[10]) + rx[3]*(qdp[11]/dp[11])) +
                   ry[3]*(rx[0]*(qdp[12]/dp[12]) + rx[1]*(qdp[13]/dp[13])  +
                          rx[2]*(qdp[14]/dp[14]) + rx[3]*(qdp[15]/dp[15])));
    }
  }
}

template <Int np, typename MT>
void calc_rmt_q (IslMpi<MT>& cm) {
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp for
#endif
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto&& xs = cm.recvbuf(ri);
    auto&& qs = cm.sendbuf(ri);
    Int xos = 0, qos = 0, nx_in_rank, padding;
    xos += getbuf(xs, xos, nx_in_rank, padding);
    if (nx_in_rank == 0) {
      cm.sendcount(ri) = 0;
      continue; 
    }
    // The upper bound is to prevent an inf loop if the msg is corrupted.
    for (Int lidi = 0; lidi < cm.nelemd; ++lidi) {
      Int lid, nx_in_lid;
      xos += getbuf(xs, xos, lid, nx_in_lid);
      const auto& ed = cm.ed_d(lid);
      for (Int levi = 0; levi < cm.nlev; ++levi) { // same re: inf loop
        Int lev, nx;
        xos += getbuf(xs, xos, lev, nx);
        slmm_assert(nx > 0);
        for (Int iq = 0; iq < cm.qsize; ++iq)
          for (int i = 0; i < 2; ++i)
            qs(qos + 2*iq + i) = ed.q_extrema(iq, lev, i);
        qos += 2*cm.qsize;
        for (Int ix = 0; ix < nx; ++ix) {
          calc_q<np>(cm, lid, lev, &xs(xos), &qs(qos), true);
          xos += 3;
          qos += cm.qsize;
        }
        nx_in_lid -= nx;
        nx_in_rank -= nx;
        if (nx_in_lid == 0) break;
      }
      slmm_assert(nx_in_lid == 0);
      if (nx_in_rank == 0) break;
    }
    slmm_assert(nx_in_rank == 0);
    cm.sendcount(ri) = qos;
  }
}

template <typename MT>
void calc_rmt_q (IslMpi<MT>& cm) {
  switch (cm.np) {
  case 4: calc_rmt_q<4>(cm); break;
  default: slmm_throw_if(true, "np " << cm.np << "not supported");
  }
}

template <Int np, typename MT>
void calc_own_q (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                 const FA4<const Real>& dep_points,
                 const FA4<Real>& q_min, const FA4<Real>& q_max) {
  const int tid = get_tid();
  for (Int tci = nets; tci <= nete; ++tci) {
    const Int ie0 = tci - nets;
    auto& ed = cm.ed_d(tci);
    const FA3<Real> q_tgt(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (const auto& e: ed.own) {
      const Int slid = ed.nbrs(ed.src(e.lev, e.k)).lid_on_rank;
      const auto& sed = cm.ed_d(slid);
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        q_min(e.k, e.lev, iq, ie0) = sed.q_extrema(iq, e.lev, 0);
        q_max(e.k, e.lev, iq, ie0) = sed.q_extrema(iq, e.lev, 1);
      }
      Real* const qtmp = &cm.rwork(tid, 0);
      calc_q<np>(cm, slid, e.lev, &dep_points(0, e.k, e.lev, tci), qtmp, false);
      for (Int iq = 0; iq < cm.qsize; ++iq)
        q_tgt(e.k, e.lev, iq) = qtmp[iq];
    }
  }
}

template <typename MT>
void calc_own_q (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                 const FA4<const Real>& dep_points,
                 const FA4<Real>& q_min, const FA4<Real>& q_max) {
  switch (cm.np) {
  case 4: calc_own_q<4>(cm, nets, nete, dep_points, q_min, q_max); break;
  default: slmm_throw_if(true, "np " << cm.np << "not supported");
  }
}

template <typename MT>
void copy_q (IslMpi<MT>& cm, const Int& nets,
             const FA4<Real>& q_min, const FA4<Real>& q_max) {
  const auto myrank = cm.p->rank();
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr(tid),
           end = cm.mylid_with_comm_tid_ptr(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm(ptr);
    const Int ie0 = tci - nets;
    auto& ed = cm.ed_d(tci);
    const FA3<Real> q_tgt(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (const auto& e: ed.rmt) {
      slmm_assert(ed.nbrs(ed.src(e.lev, e.k)).rank != myrank);
      const Int ri = ed.nbrs(ed.src(e.lev, e.k)).rank_idx;
      const auto&& recvbuf = cm.recvbuf(ri);
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        q_min(e.k, e.lev, iq, ie0) = recvbuf(e.q_extrema_ptr + 2*iq    );
        q_max(e.k, e.lev, iq, ie0) = recvbuf(e.q_extrema_ptr + 2*iq + 1);
      }
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        slmm_assert(recvbuf(e.q_ptr + iq) != -1);
        q_tgt(e.k, e.lev, iq) = recvbuf(e.q_ptr + iq);
      }
    }
  }
}

template void calc_rmt_q(IslMpi<slmm::MachineTraits>& cm);
template void calc_own_q(IslMpi<slmm::MachineTraits>& cm, const Int& nets,
                         const Int& nete, const FA4<const Real>& dep_points,
                         const FA4<Real>& q_min, const FA4<Real>& q_max);
template void copy_q(IslMpi<slmm::MachineTraits>& cm, const Int& nets,
                     const FA4<Real>& q_min, const FA4<Real>& q_max);

} // namespace islmpi
} // namespace homme
