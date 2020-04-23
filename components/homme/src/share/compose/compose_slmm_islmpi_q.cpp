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

static constexpr Real sqrt5 = 2.23606797749978969641; // std::sqrt(5.0);
static constexpr Real oosqrt5 = 1.0 / sqrt5;

SLMM_KF void gll_np4_eval (const Real x, Real y[4]) {
  static constexpr Real oo8 = 1.0/8.0;
  const Real x2 = x*x;
  y[0] = (1.0 - x)*(5.0*x2 - 1.0)*oo8;
  y[1] = -sqrt5*oo8*(sqrt5 - 5.0*x)*(x2 - 1.0);
  y[2] = -sqrt5*oo8*(sqrt5 + 5.0*x)*(x2 - 1.0);
  y[3] = (1.0 + x)*(5.0*x2 - 1.0)*oo8;
}

// Linear interp in each region.
SLMM_KF void gll_np4_subgrid_eval_impl (const Real& x, Real y[4]) {
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

SLMM_KF void gll_np4_subgrid_eval (const Real& x, Real y[4]) {
  if (x > 0) {
    gll_np4_subgrid_eval_impl(-x, y);
    ko::swap(y[0], y[3]);
    ko::swap(y[1], y[2]);    
    return;
  }
  gll_np4_subgrid_eval_impl(x, y);
}

// Quadratic interpolant across nodes 1,2,3 -- i.e., excluding node 0 -- of the
// np=4 reference element.
SLMM_KF void outer_eval (const Real& x, Real v[4]) {
  static constexpr Real
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
SLMM_KF void gll_np4_subgrid_exp_eval (const Real& x, Real y[4]) {
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
      ko::swap(y[0], y[3]);
      ko::swap(y[1], y[2]);
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

template <typename MT>
SLMM_KIF void interpolate (const typename IslMpi<MT>::Advecter::Alg::Enum& alg,
                           const Real ref_coord[2], Real rx[4], Real ry[4]) {
  typedef typename IslMpi<MT>::Advecter::Alg Alg;
  switch (alg) {
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
    slmm_kernel_assert(0);
  }  
}

SLMM_KIF Real calc_q_tgt (const Real rx[4], const Real ry[4], const Real qs[16]) {
  return (ry[0]*(rx[0]*qs[ 0] + rx[1]*qs[ 1] + rx[2]*qs[ 2] + rx[3]*qs[ 3]) +
          ry[1]*(rx[0]*qs[ 4] + rx[1]*qs[ 5] + rx[2]*qs[ 6] + rx[3]*qs[ 7]) +
          ry[2]*(rx[0]*qs[ 8] + rx[1]*qs[ 9] + rx[2]*qs[10] + rx[3]*qs[11]) +
          ry[3]*(rx[0]*qs[12] + rx[1]*qs[13] + rx[2]*qs[14] + rx[3]*qs[15]));
}

SLMM_KIF Real calc_q_tgt (const Real rx[4], const Real ry[4], const Real qdp[16],
                          const Real dp[16]) {
  return (ry[0]*(rx[0]*(qdp[ 0]/dp[ 0]) + rx[1]*(qdp[ 1]/dp[ 1])  +
                 rx[2]*(qdp[ 2]/dp[ 2]) + rx[3]*(qdp[ 3]/dp[ 3])) +
          ry[1]*(rx[0]*(qdp[ 4]/dp[ 4]) + rx[1]*(qdp[ 5]/dp[ 5])  +
                 rx[2]*(qdp[ 6]/dp[ 6]) + rx[3]*(qdp[ 7]/dp[ 7])) +
          ry[2]*(rx[0]*(qdp[ 8]/dp[ 8]) + rx[1]*(qdp[ 9]/dp[ 9])  +
                 rx[2]*(qdp[10]/dp[10]) + rx[3]*(qdp[11]/dp[11])) +
          ry[3]*(rx[0]*(qdp[12]/dp[12]) + rx[1]*(qdp[13]/dp[13])  +
                 rx[2]*(qdp[14]/dp[14]) + rx[3]*(qdp[15]/dp[15])));
}

template <typename Buffer> SLMM_KIF
Int getbuf (Buffer& buf, const Int& os, Int& i1, Int& i2) {
  const Int* const b = reinterpret_cast<const Int*>(&buf(os));
  i1 = b[0];
  i2 = b[1];
  return nreal_per_2int;
}

#ifndef COMPOSE_PORT
// Homme computational pattern.

template <Int np, typename MT>
void calc_q (const IslMpi<MT>& cm, const Int& src_lid, const Int& lev,
             const Real* const dep_point, Real* const q_tgt, const bool use_q) {
  static_assert(np == 4, "Only np 4 is supported.");

  Real ref_coord[2]; {
    const auto& m = cm.advecter->local_mesh(src_lid);
    cm.advecter->s2r().calc_sphere_to_ref(src_lid, m, dep_point,
                                          ref_coord[0], ref_coord[1]);
  }

  Real rx[4], ry[4];
  interpolate<MT>(cm.advecter->alg(), ref_coord, rx, ry);

  const auto& ed = cm.ed_d(src_lid);
  const Int levos = np*np*lev;
  const Int np2nlev = np*np*cm.nlev;
  if (use_q) {
    // We can use q from calc_q_extrema.
    const Real* const qs0 = ed.q + levos;
    // It was found that Intel 18 produced code that was not BFB between runs
    // due to this pragma.
    //#pragma ivdep
    for (Int iq = 0; iq < cm.qsize; ++iq) {
      const Real* const qs = qs0 + iq*np2nlev;
      q_tgt[iq] = calc_q_tgt(rx, ry, qs);
    }
  } else {
    // q from calc_q_extrema is being overwritten, so have to use qdp/dp.
    const Real* const dp = ed.dp + levos;
    const Real* const qdp0 = ed.qdp + levos;
    // I'm commenting out this pragma, too, to be safe.
    //#pragma ivdep
    for (Int iq = 0; iq < cm.qsize; ++iq) {
      const Real* const qdp = qdp0 + iq*np2nlev;
      q_tgt[iq] = calc_q_tgt(rx, ry, qdp, dp);
    }
  }
}

template <Int np, typename MT>
void calc_own_q (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                 const DepPoints<MT>& dep_points,
                 const QExtrema<MT>& q_min, const QExtrema<MT>& q_max) {
  const int tid = get_tid();
  for (Int tci = nets; tci <= nete; ++tci) {
    auto& ed = cm.ed_d(tci);
    const FA3<Real> q_tgt(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (const auto& e: ed.own) {
      const Int slid = ed.nbrs(ed.src(e.lev, e.k)).lid_on_rank;
      const auto& sed = cm.ed_d(slid);
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        q_min(tci, iq, e.lev, e.k) = sed.q_extrema(iq, e.lev, 0);
        q_max(tci, iq, e.lev, e.k) = sed.q_extrema(iq, e.lev, 1);
      }
      Real* const qtmp = &cm.rwork(tid, 0);
      calc_q<np>(cm, slid, e.lev, &dep_points(tci, e.lev, e.k, 0), qtmp, false);
      for (Int iq = 0; iq < cm.qsize; ++iq)
        q_tgt(e.k, e.lev, iq) = qtmp[iq];
    }
  }
}

template <typename MT>
void copy_q (IslMpi<MT>& cm, const Int& nets,
             const QExtrema<MT>& q_min, const QExtrema<MT>& q_max) {
  const auto myrank = cm.p->rank();
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr_h(tid),
           end = cm.mylid_with_comm_tid_ptr_h(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm_d(ptr);
    auto& ed = cm.ed_d(tci);
    const FA3<Real> q_tgt(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (const auto& e: ed.rmt) {
      slmm_assert(ed.nbrs(ed.src(e.lev, e.k)).rank != myrank);
      const Int ri = ed.nbrs(ed.src(e.lev, e.k)).rank_idx;
      const auto&& recvbuf = cm.recvbuf(ri);
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        q_min(tci, iq, e.lev, e.k) = recvbuf(e.q_extrema_ptr + 2*iq    );
        q_max(tci, iq, e.lev, e.k) = recvbuf(e.q_extrema_ptr + 2*iq + 1);
      }
      for (Int iq = 0; iq < cm.qsize; ++iq) {
        slmm_assert(recvbuf(e.q_ptr + iq) != -1);
        q_tgt(e.k, e.lev, iq) = recvbuf(e.q_ptr + iq);
      }
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
    Int mos = 0, qos = 0, nx_in_rank, xos;
    mos += getbuf(xs, mos, xos, nx_in_rank);
    if (nx_in_rank == 0) {
      cm.sendcount_h(ri) = 0;
      continue; 
    }
    // The upper bound is to prevent an inf loop if the msg is corrupted.
    for (Int lidi = 0; lidi < cm.nelemd; ++lidi) {
      Int lid, nx_in_lid;
      mos += getbuf(xs, mos, lid, nx_in_lid);
      const auto& ed = cm.ed_d(lid);
      for (Int levi = 0; levi < cm.nlev; ++levi) { // same re: inf loop
        Int lev, nx;
        mos += getbuf(xs, mos, lev, nx);
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
    cm.sendcount_h(ri) = qos;
  }
}
#else // COMPOSE_PORT
// Hommexx computational pattern.

template <Int np, typename MT> SLMM_KIF
void calc_coefs (const slmm::SphereToRef<typename MT::DES>& s2r,
                 const slmm::LocalMesh<typename MT::DES>& m,
                 const typename slmm::Advecter<MT>::Alg::Enum& alg,
                 const Int& src_lid, const Int& lev,
                 const Real* const dep_point, Real rx[4], Real ry[4]) {
  static_assert(np == 4, "Only np 4 is supported.");
  Real ref_coord[2];
  s2r.calc_sphere_to_ref(src_lid, m, dep_point, ref_coord[0], ref_coord[1]);
  interpolate<MT>(alg, ref_coord, rx, ry);
}

template <Int np, typename MT>
void calc_own_q (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                 const DepPoints<MT>& dep_points,
                 const QExtrema<MT>& q_min, const QExtrema<MT>& q_max) {
  const auto dp_src = cm.tracer_arrays.dp;
  const auto qdp_src = cm.tracer_arrays.qdp;
  const auto q_tgt = cm.tracer_arrays.q;
  const auto ed_d = cm.ed_d;
  const auto s2r = cm.advecter->s2r();
  const auto local_meshes = cm.advecter->local_meshes();
  const auto alg = cm.advecter->alg();
  const Int qsize = cm.qsize, nlev = cm.nlev, np2 = cm.np2;
  const auto f = KOKKOS_LAMBDA (const Int& it) {
    const Int tci = nets + it/(np2*nlev);
    const Int own_id = it % (np2*nlev);
    auto& ed = ed_d(tci);
    if (own_id >= ed.own.size()) return;
    const auto& e = ed.own(own_id);
    const Int slid = ed.nbrs(ed.src(e.lev, e.k)).lid_on_rank;
    const auto& sed = ed_d(slid);
    for (Int iq = 0; iq < qsize; ++iq) {
      q_min(tci, iq, e.lev, e.k) = sed.q_extrema(iq, e.lev, 0);
      q_max(tci, iq, e.lev, e.k) = sed.q_extrema(iq, e.lev, 1);
    }
    Real rx[4], ry[4];
    calc_coefs<np,MT>(s2r, local_meshes(slid), alg, slid, e.lev,
                      &dep_points(tci, e.lev, e.k, 0), rx, ry);
    for (Int iq = 0; iq < qsize; ++iq) {
      // q from calc_q_extrema is being overwritten, so have to use qdp/dp.
      Real dp[16];
      for (Int k = 0; k < 16; ++k) dp[k] = dp_src(slid, e.k, e.lev);
      for (Int iq = 0; iq < qsize; ++iq) {
        Real qdp[16];
        for (Int k = 0; k < 16; ++k) qdp[k] = qdp_src(slid, iq, k, e.lev);
        q_tgt(tci, iq, e.k, e.lev) = calc_q_tgt(rx, ry, qdp, dp);
      }
    }
  };
  ko::parallel_for(
    ko::RangePolicy<typename MT::DES>(0, (nete - nets + 1)*np2*nlev), f);
}

template <typename MT>
void copy_q (IslMpi<MT>& cm, const Int& nets,
             const QExtrema<MT>& q_min, const QExtrema<MT>& q_max) {
  slmm_assert(cm.mylid_with_comm_tid_ptr_h.size() == 2);
  const auto myrank = cm.p->rank();
  const auto q_tgt = cm.tracer_arrays.q;
  const auto mylid_with_comm = cm.mylid_with_comm_d;
  const auto ed_d = cm.ed_d;
  const auto recvbufs = cm.recvbuf;
  const Int nlid = cm.mylid_with_comm_h.size();
  const Int qsize = cm.qsize, nlev = cm.nlev, np2 = cm.np2;
  const auto f = KOKKOS_LAMBDA (const Int& it) {
    const Int tci = mylid_with_comm(it/(np2*nlev));
    const Int rmt_id = it % (np2*nlev);
    auto& ed = ed_d(tci);
    if (rmt_id >= ed.rmt.size()) return;
    const auto& e = ed.rmt(rmt_id);
    slmm_kernel_assert(ed.nbrs(ed.src(e.lev, e.k)).rank != myrank);
    const Int ri = ed.nbrs(ed.src(e.lev, e.k)).rank_idx;
    const auto&& recvbuf = recvbufs(ri);
    for (Int iq = 0; iq < qsize; ++iq) {
      q_min(tci, iq, e.lev, e.k) = recvbuf(e.q_extrema_ptr + 2*iq    );
      q_max(tci, iq, e.lev, e.k) = recvbuf(e.q_extrema_ptr + 2*iq + 1);
    }
    for (Int iq = 0; iq < qsize; ++iq) {
      slmm_kernel_assert(recvbuf(e.q_ptr + iq) != -1);
      q_tgt(tci, iq, e.k, e.lev) = recvbuf(e.q_ptr + iq);
    }
  };
  ko::parallel_for(ko::RangePolicy<typename MT::DES>(0, nlid*np2*nlev), f);
}

template <typename Buffer> SLMM_KIF
Int getbuf (Buffer& buf, const Int& os, Int& i1, short& i2, short& i3) {
  const Int* const b = reinterpret_cast<const Int*>(&buf(os));
  i1 = b[0];
  const short* const b2 = reinterpret_cast<const short*>(b+1);
  i2 = b2[0];
  i3 = b2[1];
  return nreal_per_2int;
}

template <Int np, typename MT>
void calc_rmt_q_pass1_scan (IslMpi<MT>& cm) {
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
  Int cnt = 0, qcnt = 0;
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto&& xs = cm.recvbuf(ri);
    Int mos = 0, qos = 0, nx_in_rank, xos;
    mos += getbuf(xs, mos, xos, nx_in_rank);
    if (nx_in_rank == 0) {
      cm.sendcount_h(ri) = 0;
      continue; 
    }
    const Int data_os = xos;
    while (mos < data_os) {
      Int lid;
      short lev, nx;
      mos += getbuf(xs, mos, lid, lev, nx);
      slmm_assert(nx > 0);
      nx_in_rank -= nx;
      {
        cm.rmt_qs_extrema_h(4*qcnt + 0) = ri;
        cm.rmt_qs_extrema_h(4*qcnt + 1) = lid;
        cm.rmt_qs_extrema_h(4*qcnt + 2) = lev;
        cm.rmt_qs_extrema_h(4*qcnt + 3) = qos;
        ++qcnt;
        qos += 2;
      }
      for (Int xi = 0; xi < nx; ++xi) {
        cm.rmt_xs_h(5*cnt + 0) = ri;
        cm.rmt_xs_h(5*cnt + 1) = lid;
        cm.rmt_xs_h(5*cnt + 2) = lev;
        cm.rmt_xs_h(5*cnt + 3) = xos;
        cm.rmt_xs_h(5*cnt + 4) = qos;
        ++cnt;
        xos += 3;
        ++qos;
      }
    }
    slmm_assert(nx_in_rank == 0);
    cm.sendcount_h(ri) = cm.qsize*qos;
  }
  cm.nrmt_xs = cnt;
  cm.nrmt_qs_extrema = qcnt;
  deep_copy(cm.rmt_xs, cm.rmt_xs_h);
  deep_copy(cm.rmt_qs_extrema, cm.rmt_qs_extrema_h);
}

template <Int np, typename MT>
void calc_rmt_q_pass1 (IslMpi<MT>& cm) {
  if (slmm::OnGpu<MT>::value) {
    calc_rmt_q_pass1_scan<np>(cm);
    return;
  }
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
#ifdef COMPOSE_PORT_SEPARATE_VIEWS
  for (Int ri = 0; ri < nrmtrank; ++ri)
    ko::deep_copy(ko::View<Real*, typename MT::HES>(cm.recvbuf_meta_h(ri).data(), 1),
                  ko::View<Real*, typename MT::HES>(cm.recvbuf.get_h(ri).data(), 1));
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto&& xs = cm.recvbuf_meta_h(ri);
    Int n, unused;
    getbuf(xs, 0, n, unused);
    if (n == 0) continue;
    slmm_assert(n <= cm.recvmetasz[ri]);
    ko::deep_copy(ko::View<Real*, typename MT::HES>(cm.recvbuf_meta_h(ri).data(), n),
                  ko::View<Real*, typename MT::HES>(cm.recvbuf.get_h(ri).data(), n));
  }
#endif
  Int cnt = 0, qcnt = 0;
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto&& xs = cm.recvbuf_meta_h(ri);
    Int mos = 0, qos = 0, nx_in_rank, xos;
    mos += getbuf(xs, mos, xos, nx_in_rank);
    if (nx_in_rank == 0) {
      cm.sendcount_h(ri) = 0;
      continue; 
    }
    // The upper bound is to prevent an inf loop if the msg is corrupted.
    for (Int lidi = 0; lidi < cm.nelemd; ++lidi) {
      Int lid, nx_in_lid;
      mos += getbuf(xs, mos, lid, nx_in_lid);
      for (Int levi = 0; levi < cm.nlev; ++levi) { // same re: inf loop
        Int lev, nx;
        mos += getbuf(xs, mos, lev, nx);
        slmm_assert(nx > 0);
        {
          cm.rmt_qs_extrema_h(4*qcnt + 0) = ri;
          cm.rmt_qs_extrema_h(4*qcnt + 1) = lid;
          cm.rmt_qs_extrema_h(4*qcnt + 2) = lev;
          cm.rmt_qs_extrema_h(4*qcnt + 3) = qos;
          ++qcnt;
          qos += 2;
        }
        for (Int xi = 0; xi < nx; ++xi) {
          cm.rmt_xs_h(5*cnt + 0) = ri;
          cm.rmt_xs_h(5*cnt + 1) = lid;
          cm.rmt_xs_h(5*cnt + 2) = lev;
          cm.rmt_xs_h(5*cnt + 3) = xos;
          cm.rmt_xs_h(5*cnt + 4) = qos;
          ++cnt;
          xos += 3;
          ++qos;
        }
        nx_in_lid -= nx;
        nx_in_rank -= nx;
        if (nx_in_lid == 0) break;
      }
      slmm_assert(nx_in_lid == 0);
      if (nx_in_rank == 0) break;
    }
    slmm_assert(nx_in_rank == 0);
    cm.sendcount_h(ri) = cm.qsize*qos;
  }
  cm.nrmt_xs = cnt;
  cm.nrmt_qs_extrema = qcnt;
  deep_copy(cm.rmt_xs, cm.rmt_xs_h);
  deep_copy(cm.rmt_qs_extrema, cm.rmt_qs_extrema_h);
}

template <Int np, typename MT>
void calc_rmt_q_pass2 (IslMpi<MT>& cm) {
  const auto q_src = cm.tracer_arrays.q;
  const auto rmt_qs_extrema = cm.rmt_qs_extrema;
  const auto rmt_xs = cm.rmt_xs;
  const auto ed_d = cm.ed_d;
  const auto sendbuf = cm.sendbuf;
  const auto recvbuf = cm.recvbuf;
  const Int qsize = cm.qsize;

  const auto fqe = KOKKOS_LAMBDA (const Int& it) {
    const Int
    ri = rmt_qs_extrema(4*it), lid = rmt_qs_extrema(4*it + 1),
    lev = rmt_qs_extrema(4*it + 2), qos = qsize*rmt_qs_extrema(4*it + 3);  
    auto&& qs = sendbuf(ri);
    const auto& ed = ed_d(lid);
    for (Int iq = 0; iq < qsize; ++iq)
      for (int i = 0; i < 2; ++i)
        qs(qos + 2*iq + i) = ed.q_extrema(iq, lev, i);
  };
  ko::fence();
  ko::parallel_for(ko::RangePolicy<typename MT::DES>(0, cm.nrmt_qs_extrema), fqe);

  const auto s2r = cm.advecter->s2r();
  const auto local_meshes = cm.advecter->local_meshes();
  const auto alg = cm.advecter->alg();

  const auto fx = KOKKOS_LAMBDA (const Int& it) {
    const Int
    ri = rmt_xs(5*it), lid = rmt_xs(5*it + 1), lev = rmt_xs(5*it + 2),
    xos = rmt_xs(5*it + 3), qos = qsize*rmt_xs(5*it + 4);
    const auto&& xs = recvbuf(ri);
    auto&& qs = sendbuf(ri);
    Real rx[4], ry[4];
    calc_coefs<np,MT>(s2r, local_meshes(lid), alg, lid, lev, &xs(xos), rx, ry);
    Real* const q_tgt = &qs(qos);
    for (Int iq = 0; iq < qsize; ++iq) {
      Real qsrc[16];
      for (Int k = 0; k < 16; ++k) qsrc[k] = q_src(lid, iq, k, lev);
      q_tgt[iq] = calc_q_tgt(rx, ry, qsrc);
    }
  };
  ko::parallel_for(ko::RangePolicy<typename MT::DES>(0, cm.nrmt_xs), fx);
  ko::fence();
}

template <Int np, typename MT>
void calc_rmt_q (IslMpi<MT>& cm) {
  calc_rmt_q_pass1<np>(cm);
  calc_rmt_q_pass2<np>(cm);
}

#endif // COMPOSE_PORT

template <typename MT>
void calc_own_q (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                 const DepPoints<MT>& dep_points,
                 const QExtrema<MT>& q_min, const QExtrema<MT>& q_max) {
  switch (cm.np) {
  case 4: calc_own_q<4>(cm, nets, nete, dep_points, q_min, q_max); break;
  default: slmm_throw_if(true, "np " << cm.np << "not supported");
  }
}

template <typename MT>
void calc_rmt_q (IslMpi<MT>& cm) {
  switch (cm.np) {
  case 4: calc_rmt_q<4>(cm); break;
  default: slmm_throw_if(true, "np " << cm.np << "not supported");
  }
}

template void calc_rmt_q(IslMpi<slmm::MachineTraits>& cm);
template void calc_own_q(IslMpi<slmm::MachineTraits>& cm,
                         const Int& nets, const Int& nete,
                         const DepPoints<slmm::MachineTraits>& dep_points,
                         const QExtrema<slmm::MachineTraits>& q_min,
                         const QExtrema<slmm::MachineTraits>& q_max);
template void copy_q(IslMpi<slmm::MachineTraits>& cm, const Int& nets,
                     const QExtrema<slmm::MachineTraits>& q_min,
                     const QExtrema<slmm::MachineTraits>& q_max);

} // namespace islmpi
} // namespace homme
