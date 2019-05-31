#include "compose_slmm.hpp"
#include "compose_slmm_siqk.hpp"
#include "compose_slmm_advecter.hpp"

#include <sys/time.h>
#include <mpi.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <list>
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>

namespace slmm {
void copy_vertices (
  const siqk::ConstVec3s::HostMirror& p, const siqk::ConstIdxs::HostMirror& c2n,
  const Int ci, Real* ps)
{
  const auto cell = slice(c2n, ci);
  for (Int i = 0; i < szslice(c2n); ++i) {
    const auto n = slice(p, cell[i]);
    for (Int k = 0; k < 3; ++k) ps[k] = n[k];
    ps += 3;
  }
}

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

int get_nearest_point (const siqk::Mesh<ko::HostSpace>& m,
                       const nearest_point::MeshNearestPointData<ko::HostSpace>& d,
                       Real* v, const Int my_ic) {
  nearest_point::calc(m, d, v);
  return get_src_cell(m, v, my_ic);
}
} // namespace slmm

#ifdef SLMM_MAIN
/*
  mpicxx -O3 -g -DSLMM_MAIN -Wall -pedantic -fopenmp -std=c++11 -I/home/ambradl/lib/kokkos/cpu/include slmm.cpp -L/home/ambradl/lib/kokkos/cpu/lib -lkokkos -ldl -llapack -lblas; if [ $? == 0 ]; then OMP_PROC_BIND=false OMP_NUM_THREADS=2 mpirun -np 14 ./a.out; fi
 */
int main (int argc, char** argv) {
  int ret = 0;
  Kokkos::initialize(argc, argv);
  try {
    int nerr = 0;
    nerr += slmm::unittest();
    std::cerr << (nerr ? "FAIL" : "PASS") << "ED\n";
  } catch (const std::exception& e) {
    std::cerr << e.what();
  }
  Kokkos::finalize();
  return ret;
}
#endif

#include "compose_slmm_islmpi.hpp"

namespace homme {
static slmm::Advecter::Ptr g_advecter;

void slmm_init (const Int np, const Int nelem, const Int nelemd,
                const Int transport_alg, const Int cubed_sphere_map,
                const Int sl_nearest_point_lev, const Int* lid2facenum) {
  g_advecter = std::make_shared<slmm::Advecter>(
    np, nelemd, transport_alg, cubed_sphere_map, sl_nearest_point_lev);
  g_advecter->init_meta_data(nelem, lid2facenum);
}

namespace islmpi {
inline int get_tid () {
#ifdef HORIZ_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline int get_num_threads () {
#ifdef HORIZ_OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

namespace extend_halo {
// Extend halo by one layer. This has two parts: finding neighbor (gid, rank) in
// collect_gid_rank, and extending the Advecter local mesh geometry in
// extend_local_meshes. The two parts have the same comm pattern: in round 1,
// request data for lists of GIDs; in round 2, fulfill these requests.

typedef IslMpi::ElemData ElemData;
typedef Int Gid;
typedef Int Rank;
struct GidRankPair {
  const Gid gid;
  const Rank rank;
  GidRankPair(const Gid& gid_, const Rank& rank_) : gid(gid_), rank(rank_) {}
  bool operator< (const GidRankPair& o) const {
    if (gid < o.gid) return true;
    if (gid > o.gid) return false;
    return rank < o.rank;
  }
};
typedef std::vector<GidRankPair> GidRankPairs;
typedef std::map<Gid, GidRankPairs> Gid2Nbrs;
typedef std::vector<Int> IntBuf;
typedef std::vector<Real> RealBuf;

GidRankPairs all_nbrs_but_me (const ElemData& ed) {
  GidRankPairs gs;
  gs.reserve(ed.nbrs.size() - 1);
  for (const auto& n : ed.nbrs)
    if (&n != ed.me)
      gs.push_back(GidRankPair(n.gid, n.rank));
  return gs;
}

void fill_gid2nbrs (const mpi::Parallel& p, const FixedCapList<ElemData>& eds,
                    Gid2Nbrs& gid2nbrs) {
  static const Int tag = 6;
  const Rank my_rank = p.rank();
  const Int n_owned = eds.size();

  // Fill in the ones we know.
  for (const auto& ed : eds) {
    slmm_assert(ed.me->rank == my_rank);
    gid2nbrs[ed.me->gid] = all_nbrs_but_me(ed);
  }

  std::vector<Rank> ranks;
  Int nrank;
  std::vector<IntBuf> req_sends, req_recvs;
  std::vector<mpi::Request> req_recv_reqs;
  {
    // Find the ranks that know the rest.
    std::map<Rank,Int> rank2rankidx;
    std::map<Gid,Rank> needgid2rank;
    {
      std::set<Rank> unique_ranks;
      for (const auto& item : gid2nbrs)
        for (const auto& n : item.second)
          if (n.rank != my_rank) {
            slmm_assert(gid2nbrs.find(n.gid) == gid2nbrs.end());
            needgid2rank.insert(std::make_pair(n.gid, n.rank));
            unique_ranks.insert(n.rank);
          }
      nrank = unique_ranks.size();
      ranks.insert(ranks.begin(), unique_ranks.begin(), unique_ranks.end());
      Int i = 0;
      for (const auto& rank : ranks)
        rank2rankidx[rank] = i++;
    }

    // Send and receive neighbor queries.
    slmm_assert(ranks.size() == static_cast<size_t>(nrank));
    req_sends.resize(nrank);
    req_recvs.resize(nrank); req_recv_reqs.resize(nrank);
    for (Int i = 0; i < nrank; ++i) {
      auto& r = req_recvs[i];
      r.resize(n_owned+1); // upper bound on #requests, plus 1 for size datum
      mpi::irecv(p, r.data(), r.size(), ranks[i], tag, &req_recv_reqs[i]);
      req_sends[i].push_back(0);
    }
    for (const auto& item : needgid2rank) {
      const auto& gid = item.first;
      const auto& rank = item.second;
      const auto& it = rank2rankidx.find(rank);
      slmm_assert(it != rank2rankidx.end());
      const auto& rank_idx = it->second;
      slmm_assert(gid2nbrs.find(gid) == gid2nbrs.end());
      req_sends[rank_idx].push_back(gid);
    }
    for (Int i = 0; i < nrank; ++i) {
      auto& s = req_sends[i];
      s[0] = s.size() - 1; // #gids in request
      mpi::isend(p, s.data(), s.size(), ranks[i], tag, nullptr);
    }
  }

  // Fullfill queries and receive answers to our queries.
  std::vector<IntBuf> nbr_sends(nrank), nbr_recvs(nrank);
  std::vector<mpi::Request> nbr_send_reqs(nrank), nbr_recv_reqs(nrank);
  for (Int i = 0; i < nrank; ++i) {
    auto& r = nbr_recvs[i];
    // 20 is from dimensions_mod::set_mesh_dimensions; factor of 2 is to get
    // (gid,rank); 1 is for size datum.
    r.resize((20*2 + 1)*(req_sends[i].size() - 1));
    mpi::irecv(p, r.data(), r.size(), ranks[i], tag, &nbr_recv_reqs[i]);
  }
  for (Int k = 0; k < nrank; ++k) {
    Int i;
    mpi::waitany(nrank, req_recv_reqs.data(), &i);
    const auto& r = req_recvs[i];
    auto& s = nbr_sends[i];
    const Int ngid = r[0];
    for (Int j = 0; j < ngid; ++j) {
      const auto gid = r[j+1];
      const auto it = gid2nbrs.find(gid);
      slmm_assert(it != gid2nbrs.end());
      const auto& nbrs = it->second;
      s.push_back(nbrs.size());
      for (const auto& n : nbrs) {
        s.push_back(n.gid);
        s.push_back(n.rank);
      }
    }
    mpi::isend(p, s.data(), s.size(), ranks[i], tag, &nbr_send_reqs[i]);
  }
  for (Int j = 0; j < nrank; ++j) {
    Int i;
    mpi::waitany(nrank, nbr_recv_reqs.data(), &i);
    const auto& s = req_sends[i];
    const Int nrequested = s.size() - 1;
    const auto& r = nbr_recvs[i];
    for (Int si = 0, ri = 0; si < nrequested; ++si) {
      const Gid gid = s[si+1];
      const Int nnbr = r[ri++];
      GidRankPairs nbrs;
      for (Int ni = 0; ni < nnbr; ++ni, ri += 2)
        nbrs.push_back(GidRankPair(r[ri], r[ri+1]));
      slmm_assert(gid2nbrs.find(gid) == gid2nbrs.end());
      gid2nbrs.insert(std::make_pair(gid, nbrs));
    }
  }
  mpi::waitall(nbr_send_reqs.size(), nbr_send_reqs.data());
}

void extend_nbrs (const Gid2Nbrs& gid2nbrs, FixedCapList<ElemData>& eds) {
  for (auto& ed : eds) {
    // Get all <=2-halo neighbors.
    std::set<GidRankPair> new_nbrs;
    for (const auto& n : ed.nbrs) {
      if (&n == ed.me) continue;
      const auto& it = gid2nbrs.find(n.gid);
      slmm_assert(it != gid2nbrs.end());
      const auto& gid_nbrs = it->second;
      for (const auto& gn : gid_nbrs)
        new_nbrs.insert(gn);
    }
    // Remove the already known ones.
    for (const auto& n : ed.nbrs)
      new_nbrs.erase(GidRankPair(n.gid, n.rank));
    // Find me b/c of the pointer reset.
    Int me = -1;
    for (Int i = 0; i < ed.nbrs.size(); ++i)
      if (&ed.nbrs(i) == ed.me) {
        me = i;
        break;
      }
    slmm_assert(me >= 0);
    // Append the, now only new, 2-halo ones.
    Int i = ed.nbrs.size();
    ed.nbrs.reset_capacity(i + new_nbrs.size(), true);
    ed.me = &ed.nbrs(me);
    for (const auto& n : new_nbrs) {
      auto& en = ed.nbrs(i++);
      en.gid = n.gid;
      en.rank = n.rank;
      en.rank_idx = -1;
      en.lid_on_rank = -1;
      en.lid_on_rank_idx = -1;
    }
  }
}

void collect_gid_rank (const mpi::Parallel& p, FixedCapList<ElemData>& eds) {
  Gid2Nbrs gid2nbrs;
  fill_gid2nbrs(p, eds, gid2nbrs);
  extend_nbrs(gid2nbrs, eds);
}

void extend_local_meshes (const mpi::Parallel& p, const FixedCapList<ElemData>& eds,
                          slmm::Advecter& advecter) {
  using slmm::slice;
  using slmm::nslices;
  using slmm::szslice;
  using slmm::len;
  using slmm::LocalMesh;
  using RealArray3 = ko::View<Real***, ko::HostSpace>;

  static const Int tag = 24;
  const Int my_rank = p.rank();
  const Int n_owned = eds.size();

  RealArray3 corners;
  std::map<Gid,Int> owngid2lid, rmtgid2idx;
  {
    // Find relevant ranks.
    std::vector<Int> ranks;
    Int nrank;
    std::vector<IntBuf> req_sends, req_recvs;
    std::vector<mpi::Request> req_recv_reqs;
    {
      std::map<Int,Int> rank2rankidx;
      {
        std::set<Int> uranks;
        for (const auto& ed : eds)
          for (const auto& n : ed.nbrs)
            uranks.insert(n.rank);
        uranks.erase(my_rank);
        ranks.insert(ranks.begin(), uranks.begin(), uranks.end());
        nrank = ranks.size();
        Int i = 0;
        for (const auto& rank : ranks)
          rank2rankidx[rank] = i++;
      }

      // Trade requests.
      req_sends.resize(nrank); req_recvs.resize(nrank);
      req_recv_reqs.resize(nrank);
      for (Int i = 0; i < nrank; ++i) { // Set up recvs.
        auto& r = req_recvs[i];
        r.resize(n_owned+1);
        mpi::irecv(p, r.data(), r.size(), ranks[i], tag, &req_recv_reqs[i]);
      }
      std::set<Gid> unique_rmtgids;
      { // Collect the remote GIDs by rank.
        std::vector<std::set<Gid> > req_gids(nrank);
        for (const auto& ed : eds)
          for (Int i = ed.nin1halo; i < ed.nbrs.size(); ++i) {
            const auto& n = ed.nbrs(i);
            if (n.rank != my_rank) {
              req_gids[rank2rankidx[n.rank]].insert(n.gid);
              unique_rmtgids.insert(n.gid);
            }
          }
        for (Int i = 0; i < nrank; ++i) { // Set up sends.
          auto& s = req_sends[i];
          s.push_back(0);
          s.insert(s.end(), req_gids[i].begin(), req_gids[i].end());
          s[0] = s.size() - 1;
          mpi::isend(p, s.data(), s.size(), ranks[i], tag, nullptr);
        }
        Int i = 0;
        for (const auto& gid : unique_rmtgids)
          rmtgid2idx[gid] = i++;
      }
    }

    for (Int i = 0; i < eds.size(); ++i) {
      const Gid gid = eds(i).me->gid;
      owngid2lid[gid] = i;
    }

    // Fulfill our requests and obtain the data in reply to ours.
    std::vector<RealBuf> geo_recvs(nrank), geo_sends(nrank);
    std::vector<mpi::Request> geo_recv_reqs(nrank), geo_send_reqs(nrank);
    for (Int i = 0; i < nrank; ++i) {
      auto& r = geo_recvs[i];
      r.resize(12*(req_sends[i].size() - 1)); // 4 vertices x 3 dimensions
      mpi::irecv(p, r.data(), r.size(), ranks[i], tag, &geo_recv_reqs[i]);
    }
    for (Int k = 0; k < nrank; ++k) { // Pack cell corner points for requesters.
      Int i;
      mpi::waitany(nrank, req_recv_reqs.data(), &i);
      const auto& r = req_recvs[i];
      auto& s = geo_sends[i];
      const Int ngid = r[0];
      s.resize(12*ngid);
      Int si = 0;
      for (Int j = 0; j < ngid; ++j) {
        const auto gid = r[j+1];
        const auto it = owngid2lid.find(gid);
        slmm_assert(it != owngid2lid.end());
        const auto lid = it->second;
        const auto& local_mesh = advecter.local_mesh(lid);
        const auto cell = slice(local_mesh.e, local_mesh.tgt_elem);
        slmm_assert(szslice(local_mesh.e) == 4 && cell[3] != -1);
        const auto& p = local_mesh.p;
        for (Int v = 0; v < 4; ++v)
          for (Int d = 0; d < 3; ++d)
            s[si++] = p(cell[v],d);
      }
      slmm_assert(si == static_cast<Int>(s.size()));
      mpi::isend(p, s.data(), s.size(), ranks[i], tag, &geo_send_reqs[i]);
    }
    corners = RealArray3("corners", rmtgid2idx.size(), 4, 3);
    for (Int j = 0; j < nrank; ++j) { // Pack cell corner points for me.
      Int i;
      mpi::waitany(nrank, geo_recv_reqs.data(), &i);
      const auto& s = req_sends[i];
      const Int nrequested = s.size() - 1;
      const auto& r = geo_recvs[i];
      for (Int si = 0, ri = 0; si < nrequested; ++si) {
        const Gid gid = s[si+1];
        const auto it = rmtgid2idx.find(gid);
        slmm_assert(it != rmtgid2idx.end());
        const auto idx = it->second;
        for (Int v = 0; v < 4; ++v)
          for (Int d = 0; d < 3; ++d)
            corners(idx,v,d) = r[ri++];
      }
    }
    // Do this here so we can release the memory.
    mpi::waitall(geo_send_reqs.size(), geo_send_reqs.data());
  }

  // Extend the local meshes.
  for (Int lid = 0; lid < eds.size(); ++lid) {
    const auto& ed = eds(lid);
    auto& local_mesh = advecter.local_mesh(lid);
    auto p0 = local_mesh.p;
    auto e0 = local_mesh.e;
    const Int ncell = ed.nbrs.size();
    slmm_assert(szslice(p0) == 3 && szslice(e0) == 4);
    local_mesh.p = LocalMesh::RealArray("p", 4*ncell);
    auto& p = local_mesh.p;
    local_mesh.e = LocalMesh::IntArray("e", ncell, 4);
    auto& e = local_mesh.e;
    // Copy in old data.
    for (Int pi = 0; pi < nslices(p0); ++pi)
      for (Int d = 0; d < 3; ++d)
        p(pi,d) = p0(pi,d);
    for (Int ei = 0; ei < nslices(e0); ++ei)
      for (Int d = 0; d < 4; ++d)
        e(ei,d) = e0(ei,d);
    // Fill in new data.
    for (Int ni = ed.nin1halo; ni < ed.nbrs.size(); ++ni) {
      const auto& n = ed.nbrs(ni);
      const Gid gid = n.gid;
      if (n.rank != my_rank) {
        const auto it = rmtgid2idx.find(gid);
        slmm_assert(it != rmtgid2idx.end());
        const Int idx = it->second;
        for (Int v = 0; v < 4; ++v) {
          const Int slot = 4*ni+v;
          e(ni,v) = slot;
          slmm_assert(slot < nslices(p));
          for (Int d = 0; d < 3; ++d) p(slot,d) = corners(idx,v,d);
        }
      } else {
        const auto it = owngid2lid.find(gid);
        slmm_assert(it != owngid2lid.end());
        const Int lid = it->second;
        const auto& local_mesh_other = advecter.local_mesh(lid);
        const auto cell_other = slice(local_mesh_other.e, local_mesh_other.tgt_elem);
        const auto& p_other = local_mesh_other.p;
        for (Int v = 0; v < 4; ++v) {
          const Int slot = 4*ni+v;
          e(ni,v) = slot;
          slmm_assert(slot < nslices(p));
          for (Int d = 0; d < 3; ++d) p(slot,d) = p_other(cell_other[v],d);
        }
      }
    }
    // Recompute all normals.
    siqk::test::fill_normals<siqk::SphereGeometry>(local_mesh);
  }
}
} // namespace extend_halo

// Fill in (gid, rank), the list of owning rank per gid.
void collect_gid_rank (IslMpi& cm, const Int* nbr_id_rank, const Int* nirptr) {
  cm.ed.reset_capacity(cm.nelemd, true);
  for (Int i = 0; i < cm.nelemd; ++i) {
    auto& ed = cm.ed(i);
    const Int* nir = nbr_id_rank + nirptr[i];
    const Int nnir = (nirptr[i+1] - nirptr[i]) / 2;
    const Int mygid = nir[0];
    ed.me = nullptr;
    ed.nin1halo = nnir-1;
    ed.nbrs.reset_capacity(ed.nin1halo, true);
    ed.own.reset_capacity(cm.nlev * cm.np2);
    ed.rmt.reset_capacity(cm.nlev * cm.np2);
    for (Int j = 1; j < nnir; ++j) {
      auto& n = ed.nbrs(j-1);
      n.gid = nir[2*j];
      if (n.gid == mygid) {
        slmm_assert( ! ed.me);
        ed.me = &n;
      }
      n.rank = nir[2*j+1];
      n.rank_idx = -1;
      n.lid_on_rank = -1;
      n.lid_on_rank_idx = -1;
    }
    slmm_assert(ed.me);
  }
  if (cm.halo == 2) extend_halo::collect_gid_rank(*cm.p, cm.ed);
}

typedef std::map<Int, std::set<Int> > Rank2Gids;

void get_rank2gids (const IslMpi& cm, Rank2Gids& rank2rmtgids,
                    Rank2Gids& rank2owngids) {
  const Int myrank = cm.p->rank();
  for (Int i = 0; i < cm.nelemd; ++i) {
    const auto& ed = cm.ed(i);
    for (const auto& n: ed.nbrs) {
      if (n.rank == myrank) continue;
      // I need this rmt gid's lid.
      rank2rmtgids[n.rank].insert(n.gid);
      // This rank needs this gid's lid.
      rank2owngids[n.rank].insert(ed.me->gid);
    }
  }
}

// Fill in nbrs.lid_on_rank, the lid on the remote rank corresponding to the gid
// I share but do not own.
void comm_lid_on_rank (IslMpi& cm, const Rank2Gids& rank2rmtgids,
                       const Rank2Gids& rank2owngids,
                       std::map<Int, Int>& gid2rmt_owning_lid) {
  const Int myrank = cm.p->rank();

  std::map<Int, Int> gid2mylid;
  for (Int i = 0; i < cm.nelemd; ++i)
    gid2mylid[cm.ed(i).me->gid] = i;
  
  // Set up to recv remote (gid, lid) lists.
  Int rn = 0;
  for (const auto& e: rank2rmtgids)
    rn += e.second.size();
  const Int nrecv = rank2rmtgids.size();
  std::vector<Int> recv(rn), recvptr(nrecv+1), recvrank(nrecv);
  std::vector<mpi::Request> reqs(nrecv);
  recvptr[0] = 0;
  Int ir = 0;
  rn = 0;
  for (const auto& e: rank2rmtgids) {
    const Int ne = e.second.size();
    const Int rank = e.first;
    slmm_assert(rank != myrank);
    recvrank[ir] = rank;
    mpi::irecv(*cm.p, recv.data() + rn, ne, rank, 42, &reqs[ir++]);
    rn += ne;
    recvptr[ir] = rn;
  }

  // Send my (gid, lid) lists.
  Int sn = 0, i = 0;
  for (const auto& e: rank2owngids)
    sn += e.second.size();
  std::vector<Int> send(sn);
  std::vector<mpi::Request> sendreqs(rank2owngids.size());
  sn = 0;
  for (const auto& e: rank2owngids) {
    // Iteration through a set gives increasing GID, which means the LID list is
    // consistent between communicating ranks.
    Int slot = sn, pgid = -1;
    for (auto gid: e.second) {
      slmm_assert(gid > pgid);
      pgid = gid;
      send[slot++] = gid2mylid[gid];
    }
    const Int ne = e.second.size();
    const Int rank = e.first;
    mpi::isend(*cm.p, send.data() + sn, ne, rank, 42, &sendreqs[i++]);
    sn += ne;
  }

  // Actually recv.
  const Int count = rank2owngids.size();
  for (Int c = 0; c < count; ++c) {
    int idx;
    mpi::waitany(count, reqs.data(), &idx);
    const Int* rmtlids = recv.data() + recvptr[idx];
    const auto& gids = rank2rmtgids.at(recvrank[idx]);
    slmm_assert(recvptr[idx+1] - recvptr[idx] == gids.size());
    Int i = 0;
    for (auto gid: gids)
      gid2rmt_owning_lid[gid] = rmtlids[i++];
  }

  // Fill lid_on_rank and mylid_with_comm.
  std::vector<Int> mylid_with_comm;
  for (Int i = 0; i < cm.nelemd; ++i) {
    auto& ed = cm.ed(i);
    bool has_comm = false;
    for (auto& n: ed.nbrs)
      if (n.rank == myrank) {
        const auto it = gid2mylid.find(n.gid);
        slmm_throw_if(it == gid2mylid.end(),
                      "comm_lid_on_rank: On rank " << myrank << ", gid " << n.gid
                      << " is not in gid2mylid.");
        n.lid_on_rank = it->second;
      } else {
        has_comm = true;
        const auto it = gid2rmt_owning_lid.find(n.gid);
        slmm_throw_if(it == gid2rmt_owning_lid.end(),
                      "comm_lid_on_rank: On rank " << myrank << ", gid " << n.gid
                      << ", which I believe to be owned by rank " << n.rank
                      << ", is not in gid2rmt_owning_lid.");
        n.lid_on_rank = it->second;
      }
    if (has_comm)
      mylid_with_comm.push_back(i);
  }
  cm.mylid_with_comm.reset_capacity(mylid_with_comm.size(), true);
  for (Int i = 0; i < cm.mylid_with_comm.n(); ++i)
    cm.mylid_with_comm(i) = mylid_with_comm[i];

  mpi::waitall(sendreqs.size(), sendreqs.data());
}

// Useful maps between a linear index space 1:K to a set of K unique
// integers. These obviate sorts and use of hash or binary maps during time
// stepping.
void set_idx2_maps (IslMpi& cm, const Rank2Gids& rank2rmtgids,
                    const std::map<Int, Int>& gid2rmt_owning_lid) {
  const Int myrank = cm.p->rank();
  std::map<Int, Int> ranks;
  Int i = 0;
  for (const auto& e: rank2rmtgids)
    if (ranks.find(e.first) == ranks.end())
      ranks[e.first] = i++;
  ranks[myrank] = i;

  cm.ranks.reset_capacity(i+1, true);
  cm.ranks(i) = myrank;
  for (const auto& e: ranks)
    cm.ranks(e.second) = e.first;
  cm.sendcount.reset_capacity(i, true);
  cm.sendreq.reset_capacity(i, true);
  cm.recvreq.reset_capacity(i, true);

  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
  std::vector<std::map<Int, Int> > lor2idx(nrmtrank);
  std::vector<Int> nlid_on_rank(nrmtrank);
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto& rmtgids = rank2rmtgids.at(cm.ranks(ri));
    nlid_on_rank[ri] = rmtgids.size();
  }
  cm.lid_on_rank.init(nrmtrank, nlid_on_rank.data());
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto& rmtgids = rank2rmtgids.at(cm.ranks(ri));
    auto& lor2idxri = lor2idx[ri];
    Int idx = 0;
    for (const auto& gid: rmtgids) {
      const auto lid = gid2rmt_owning_lid.at(gid);
      lor2idxri[lid] = idx;
      cm.lid_on_rank(ri,idx) = lid;
      ++idx;
    }
    slmm_assert(idx == nlid_on_rank[ri]);
  }

  for (i = 0; i < cm.nelemd; ++i) {
    auto& ed = cm.ed(i);
    ed.me->rank_idx = ranks.at(ed.me->rank);
    ed.me->lid_on_rank_idx = i;
    for (auto& n: ed.nbrs) {
      n.rank_idx = ranks.at(n.rank);
      n.lid_on_rank_idx = n.rank == myrank ?
        i :
        lor2idx[n.rank_idx].at(n.lid_on_rank);
    }
    ed.src = IslMpi::IntArray2D("src", cm.nlev, cm.np2);
    ed.q_extrema = IslMpi::Array<Real**[2]>("q_extrema", cm.qsize, cm.nlev);
  }
}

// In the original MPI pattern that has been in HOMME for years, each owned cell
// has a 1-halo patch of bulk data. For a 1-halo, allocations in this routine
// use essentially the same amount of memory, but not more. We could use less if
// we were willing to realloc space at each SL time step.
void alloc_mpi_buffers (IslMpi& cm, const Rank2Gids& rank2rmtgids,
                        const Rank2Gids& rank2owngids) {
  const auto myrank = cm.p->rank();
  // sizeof real, int, single int (b/c of alignment)
  const Int sor = sizeof(Real), soi = sizeof(Int), sosi = sor;
  static_assert(sizeof(Real) >= sizeof(Int),
                "For buffer packing, we require sizeof(Real) >= sizeof(Int)");
  const auto xbufcnt = [&] (const std::set<Int>& rmtgids,
                            const std::set<Int>& owngids) -> Int {
    return (sosi + (2*soi + (2*soi)*cm.nlev)*rmtgids.size() + // meta data
            owngids.size()*cm.nlev*cm.np2*3*sor);             // bulk data
  };
  const auto qbufcnt = [&] (const std::set<Int>& rmtgids,
                            const std::set<Int>& owngids) -> Int {
    return ((rmtgids.size()*2 +      // min/max q
             owngids.size()*cm.np2)* // q
            cm.qsize*cm.nlev*sor);
  };
  const auto bytes2real = [&] (const Int& bytes) {
    return (bytes + sor - 1)/sor;
  };

  slmm_assert(cm.ranks.back() == myrank);
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
  std::vector<Int> nlid_per_rank(nrmtrank), sendsz(nrmtrank), recvsz(nrmtrank);
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    const auto& rmtgids = rank2rmtgids.at(cm.ranks(ri));
    const auto& owngids = rank2owngids.at(cm.ranks(ri));
    nlid_per_rank[ri] = rmtgids.size();
    sendsz[ri] = bytes2real(std::max(xbufcnt(rmtgids, owngids),
                                     qbufcnt(owngids, rmtgids)));
    recvsz[ri] = bytes2real(std::max(xbufcnt(owngids, rmtgids),
                                     qbufcnt(rmtgids, owngids)));
  }
  cm.nx_in_rank.reset_capacity(nrmtrank, true);
  cm.nx_in_lid.init(nrmtrank, nlid_per_rank.data());
  cm.bla.init(nrmtrank, nlid_per_rank.data(), cm.nlev);
  cm.sendbuf.init(nrmtrank, sendsz.data());
  cm.recvbuf.init(nrmtrank, recvsz.data());
#ifdef HORIZ_OPENMP
  cm.ri_lidi_locks.init(nrmtrank, nlid_per_rank.data());
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    auto&& locks = cm.ri_lidi_locks(ri);
    for (auto& lock: locks)
      omp_init_lock(&lock);
  }
#endif
}

// At simulation initialization, set up a bunch of stuff to make the work at
// each step as small as possible.
void setup_comm_pattern (IslMpi& cm, const Int* nbr_id_rank, const Int* nirptr) {
  collect_gid_rank(cm, nbr_id_rank, nirptr);
  Rank2Gids rank2rmtgids, rank2owngids;
  get_rank2gids(cm, rank2rmtgids, rank2owngids);
  {
    std::map<Int, Int> gid2rmt_owning_lid;
    comm_lid_on_rank(cm, rank2rmtgids, rank2owngids, gid2rmt_owning_lid);
    set_idx2_maps(cm, rank2rmtgids, gid2rmt_owning_lid);
  }
  alloc_mpi_buffers(cm, rank2rmtgids, rank2owngids);
}

// mylid_with_comm(rankidx) is a list of element LIDs that have relations with
// other elements on other ranks. For horizontal threading, need to find the
// subsets that fit within the usual horizontal-threading nets:nete ranges.
void init_mylid_with_comm_threaded (IslMpi& cm, const Int& nets, const Int& nete) {
#ifdef HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    const int nthr = get_num_threads();
    cm.rwork = IslMpi::Array<Real**>("rwork", nthr, cm.qsize);
    cm.mylid_with_comm_tid_ptr.reset_capacity(nthr+1, true);
    cm.horiz_openmp = get_num_threads() > 1;
  }
#ifdef HORIZ_OPENMP
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
#ifdef HORIZ_OPENMP
# pragma omp barrier
#endif
}

IslMpi::Ptr init (const slmm::Advecter::ConstPtr& advecter,
                  const mpi::Parallel::Ptr& p,
                  Int np, Int nlev, Int qsize, Int qsized, Int nelemd,
                  const Int* nbr_id_rank, const Int* nirptr,
                  Int halo) {
  slmm_throw_if(halo < 1 || halo > 2, "halo must be 1 (default) or 2.");
  auto cm = std::make_shared<IslMpi>(p, advecter, np, nlev, qsize, qsized,
                                     nelemd, halo);
  setup_comm_pattern(*cm, nbr_id_rank, nirptr);
  return cm;
}

// For const clarity, take the non-const advecter as an arg, even though cm
// already has a ref to the const'ed one.
void finalize_local_meshes (IslMpi& cm, slmm::Advecter& advecter) {
  if (cm.halo == 2) extend_halo::extend_local_meshes(*cm.p, cm.ed, advecter);
}

// Set pointers to HOMME data arrays.
void set_elem_data (IslMpi& cm, const Int ie, const Real* metdet, const Real* qdp,
                    const Real* dp, Real* q, const Int nelem_in_patch) {
  slmm_assert(ie < cm.ed.size());
  slmm_assert(cm.halo > 1 || cm.ed(ie).nbrs.size() == nelem_in_patch);
  auto& e = cm.ed(ie);
  e.metdet = metdet;
  e.qdp = qdp;
  e.dp = dp;
  e.q = q;
}

void setup_irecv (IslMpi& cm, const bool skip_if_empty = false) {
#ifdef HORIZ_OPENMP
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

void isend (IslMpi& cm, const bool want_req = true, const bool skip_if_empty = false) {
#ifdef HORIZ_OPENMP
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

void recv_and_wait_on_send (IslMpi& cm) {
#ifdef HORIZ_OPENMP
# pragma omp master
#endif
  {
    mpi::waitall(cm.sendreq.n(), cm.sendreq.data());
    mpi::waitall(cm.recvreq.n(), cm.recvreq.data());
  }
#ifdef HORIZ_OPENMP
# pragma omp barrier
#endif
}

void recv (IslMpi& cm, const bool skip_if_empty = false) {
#ifdef HORIZ_OPENMP
# pragma omp master
#endif
  {
    mpi::waitall(cm.recvreq.n(), cm.recvreq.data());
  }
#ifdef HORIZ_OPENMP
# pragma omp barrier
#endif
}

// Find where each departure point is.
void analyze_dep_points (IslMpi& cm, const Int& nets, const Int& nete,
                         const FA4<Real>& dep_points) {
  const auto myrank = cm.p->rank();
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
  cm.bla.zero();
#ifdef HORIZ_OPENMP
# pragma omp for
#endif
  for (Int ri = 0; ri < nrmtrank; ++ri)
    cm.nx_in_lid(ri).zero();
  for (Int tci = nets; tci <= nete; ++tci) {
    const auto& mesh = cm.advecter->local_mesh(tci);
    const auto tgt_idx = mesh.tgt_elem;
    auto& ed = cm.ed(tci);
    ed.own.clear();
    for (Int lev = 0; lev < cm.nlev; ++lev)
      for (Int k = 0; k < cm.np2; ++k) {
        Int sci = slmm::get_src_cell(mesh, &dep_points(0,k,lev,tci), tgt_idx);
        if (sci == -1 && cm.advecter->nearest_point_permitted(lev))
          sci = slmm::get_nearest_point(
            mesh, cm.advecter->nearest_point_data(tci),
            &dep_points(0,k,lev,tci), tgt_idx);
        if (sci == -1) {
          std::stringstream ss;
          ss.precision(17);
          const auto* v = &dep_points(0,k,lev,tci);
          ss << "Departure point is outside of halo:\n"
             << "  nearest point permitted: "
             << cm.advecter->nearest_point_permitted(lev)
             << "\n  elem LID " << tci
             << " elem GID " << ed.me->gid
             << " (lev, k) (" << lev << ", " << k << ")"
             << " v " << v[0] << " " << v[1] << " " << v[2]
             << "\n  tgt_idx " << tgt_idx
             << " local mesh:\n  " << slmm::to_string(mesh) << "\n";
          slmm_throw_if(sci == -1, ss.str());
        }
        ed.src(lev,k) = sci;
        if (ed.nbrs(sci).rank == myrank) {
          ed.own.inc();
          auto& t = ed.own.back();
          t.lev = lev; t.k = k;
        } else {
          const auto ri = ed.nbrs(sci).rank_idx;
          const auto lidi = ed.nbrs(sci).lid_on_rank_idx;
#ifdef HORIZ_OPENMP
          omp_lock_t* lock;
          if (cm.horiz_openmp) {
            lock = &cm.ri_lidi_locks(ri,lidi);
            omp_set_lock(lock);
          }
#endif
          {
            ++cm.nx_in_lid(ri,lidi);
            ++cm.bla(ri,lidi,lev).xptr;
          }
#ifdef HORIZ_OPENMP
          if (cm.horiz_openmp) omp_unset_lock(lock);
#endif
        }
      }
  }
#ifdef HORIZ_OPENMP
# pragma omp barrier
# pragma omp for
#endif
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    auto& nx_in_rank = cm.nx_in_rank(ri);
    nx_in_rank = 0;
    for (Int i = 0, n = cm.lid_on_rank(ri).n(); i < n; ++i)
      nx_in_rank += cm.nx_in_lid(ri,i);
  }
}

static const int nreal_per_2int = (2*sizeof(Int) + sizeof(Real) - 1) / sizeof(Real);

template <typename Buffer>
Int setbuf (Buffer& buf, const Int& os, const Int& i1, const Int& i2) {
  Int* const b = reinterpret_cast<Int*>(&buf(os));
  b[0] = i1;
  b[1] = i2;
  return nreal_per_2int;
}

template <typename Buffer>
Int getbuf (Buffer& buf, const Int& os, Int& i1, Int& i2) {
  const Int* const b = reinterpret_cast<const Int*>(&buf(os));
  i1 = b[0];
  i2 = b[1];
  return nreal_per_2int;
}

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
void pack_dep_points_sendbuf_pass1 (IslMpi& cm) {
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
#ifdef HORIZ_OPENMP
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

void pack_dep_points_sendbuf_pass2 (IslMpi& cm, const FA4<const Real>& dep_points) {
  const auto myrank = cm.p->rank();
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr(tid),
           end = cm.mylid_with_comm_tid_ptr(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm(ptr);
    auto& ed = cm.ed(tci);
    ed.rmt.clear();
    for (Int lev = 0; lev < cm.nlev; ++lev) {
      for (Int k = 0; k < cm.np2; ++k) {
        const Int sci = ed.src(lev,k);
        const auto& nbr = ed.nbrs(sci);
        if (nbr.rank == myrank) continue;
        const Int ri = nbr.rank_idx;
        const Int lidi = nbr.lid_on_rank_idx;
        auto&& sb = cm.sendbuf(ri);
#ifdef HORIZ_OPENMP
        omp_lock_t* lock;
        if (cm.horiz_openmp) {
          lock = &cm.ri_lidi_locks(ri,lidi);
          omp_set_lock(lock);
        }
#endif
        Int xptr, qptr, cnt; {
          auto& t = cm.bla(ri,lidi,lev);
          qptr = t.qptr;
          cnt = t.cnt;
          xptr = t.xptr + 3*cnt;
          ++t.cnt;
        }
#ifdef HORIZ_OPENMP
        if (cm.horiz_openmp) omp_unset_lock(lock);
#endif
        slmm_assert_high(xptr > 0);
        for (Int i = 0; i < 3; ++i)
          sb(xptr + i) = dep_points(i,k,lev,tci);
        ed.rmt.inc();
        auto& item = ed.rmt.back();
        item.q_extrema_ptr = cm.qsize * qptr;
        item.q_ptr = item.q_extrema_ptr + cm.qsize*(2 + cnt);
        item.lev = lev;
        item.k = k;
      }
    }
  }
}

template <Int np>
void calc_q_extrema (IslMpi& cm, const Int& nets, const Int& nete) {
  for (Int tci = nets; tci <= nete; ++tci) {
    auto& ed = cm.ed(tci);
    const FA2<const Real> dp(ed.dp, cm.np2, cm.nlev);
    const FA3<const Real> qdp(ed.qdp, cm.np2, cm.nlev, cm.qsize);
    const FA3<Real> q(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (Int iq = 0; iq < cm.qsize; ++iq)
      for (Int lev = 0; lev < cm.nlev; ++lev) {
        const Real* const dp0 = &dp(0,lev);
        const Real* const qdp0 = &qdp(0,lev,iq);
        Real* const q0 = &q(0,lev,iq);
        Real q_min_s, q_max_s;
        q0[0] = qdp0[0] / dp0[0];
        q_min_s = q_max_s = q0[0];
        for (Int k = 1; k < np*np; ++k) {
          q0[k] = qdp0[k] / dp0[k];
          q_min_s = std::min(q_min_s, q0[k]);
          q_max_s = std::max(q_max_s, q0[k]);
        }
        ed.q_extrema(iq,lev,0) = q_min_s;
        ed.q_extrema(iq,lev,1) = q_max_s;
      }
  }  
}

template <Int np>
void calc_q (const IslMpi& cm, const Int& src_lid, const Int& lev,
             const Real* const dep_point, Real* const q_tgt, const bool use_q) {
  Real ref_coord[2]; {
    const auto& m = cm.advecter->local_mesh(src_lid);
    cm.advecter->s2r().calc_sphere_to_ref(src_lid, m, dep_point,
                                          ref_coord[0], ref_coord[1]);
  }

  // Interpolate.
  Real rx[4], ry[4];
  switch (cm.advecter->alg()) {
  case slmm::Advecter::Alg::csl_gll:
    slmm::gll_np4_eval(ref_coord[0], rx);
    slmm::gll_np4_eval(ref_coord[1], ry);
    break;
  case slmm::Advecter::Alg::csl_gll_subgrid:
    slmm::gll_np4_subgrid_eval(ref_coord[0], rx);
    slmm::gll_np4_subgrid_eval(ref_coord[1], ry);
    break;
  case slmm::Advecter::Alg::csl_gll_exp:
    slmm::gll_np4_subgrid_exp_eval(ref_coord[0], rx);
    slmm::gll_np4_subgrid_exp_eval(ref_coord[1], ry);
    break;
  default:
    slmm_assert(0);
  }

  const auto& ed = cm.ed(src_lid);
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

template <Int np>
void calc_rmt_q (IslMpi& cm) {
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
#ifdef HORIZ_OPENMP
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
      const auto& ed = cm.ed(lid);
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

template <Int np>
void calc_own_q (IslMpi& cm, const Int& nets, const Int& nete,
                 const FA4<const Real>& dep_points,
                 const FA4<Real>& q_min, const FA4<Real>& q_max) {
  const int tid = get_tid();
  for (Int tci = nets; tci <= nete; ++tci) {
    const Int ie0 = tci - nets;
    auto& ed = cm.ed(tci);
    const FA3<Real> q_tgt(ed.q, cm.np2, cm.nlev, cm.qsize);
    for (const auto& e: ed.own) {
      const Int slid = ed.nbrs(ed.src(e.lev, e.k)).lid_on_rank;
      const auto& sed = cm.ed(slid);
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

void copy_q (IslMpi& cm, const Int& nets,
             const FA4<Real>& q_min, const FA4<Real>& q_max) {
  const auto myrank = cm.p->rank();
  const int tid = get_tid();
  for (Int ptr = cm.mylid_with_comm_tid_ptr(tid),
           end = cm.mylid_with_comm_tid_ptr(tid+1);
       ptr < end; ++ptr) {
    const Int tci = cm.mylid_with_comm(ptr);
    const Int ie0 = tci - nets;
    auto& ed = cm.ed(tci);
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

/* dep_points is const in principle, but if lev <=
   semi_lagrange_nearest_point_lev, a departure point may be altered if the
   winds take it outside of the comm halo.
 */
template <int np>
void step (
  IslMpi& cm, const Int nets, const Int nete,
  Cartesian3D* dep_points_r,    // dep_points(1:3, 1:np, 1:np)
  Real* q_min_r, Real* q_max_r) // q_{min,max}(1:np, 1:np, lev, 1:qsize, ie-nets+1)
{
  static_assert(np == 4, "SLMM CSL with special MPI is supported for np 4 only.");
  slmm_assert(cm.np == 4);

  const FA4<Real>
    dep_points(reinterpret_cast<Real*>(dep_points_r),
               3, cm.np2, cm.nlev, cm.nelemd);
  const Int nelem = nete - nets + 1;
  const FA4<Real>
    q_min(q_min_r, cm.np2, cm.nlev, cm.qsize, nelem),
    q_max(q_max_r, cm.np2, cm.nlev, cm.qsize, nelem);

  // Partition my elements that communicate with remotes among threads, if I
  // haven't done that yet.
  if (cm.mylid_with_comm_tid_ptr.n() == 0)
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
  calc_q_extrema<np>(cm, nets, nete);
  // Wait for the departure point requests. Since this requires a thread
  // barrier, at the same time make sure the send buffer is free for use.
  recv_and_wait_on_send(cm);
  // Compute the requested q for departure points from remotes.
  calc_rmt_q<np>(cm);
  // Send q data.
  isend(cm, false /* want_req */, true /* skip_if_empty */);
  // Set up to receive q for each of my departure point requests sent to
  // remotes. We can't do this until the OpenMP barrier in isend assures that
  // all threads are done with the receive buffer's departure points.
  setup_irecv(cm, true /* skip_if_empty */);
  // While waiting to get my data from remotes, compute q for departure points
  // that have remained in my elements.
  calc_own_q<np>(cm, nets, nete, dep_points, q_min, q_max);
  // Receive remote q data and use this to fill in the rest of my fields.
  recv(cm, true /* skip_if_empty */);
  copy_q(cm, nets, q_min, q_max);
  // Don't need to wait on send buffer again because MPI-level synchronization
  // outside of SL transport assures the send buffer is ready at the next call
  // to step. But do need to dealloc the send requests.
}
} // namespace islmpi
} // namespace homme

// Valid after slmm_init_local_mesh_ is called.
int slmm_unittest () {
  int nerr = 0, ne;
  {
    ne = 0;
    for (int i = 0; i < homme::g_advecter->nelem(); ++i) {
      const auto& m = homme::g_advecter->local_mesh(i);
      ne += slmm::unittest(m, m.tgt_elem);
    }
    if (ne)
      fprintf(stderr, "slmm_unittest: slmm::unittest returned %d\n", ne);
    nerr += ne;
  }
  return nerr;
}

#include <cstdlib>

struct Experiment {
  int sl_mpi, halo;
};

template <typename T> T strto(const char* s);
template <> inline int strto (const char* s) { return std::atoi(s); }
template <> inline bool strto (const char* s) { return std::atoi(s); }
template <> inline double strto (const char* s) { return std::atof(s); }
template <> inline std::string strto (const char* s) { return std::string(s); }

static homme::islmpi::IslMpi::Ptr g_csl_mpi;

extern "C" {
void slmm_init_impl (
  homme::Int fcomm, homme::Int transport_alg, homme::Int np,
  homme::Int nlev, homme::Int qsize, homme::Int qsized, homme::Int nelem,
  homme::Int nelemd, homme::Int cubed_sphere_map,
  const homme::Int** lid2gid, const homme::Int** lid2facenum,
  const homme::Int** nbr_id_rank, const homme::Int** nirptr,
  homme::Int sl_nearest_point_lev)
{
  homme::slmm_init(np, nelem, nelemd, transport_alg, cubed_sphere_map,
                   sl_nearest_point_lev - 1, *lid2facenum);
  slmm_throw_if(homme::g_advecter->is_cisl(),
                "CISL code was removed.");
  const auto p = homme::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  g_csl_mpi = homme::islmpi::init(homme::g_advecter, p, np, nlev, qsize,
                                  qsized, nelemd, *nbr_id_rank, *nirptr,
                                  2 /* halo */);
}

void slmm_get_mpi_pattern (homme::Int* sl_mpi) {
  *sl_mpi = g_csl_mpi ? 1 : 0;
}

void slmm_init_local_mesh (
  homme::Int ie, homme::Cartesian3D** neigh_corners, homme::Int nnc,
  homme::Cartesian3D* p_inside)
{
  homme::g_advecter->init_local_mesh_if_needed(
    ie - 1, homme::FA3<const homme::Real>(
      reinterpret_cast<const homme::Real*>(*neigh_corners), 3, 4, nnc),
    reinterpret_cast<const homme::Real*>(p_inside));
}

void slmm_init_finalize () {
  if (g_csl_mpi)
    homme::islmpi::finalize_local_meshes(*g_csl_mpi, *homme::g_advecter);
}

void slmm_check_ref2sphere (homme::Int ie, homme::Cartesian3D* p) {
  homme::g_advecter->check_ref2sphere(
    ie - 1, reinterpret_cast<const homme::Real*>(p));
}

void slmm_csl_set_elem_data (
  homme::Int ie, homme::Real* metdet, homme::Real* qdp, homme::Real* dp,
  homme::Real* q, homme::Int nelem_in_patch)
{
  slmm_assert(g_csl_mpi);
  homme::islmpi::set_elem_data(*g_csl_mpi, ie - 1, metdet, qdp, dp, q,
                               nelem_in_patch);
}

void slmm_csl (
  homme::Int nets, homme::Int nete, homme::Cartesian3D* dep_points,
  homme::Real* minq, homme::Real* maxq, homme::Int* info)
{
  slmm_assert(g_csl_mpi);
  *info = 0;
  try {
    homme::islmpi::step<4>(*g_csl_mpi, nets - 1, nete - 1,
                           dep_points, minq, maxq);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    *info = -1;
  }
}
} // extern "C"
