#include "cedr.hpp"
// Use these when rewriting each CDR's run() function to interact nicely with
// Homme's nested OpenMP and top-level horizontal threading scheme.
#include "cedr_qlt.hpp"
#include "cedr_caas.hpp"

#define THREAD_QLT_RUN
#ifndef QLT_MAIN
# ifdef HAVE_CONFIG_H
#  include "config.h.c"
# endif
#endif
#include "compose.hpp"

namespace homme {
namespace compose {

template <typename ES>
class QLT : public cedr::qlt::QLT<ES> {
  typedef cedr::Int Int;
  typedef cedr::Real Real;
  typedef cedr::qlt::QLT<ES> Super;
  typedef typename Super::RealList RealList;

  //todo All of this VerticalLevelsData-related code should be impl'ed
  // using a new optional root-node-function QLT registration function.
  struct VerticalLevelsData {
    typedef std::shared_ptr<VerticalLevelsData> Ptr;

    RealList lo, hi, mass, ones, wrk;

    VerticalLevelsData (const cedr::Int n)
      : lo("lo", n), hi("hi", n), mass("mass", n), ones("ones", n), wrk("wrk", n)
    {
      for (cedr::Int k = 0; k < n; ++k) ones(k) = 1;
    }
  };

  typename VerticalLevelsData::Ptr vld_;

  void reconcile_vertical (const Int problem_type, const Int bd_os,
                           const Int bis, const Int bie) {
    using cedr::ProblemType;

    cedr_assert((problem_type & ProblemType::shapepreserve) &&
                (problem_type & ProblemType::conserve));

    auto& md = this->md_;
    auto& bd = this->bd_;
    const auto& vld = *vld_;
    const Int nlev = vld.lo.extent_int(0);
    const Int nprob = (bie - bis)/nlev;

#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#   pragma omp master
#endif
    for (Int pi = 0; pi < nprob; ++pi) {
      const Int bd_os_pi = bd_os + md.a_d.trcr2bl2r(md.a_d.bidx2trcr(bis + pi));
#ifdef RV_DIAG
      Real oob = 0, tot_mass_slv = 0, oob_slv = 0;
#endif
      Real tot_mass = 0;
      for (Int k = 0; k < nlev; ++k) {
        const Int bd_os_k = bd_os_pi + nprob*4*k;
        vld.lo  (k) = bd.l2r_data(bd_os_k    );
        vld.hi  (k) = bd.l2r_data(bd_os_k + 2);
        vld.mass(k) = bd.l2r_data(bd_os_k + 3); // previous mass, not current one
        tot_mass += vld.mass(k);
#ifdef RV_DIAG
        if (vld.mass(k) < vld.lo(k)) oob += vld.lo(k) - vld.mass(k);
        if (vld.mass(k) > vld.hi(k)) oob += vld.mass(k) - vld.hi(k);
#endif
      }
      solve(nlev, vld, tot_mass);
      for (Int k = 0; k < nlev; ++k) {
        const Int bd_os_k = bd_os_pi + nprob*4*k;
        bd.l2r_data(bd_os_k + 3) = vld.mass(k); // previous mass, not current one
#ifdef RV_DIAG
        tot_mass_slv += vld.mass(k);
        if (vld.mass(k) < vld.lo(k)) oob_slv += vld.lo(k) - vld.mass(k);
        if (vld.mass(k) > vld.hi(k)) oob_slv += vld.mass(k) - vld.hi(k);
#endif
      }
#ifdef RV_DIAG
      printf("%2d %9.2e %9.2e %9.2e %9.2e\n", pi,
             oob/tot_mass, oob_slv/tot_mass,
             tot_mass, std::abs(tot_mass_slv - tot_mass)/tot_mass);
#endif
    }
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#   pragma omp barrier
#endif
  }

  static Int solve (const Int n, const Real* a, const Real b,
                    const Real* xlo, const Real* xhi,
                    Real* x, Real* wrk) {
#ifndef NDEBUG
    cedr_assert(b >= 0);
    for (Int i = 0; i < n; ++i) {
      cedr_assert(a[i] > 0);
      cedr_assert(xlo[i] >= 0);
      cedr_assert(xhi[i] >= xlo[i]);
    }
#endif
    Int status = 0;
    Real tot_lo = 0, tot_hi = 0;
    for (Int i = 0; i < n; ++i) tot_lo += a[i]*xlo[i];
    for (Int i = 0; i < n; ++i) tot_hi += a[i]*xhi[i];
    if (b < tot_lo) {
      status = -2;
      for (Int i = 0; i < n; ++i) wrk[i] = 0;
      for (Int i = 0; i < n; ++i) x[i] = xlo[i];
      // Find a new xlo >= 0 minimally far from the current one. This
      // is also the solution x.
      cedr::local::caas(n, a, b, wrk, xlo, x, x, false);
    } else if (b > tot_hi) {
      status = -1;
      const Real f = b/tot_hi;
      // a[i] divides out.
      for (Int i = 0; i < n; ++i) x[i] = f*xhi[i];
    } else {
      cedr::local::caas(n, a, b, xlo, xhi, x, x, false);
    }
    return status;
  }

  static Int solve (const Int nlev, const VerticalLevelsData& vld,
                    const Real& tot_mass) {
    return solve(nlev, vld.ones.data(), tot_mass, vld.lo.data(), vld.hi.data(),
                 vld.mass.data(), vld.wrk.data());    
  }

  static Int solve_unittest () {
    static const auto eps = std::numeric_limits<Real>::epsilon();
    static const Int n = 7;

    Real a[n], xlo[n], xhi[n], x[n], wrk[n];
    static const Real x0  [n] = { 1.2, 0.5,3  , 2  , 1.5, 1.8,0.2};
    static const Real dxlo[n] = {-0.1,-0.2,0.5,-1.5,-0.1,-1.1,0.1};
    static const Real dxhi[n] = { 0.1,-0.1,1  ,-0.5, 0.1,-0.2,0.5};
    for (Int i = 0; i < n; ++i) a[i] = i+1;
    for (Int i = 0; i < n; ++i) xlo[i] = x0[i] + dxlo[i];
    for (Int i = 0; i < n; ++i) xhi[i] = x0[i] + dxhi[i];
    Real b, b1;
    Int status, nerr = 0;

    const auto check_mass = [&] () {
      b1 = 0;
      for (Int i = 0; i < n; ++i) b1 += a[i]*x[i];
      if (std::abs(b1 - b) >= 10*eps*b) ++nerr;
    };

    for (Int i = 0; i < n; ++i) x[i] = x0[i];
    b = 0;
    for (Int i = 0; i < n; ++i) b += a[i]*xlo[i];
    b *= 0.9;
    status = solve(n, a, b, xlo, xhi, x, wrk);
    if (status != -2) ++nerr;
    check_mass();
    for (Int i = 0; i < n; ++i) if (x[i] > xhi[i]*(1 + 10*eps)) ++nerr;

    for (Int i = 0; i < n; ++i) x[i] = x0[i];
    b = 0;
    for (Int i = 0; i < n; ++i) b += a[i]*xhi[i];
    b *= 1.1;
    status = solve(n, a, b, xlo, xhi, x, wrk);
    if (status != -1) ++nerr;
    check_mass();
    for (Int i = 0; i < n; ++i) if (x[i] < xlo[i]*(1 - 10*eps)) ++nerr;

    for (Int i = 0; i < n; ++i) x[i] = x0[i];
    b = 0;
    for (Int i = 0; i < n; ++i) b += 0.5*a[i]*(xlo[i] + xhi[i]);
    status = solve(n, a, b, xlo, xhi, x, wrk);
    if (status != 0) ++nerr;
    check_mass();
    for (Int i = 0; i < n; ++i) if (x[i] < xlo[i]*(1 - 10*eps)) ++nerr;
    for (Int i = 0; i < n; ++i) if (x[i] > xhi[i]*(1 + 10*eps)) ++nerr;

    return nerr;
  }

public:
  QLT (const cedr::mpi::Parallel::Ptr& p, const cedr::Int& ncells,
       const cedr::qlt::tree::Node::Ptr& tree, const cedr::CDR::Options& options,
       const cedr::Int& vertical_levels)
    : cedr::qlt::QLT<ES>(p, ncells, tree, options)
  {
    if (vertical_levels)
      vld_ = std::make_shared<VerticalLevelsData>(vertical_levels);
  }

  void run () override {
    static const int mpitag = 42;
    using cedr::Int;
    using cedr::Real;
    using cedr::ProblemType;
    using cedr::qlt::impl::NodeSets;
    namespace mpi = cedr::mpi;
    auto& md_ = this->md_;
    auto& bd_ = this->bd_;
    auto& ns_ = this->ns_;
    auto& p_ = this->p_;
#if ! defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#   pragma omp master
    {
#endif
      // Number of data per slot.
      const Int l2rndps = md_.a_d.prob2bl2r[md_.nprobtypes];
      const Int r2lndps = md_.a_d.prob2br2l[md_.nprobtypes];

      // Leaves to root.
      for (size_t il = 0; il < ns_->levels.size(); ++il) {
        auto& lvl = ns_->levels[il];

        // Set up receives.
        if (lvl.kids.size()) {
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#         pragma omp master
#endif
          {
            for (size_t i = 0; i < lvl.kids.size(); ++i) {
              const auto& mmd = lvl.kids[i];
              mpi::irecv(*p_, &bd_.l2r_data(mmd.offset*l2rndps), mmd.size*l2rndps, mmd.rank,
                         mpitag, &lvl.kids_req[i]);
            }
            mpi::waitall(lvl.kids_req.size(), lvl.kids_req.data());
          }
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#         pragma omp barrier
#endif
        }

        // Combine kids' data.
        if (lvl.nodes.size()) {
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#         pragma omp for
#endif
          for (size_t ni = 0; ni < lvl.nodes.size(); ++ni) {
            const auto lvlidx = lvl.nodes[ni];
            const auto n = ns_->node_h(lvlidx);
            if ( ! n->nkids) continue;
            cedr_kernel_assert(n->nkids == 2);
            // Total density.
            bd_.l2r_data(n->offset*l2rndps) =
              (bd_.l2r_data(ns_->node_h(n->kids[0])->offset*l2rndps) +
               bd_.l2r_data(ns_->node_h(n->kids[1])->offset*l2rndps));
            // Tracers.
            for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
              const Int problem_type = md_.get_problem_type(pti);
              const bool nonnegative = problem_type & ProblemType::nonnegative;
              const bool shapepreserve = problem_type & ProblemType::shapepreserve;
              const bool conserve = problem_type & ProblemType::conserve;
              const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
#if defined THREAD_QLT_RUN && defined COMPOSE_COLUMN_OPENMP
#             pragma omp parallel for
#endif
              for (Int bi = bis; bi < bie; ++bi) {
                const Int bdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
                Real* const me = &bd_.l2r_data(n->offset*l2rndps + bdi);
                const auto kid0 = ns_->node_h(n->kids[0]);
                const auto kid1 = ns_->node_h(n->kids[1]);
                const Real* const k0 = &bd_.l2r_data(kid0->offset*l2rndps + bdi);
                const Real* const k1 = &bd_.l2r_data(kid1->offset*l2rndps + bdi);
                if (nonnegative) {
                  me[0] = k0[0] + k1[0];
                  if (conserve) me[1] = k0[1] + k1[1];
                } else {
                  me[0] = shapepreserve ? k0[0] + k1[0] : cedr::impl::min(k0[0], k1[0]);
                  me[1] = k0[1] + k1[1];
                  me[2] = shapepreserve ? k0[2] + k1[2] : cedr::impl::max(k0[2], k1[2]);
                  if (conserve) me[3] = k0[3] + k1[3] ;
                }
              }
            }
          }
        }

        // Send to parents.
        if (lvl.me.size())
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#       pragma omp master
#endif
        {
          for (size_t i = 0; i < lvl.me.size(); ++i) {
            const auto& mmd = lvl.me[i];
            mpi::isend(*p_, &bd_.l2r_data(mmd.offset*l2rndps), mmd.size*l2rndps,
                       mmd.rank, mpitag);
          }
        }

#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#       pragma omp barrier
#endif
      }

      // Root.
      if ( ! (ns_->levels.empty() || ns_->levels.back().nodes.size() != 1 ||
              ns_->node_h(ns_->levels.back().nodes[0])->parent >= 0)) {
        const auto n = ns_->node_h(ns_->levels.back().nodes[0]);
        for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
          const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
          if (bie == bis) continue;
          const Int problem_type = md_.get_problem_type(pti);
          if (vld_) reconcile_vertical(problem_type, n->offset*l2rndps, bis, bie);
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP && defined COMPOSE_COLUMN_OPENMP
#         pragma omp parallel
#endif
#if defined THREAD_QLT_RUN && (defined COMPOSE_HORIZ_OPENMP || defined COMPOSE_COLUMN_OPENMP)
#         pragma omp for
#endif
          for (Int bi = bis; bi < bie; ++bi) {
            const Int l2rbdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
            const Int r2lbdi = md_.a_d.trcr2br2l(md_.a_d.bidx2trcr(bi));
            // If QLT is enforcing global mass conservation, set the root's r2l Qm
            // value to the l2r Qm_prev's sum; otherwise, copy the l2r Qm value to
            // the r2l one.
            const Int os = (problem_type & ProblemType::conserve ?
                            md_.get_problem_type_l2r_bulk_size(problem_type) - 1 :
                            (problem_type & ProblemType::nonnegative ? 0 : 1));
            bd_.r2l_data(n->offset*r2lndps + r2lbdi) =
              bd_.l2r_data(n->offset*l2rndps + l2rbdi + os);
            if ((problem_type & ProblemType::consistent) &&
                ! (problem_type & ProblemType::shapepreserve)) {
              // Consistent but not shape preserving, so we're solving a dynamic range
              // preservation problem. We now know the global q_{min,max}. Start
              // propagating it leafward.
              bd_.r2l_data(n->offset*r2lndps + r2lbdi + 1) =
                bd_.l2r_data(n->offset*l2rndps + l2rbdi + 0);
              bd_.r2l_data(n->offset*r2lndps + r2lbdi + 2) =
                bd_.l2r_data(n->offset*l2rndps + l2rbdi + 2);
            }
          }
        }
      }

      // Root to leaves.
      for (size_t il = ns_->levels.size(); il > 0; --il) {
        auto& lvl = ns_->levels[il-1];

        if (lvl.me.size()) {
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#         pragma omp master
#endif
          {
            for (size_t i = 0; i < lvl.me.size(); ++i) {
              const auto& mmd = lvl.me[i];
              mpi::irecv(*p_, &bd_.r2l_data(mmd.offset*r2lndps), mmd.size*r2lndps, mmd.rank,
                         mpitag, &lvl.me_recv_req[i]);
            }
            mpi::waitall(lvl.me_recv_req.size(), lvl.me_recv_req.data());
          }
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#         pragma omp barrier
#endif
        }

        // Solve QP for kids' values.
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#       pragma omp for
#endif
        for (size_t ni = 0; ni < lvl.nodes.size(); ++ni) {
          const auto lvlidx = lvl.nodes[ni];
          const auto n = ns_->node_h(lvlidx);
          if ( ! n->nkids) continue;
          for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
            const Int problem_type = md_.get_problem_type(pti);
            const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
#if defined THREAD_QLT_RUN && defined COMPOSE_COLUMN_OPENMP
#           pragma omp parallel for
#endif
            for (Int bi = bis; bi < bie; ++bi) {
              const Int l2rbdi = md_.a_d.trcr2bl2r(md_.a_d.bidx2trcr(bi));
              const Int r2lbdi = md_.a_d.trcr2br2l(md_.a_d.bidx2trcr(bi));
              cedr_assert(n->nkids == 2);
              if ((problem_type & ProblemType::consistent) &&
                  ! (problem_type & ProblemType::shapepreserve)) {
                // Pass q_{min,max} info along. l2r data are updated for use in
                // solve_node_problem. r2l data are updated for use in isend.
                const Real q_min = bd_.r2l_data(n->offset*r2lndps + r2lbdi + 1);
                const Real q_max = bd_.r2l_data(n->offset*r2lndps + r2lbdi + 2);
                bd_.l2r_data(n->offset*l2rndps + l2rbdi + 0) = q_min;
                bd_.l2r_data(n->offset*l2rndps + l2rbdi + 2) = q_max;
                for (Int k = 0; k < 2; ++k) {
                  const auto os = ns_->node_h(n->kids[k])->offset;
                  bd_.l2r_data(os*l2rndps + l2rbdi + 0) = q_min;
                  bd_.l2r_data(os*l2rndps + l2rbdi + 2) = q_max;
                  bd_.r2l_data(os*r2lndps + r2lbdi + 1) = q_min;
                  bd_.r2l_data(os*r2lndps + r2lbdi + 2) = q_max;
                }
              }
              const auto k0 = ns_->node_h(n->kids[0]);
              const auto k1 = ns_->node_h(n->kids[1]);
              cedr::qlt::impl::solve_node_problem(
                problem_type,
                 bd_.l2r_data( n->offset*l2rndps),
                &bd_.l2r_data( n->offset*l2rndps + l2rbdi),
                 bd_.r2l_data( n->offset*r2lndps + r2lbdi),
                 bd_.l2r_data(k0->offset*l2rndps),
                &bd_.l2r_data(k0->offset*l2rndps + l2rbdi),
                 bd_.r2l_data(k0->offset*r2lndps + r2lbdi),
                 bd_.l2r_data(k1->offset*l2rndps),
                &bd_.l2r_data(k1->offset*l2rndps + l2rbdi),
                 bd_.r2l_data(k1->offset*r2lndps + r2lbdi),
                this->options_.prefer_numerical_mass_conservation_to_numerical_bounds);
            }
          }
        }

        // Send.
        if (lvl.kids.size())
#if defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
#       pragma omp master
#endif
        {
          for (size_t i = 0; i < lvl.kids.size(); ++i) {
            const auto& mmd = lvl.kids[i];
            mpi::isend(*p_, &bd_.r2l_data(mmd.offset*r2lndps), mmd.size*r2lndps,
                       mmd.rank, mpitag);
          }
        }
      }
#if ! defined THREAD_QLT_RUN && defined COMPOSE_HORIZ_OPENMP
    }
#endif
  }

  static Int unittest () {
    return solve_unittest();
  }
};

// We explicitly use Kokkos::Serial here so we can run the Kokkos kernels in the
// super class w/o triggering an expecution-space initialization error in
// Kokkos. This complication results from the interaction of Homme's
// COMPOSE_HORIZ_OPENMP threading with Kokkos kernels.
struct CAAS : public cedr::caas::CAAS<Kokkos::Serial> {
  typedef cedr::caas::CAAS<Kokkos::Serial> Super;

  CAAS (const cedr::mpi::Parallel::Ptr& p, const cedr::Int nlclcells,
        const typename Super::UserAllReducer::Ptr& uar)
    : Super(p, nlclcells, uar)
  {}

  void run () override {
#if defined COMPOSE_HORIZ_OPENMP
#   pragma omp master
#endif
    {
      Super::run();
    }
  }
};

} // namespace compose
} // namespace homme

#ifdef QLT_MAIN
int main (int argc, char** argv) {
  int nerr = 0, retval = 0;
  MPI_Init(&argc, &argv);
  auto p = cedr::mpi::make_parallel(MPI_COMM_WORLD);
  srand(p->rank());
  Kokkos::initialize(argc, argv);
#if 0
  try
#endif
  {
    cedr::InputParser inp(argc, argv, p);
    if (p->amroot()) inp.print(std::cout);
    if (inp.qin.unittest) {
      nerr += cedr::local::unittest();
      nerr += cedr::caas::test::unittest(p);
    }
    if (inp.qin.unittest || inp.qin.perftest)
      nerr += cedr::qlt::test::run_unit_and_randomized_tests(p, inp.qin);
    if (inp.tin.ncells > 0)
      nerr += cedr::test::transport1d::run(p, inp.tin);
    {
      int gnerr;
      cedr::mpi::all_reduce(*p, &nerr, &gnerr, 1, MPI_SUM);
      retval = gnerr != 0 ? -1 : 0;
      if (p->amroot())
        std::cout << (gnerr != 0 ? "FAIL" : "PASS") << "\n";
    }
  }
#if 0
  catch (const std::exception& e) {
    if (p->amroot())
      std::cerr << e.what();
    retval = -1;
  }
#endif
  Kokkos::finalize();
  if (nerr) prc(nerr);
  MPI_Finalize();
  return retval;
}
#endif

namespace homme {
namespace qlt = cedr::qlt;
using cedr::Int;
using cedr::Real;

Int rank2sfc_search (const Int* rank2sfc, const Int& nrank, const Int& sfc) {
  Int lo = 0, hi = nrank+1;
  while (hi > lo + 1) {
    const Int mid = (lo + hi)/2;
    if (sfc >= rank2sfc[mid])
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

// Change leaf node->cellidx from index into space-filling curve to global cell
// index. owned_ids is in SFC index order for this rank.
void renumber (const Int nrank, const Int nelem, const Int my_rank, const Int* owned_ids,
               const Int* rank2sfc, const qlt::tree::Node::Ptr& node) {
  if (node->nkids) {
    for (Int k = 0; k < node->nkids; ++k)
      renumber(nrank, nelem, my_rank, owned_ids, rank2sfc, node->kids[k]);
  } else {
    const Int sfc = node->cellidx;
    node->cellidx = node->rank == my_rank ? owned_ids[sfc - rank2sfc[my_rank]] : -1;
    cedr_assert((node->rank != my_rank && node->cellidx == -1) ||
                (node->rank == my_rank && node->cellidx >= 0 && node->cellidx < nelem));
  }
}

void renumber (const Int* sc2gci, const Int* sc2rank,
               const qlt::tree::Node::Ptr& node) {
  if (node->nkids) {
    for (Int k = 0; k < node->nkids; ++k)
      renumber(sc2gci, sc2rank, node->kids[k]);
  } else {
    const Int ci = node->cellidx;
    node->cellidx = sc2gci[ci];
    node->rank = sc2rank[ci];
  }
}

// Build a subtree over [0, nsublev).
void add_sub_levels (const qlt::tree::Node::Ptr& node, const Int nsublev,
                     const Int gci, const Int my_rank, const Int rank,
                     const bool calc_level, const Int slb, const Int sle) {
  if (slb+1 == sle) {
    node->cellidx = rank == my_rank ? nsublev*gci + slb : -1;
    if (calc_level) node->level = 0;
  } else {
    node->nkids = 2;
    for (Int k = 0; k < 2; ++k) {
      auto kid = std::make_shared<qlt::tree::Node>();
      kid->parent = node.get();
      kid->rank = rank;
      node->kids[k] = kid;
    }
    const Int mid = slb + (sle - slb)/2;
    add_sub_levels(node->kids[0], nsublev, gci, my_rank, rank, calc_level, slb, mid);
    add_sub_levels(node->kids[1], nsublev, gci, my_rank, rank, calc_level, mid, sle);
    if (calc_level)
      node->level = 1 + std::max(node->kids[0]->level, node->kids[1]->level);
  }
}

// Recurse to each leaf and call add_sub_levels above.
void add_sub_levels (const Int my_rank, const qlt::tree::Node::Ptr& node,
                     const Int nsublev, const Int level_offset) {
  if (node->nkids) {
    for (Int k = 0; k < node->nkids; ++k)
      add_sub_levels(my_rank, node->kids[k], nsublev, level_offset);
    node->level += level_offset;
  } else {
    const Int gci = node->cellidx;
    const Int rank = node->rank;
    add_sub_levels(node, nsublev, gci, my_rank, rank, level_offset, 0, nsublev);
    // Level already calculated if requested.
    cedr_assert(level_offset == 0 || node->level == level_offset);
  }
}

// Tree for a 1-D periodic domain, for unit testing.
namespace oned {
struct Mesh {
  Mesh (const Int nc, const cedr::mpi::Parallel::Ptr& p) {
    init(nc, p);
  }
  
  void init (const Int nc, const cedr::mpi::Parallel::Ptr& p) {
    nc_ = nc;
    nranks_ = p->size();
    p_ = p;
    cedr_throw_if(nranks_ > nc_, "#GIDs < #ranks is not supported.");
  }

  Int ncell () const { return nc_; }

  const cedr::mpi::Parallel::Ptr& parallel () const { return p_; }

  Int rank (const Int& ci) const {
    return std::min(nranks_ - 1, ci / (nc_ / nranks_));
  }

private:
  Int nc_, nranks_;
  cedr::mpi::Parallel::Ptr p_;
};
} // namespace oned

// This impl carefully follows the requirements that
// cedr::qlt::impl::init_tree, level_schedule_and_collect
// establish. init_tree has to be modified to have the condition in
// the line
//   if (node->rank < 0) node->rank = node->kids[0]->rank
// since here we're assigning the ranks ourselves. Similarly, it must
// check for node->level >= 0; if the tree is partial, it is unable to
// compute node level.
qlt::tree::Node::Ptr
make_my_tree_part (const oned::Mesh& m, const Int cs, const Int ce,
                   const qlt::tree::Node* parent,
                   const Int& nrank, const Int* rank2sfc) {
  const auto my_rank = m.parallel()->rank();
  const Int cn = ce - cs, cn0 = cn/2;
  qlt::tree::Node::Ptr n = std::make_shared<qlt::tree::Node>();
  n->parent = parent;
  n->rank = rank2sfc_search(rank2sfc, nrank, cs);
  n->cellidx = n->rank == my_rank ? cs : -1;
  cedr_assert(n->rank >= 0 && n->rank < nrank);
  if (cn == 1) {
    n->nkids = 0;
    n->level = 0;
    return n;
  }
  const auto k1 = make_my_tree_part(m, cs, cs + cn0, n.get(), nrank, rank2sfc);
  const auto k2 = make_my_tree_part(m, cs + cn0, ce, n.get(), nrank, rank2sfc);
  n->level = 1 + std::max(k1->level, k2->level);
  if (n->rank == my_rank) {
    // Need to know both kids for comm.
    n->nkids = 2;
    n->kids[0] = k1;
    n->kids[1] = k2;
  } else {
    // Prune parts of the tree irrelevant to my rank.
    n->nkids = 0;
    if (k1->nkids > 0 || k1->rank == my_rank) n->kids[n->nkids++] = k1;
    if (k2->nkids > 0 || k2->rank == my_rank) n->kids[n->nkids++] = k2;
    if (n->nkids == 0) {
      // Signal a non-leaf node with 0 kids to init_tree.
      n->nkids = -1;
    }
  }
  cedr_assert(n->level > 0 || n->nkids == 0);
  return n;
}

qlt::tree::Node::Ptr
make_my_tree_part (const cedr::mpi::Parallel::Ptr& p, const Int& ncells,
                   const Int& nrank, const Int* rank2sfc) {
  oned::Mesh m(ncells, p);
  return make_my_tree_part(m, 0, m.ncell(), nullptr, nrank, rank2sfc);
}

static size_t nextpow2 (size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

static size_t get_tree_height (size_t nleaf) {
  size_t height = 0;
  nleaf = nextpow2(nleaf);
  while (nleaf) {
    ++height;
    nleaf >>= 1;
  }
  return height;
}

qlt::tree::Node::Ptr
make_tree_sgi (const cedr::mpi::Parallel::Ptr& p, const Int nelem,
               const Int* owned_ids, const Int* rank2sfc, const Int nsublev) {
  // Partition 0:nelem-1, the space-filling curve space.
  auto tree = make_my_tree_part(p, nelem, p->size(), rank2sfc);
  // Renumber so that node->cellidx records the global element number, and
  // associate the correct rank with the element.
  const auto my_rank = p->rank();
  renumber(p->size(), nelem, my_rank, owned_ids, rank2sfc, tree);
  if (nsublev > 1) {
    const Int level_offset = get_tree_height(nsublev) - 1;
    add_sub_levels(my_rank, tree, nsublev, level_offset);
  }
  return tree;
}

qlt::tree::Node::Ptr
make_tree_non_sgi (const cedr::mpi::Parallel::Ptr& p, const Int nelem,
                   const Int* sc2gci, const Int* sc2rank, const Int nsublev) {
  auto tree = qlt::tree::make_tree_over_1d_mesh(p, nelem);
  renumber(sc2gci, sc2rank, tree);
  const auto my_rank = p->rank();
  if (nsublev > 1) add_sub_levels(my_rank, tree, nsublev, 0);
  return tree;
}

qlt::tree::Node::Ptr
clone (const qlt::tree::Node::Ptr& in, const qlt::tree::Node* parent = nullptr) {
  const auto out = std::make_shared<qlt::tree::Node>(*in);
  cedr_assert(out->rank == in->rank && out->level == in->level &&
              out->nkids == in->nkids && out->cellidx == in->cellidx);
  out->parent = parent;
  for (Int k = 0; k < in->nkids; ++k)
    out->kids[k] = clone(in->kids[k], out.get());
  return out;
}

void renumber_leaves (const qlt::tree::Node::Ptr& node, const Int horiz_nleaf,
                      const Int supidx) {
  if (node->nkids) {
    for (Int k = 0; k < node->nkids; ++k)
      renumber_leaves(node->kids[k], horiz_nleaf, supidx);
  } else {
    if (node->cellidx != -1) {
      cedr_assert(node->cellidx >= 0 && node->cellidx < horiz_nleaf);
      node->cellidx += horiz_nleaf*supidx;
    }
  }
}

void attach_and_renumber_horizontal_trees (const qlt::tree::Node::Ptr& supnode,
                                           const qlt::tree::Node::Ptr& htree,
                                           const Int horiz_nleaf) {
  Int level = -1, rank;
  for (Int k = 0; k < supnode->nkids; ++k) {
    auto& kid = supnode->kids[k];
    if (kid->nkids) {
      attach_and_renumber_horizontal_trees(kid, htree, horiz_nleaf);
    } else {
      const auto supidx = kid->cellidx;
      supnode->kids[k] = clone(htree);
      kid = supnode->kids[k];
      kid->parent = supnode.get();
      kid->cellidx = -1;
      renumber_leaves(kid, horiz_nleaf, supidx);
    }
    rank = kid->rank;
    level = std::max(level, kid->level);
  }
  if (level != -1) ++level;
  supnode->level = level;
  supnode->rank = rank;
}

qlt::tree::Node::Ptr
make_tree_over_index_range (const Int cs, const Int ce,
                            const qlt::tree::Node* parent = nullptr) {
  const Int cn = ce - cs, cn0 = cn/2;
  const auto n = std::make_shared<qlt::tree::Node>();
  n->parent = parent;
  if (cn == 1) {
    n->nkids = 0;
    n->cellidx = cs;
  } else {
    n->nkids = 2;
    n->kids[0] = make_tree_over_index_range(cs, cs + cn0, n.get());
    n->kids[1] = make_tree_over_index_range(cs + cn0, ce, n.get());
  }
  return n;
}

qlt::tree::Node::Ptr
combine_superlevels(const qlt::tree::Node::Ptr& horiz_tree, const Int horiz_nleaf,
                    const Int nsuplev) {
  cedr_assert(horiz_tree->nkids > 0);
  // In this tree, cellidx 0 is the top super level.
  const auto suptree = make_tree_over_index_range(0, nsuplev);
  attach_and_renumber_horizontal_trees(suptree, horiz_tree, horiz_nleaf);
  return suptree;
}

void check_tree (const cedr::mpi::Parallel::Ptr& p, const qlt::tree::Node::Ptr& n,
                 const Int nleaf) {
#ifndef NDEBUG
  cedr_assert(n->nkids >= -1 && n->nkids <= 2);
  cedr_assert(n->rank >= 0);
  cedr_assert(n->reserved == -1);
  if (n->nkids == 2)
    cedr_assert(n->level == 1 + std::max(n->kids[0]->level, n->kids[1]->level));
  if (n->nkids == 1) cedr_assert(n->level >= 1 + n->kids[0]->level);
  if (n->nkids == 0) cedr_assert(n->level == 0);
  if (n->rank != p->rank()) cedr_assert(n->cellidx == -1);
  else cedr_assert(n->cellidx < nleaf);
  for (Int k = 0; k < n->nkids; ++k) {
    cedr_assert(n.get() == n->kids[k]->parent);
    check_tree(p, n->kids[k], nleaf);
  }
#endif
}

qlt::tree::Node::Ptr
make_tree (const cedr::mpi::Parallel::Ptr& p, const Int nelem,
           const Int* gid_data, const Int* rank_data, const Int nsublev,
           const bool use_sgi, const bool cdr_over_super_levels,
           const Int nsuplev) {
  auto tree = use_sgi ?
    make_tree_sgi    (p, nelem, gid_data, rank_data, nsublev) :
    make_tree_non_sgi(p, nelem, gid_data, rank_data, nsublev);
  Int nleaf = nelem*nsublev;
  if (cdr_over_super_levels) {
    tree = combine_superlevels(tree, nleaf, nsuplev);
    nleaf *= nsuplev;
  }
  if (use_sgi) check_tree(p, tree, nleaf);
  return tree;
}

Int test_tree_maker () {
  Int nerr = 0;
  if (nextpow2(3) != 4) ++nerr;
  if (nextpow2(4) != 4) ++nerr;
  if (nextpow2(5) != 8) ++nerr;
  if (get_tree_height(3) != 3) ++nerr;
  if (get_tree_height(4) != 3) ++nerr;
  if (get_tree_height(5) != 4) ++nerr;
  if (get_tree_height(8) != 4) ++nerr;
  return nerr;
}

extern "C"
void compose_repro_sum(const Real* send, Real* recv,
                       Int nlocal, Int nfld, Int fcomm);

struct ReproSumReducer :
    public compose::CAAS::UserAllReducer {
  ReproSumReducer (Int fcomm) : fcomm_(fcomm) {}

  int operator() (const cedr::mpi::Parallel& p, Real* sendbuf, Real* rcvbuf,
                  int nlocal, int count, MPI_Op op) const override {
    cedr_assert(op == MPI_SUM);
    compose_repro_sum(sendbuf, rcvbuf, nlocal, count, fcomm_);
    return 0;
  }

private:
  const Int fcomm_;
};

struct CDR {
  typedef std::shared_ptr<CDR> Ptr;
  typedef compose::QLT<Kokkos::DefaultExecutionSpace> QLTT;
  typedef compose::CAAS CAAST;

  struct Alg {
    enum Enum { qlt, qlt_super_level, qlt_super_level_local_caas, caas, caas_super_level };
    static Enum convert (Int cdr_alg) {
      switch (cdr_alg) {
      case 2:  return qlt;
      case 20: return qlt_super_level;
      case 21: return qlt_super_level_local_caas;
      case 3:  return caas;
      case 30: return caas_super_level;
      case 42: return caas_super_level; // actually none
      default: cedr_throw_if(true,  "cdr_alg " << cdr_alg << " is invalid.");
      }
    }
    static bool is_qlt (Enum e) {
      return (e == qlt || e == qlt_super_level ||
              e == qlt_super_level_local_caas);
    }
    static bool is_caas (Enum e) {
      return e == caas || e == caas_super_level;
    }
    static bool is_suplev (Enum e) {
      return (e == qlt_super_level || e == caas_super_level ||
              e == qlt_super_level_local_caas);
    }
  };

  enum { nsublev_per_suplev = 8 };
  
  const Alg::Enum alg;
  const Int ncell, nlclcell, nlev, nsublev, nsuplev;
  const bool threed, cdr_over_super_levels, caas_in_suplev, hard_zero;
  const cedr::mpi::Parallel::Ptr p;
  qlt::tree::Node::Ptr tree; // Don't need this except for unit testing.
  cedr::CDR::Ptr cdr;
  std::vector<Int> ie2gci; // Map Homme ie to Homme global cell index.
  std::vector<Int> ie2lci; // Map Homme ie to CDR local cell index (lclcellidx).
  std::vector<char> nonneg;

  CDR (Int cdr_alg_, Int ngblcell_, Int nlclcell_, Int nlev_, bool use_sgi,
       bool independent_time_steps, const bool hard_zero_, const Int* gid_data,
       const Int* rank_data, const cedr::mpi::Parallel::Ptr& p_, Int fcomm)
    : alg(Alg::convert(cdr_alg_)),
      ncell(ngblcell_), nlclcell(nlclcell_), nlev(nlev_),
      nsublev(Alg::is_suplev(alg) ? nsublev_per_suplev : 1),
      nsuplev((nlev + nsublev - 1) / nsublev),
      threed(independent_time_steps),
      cdr_over_super_levels(threed && Alg::is_caas(alg)),
      caas_in_suplev(alg == Alg::qlt_super_level_local_caas && nsublev > 1),
      hard_zero(hard_zero_),
      p(p_), inited_tracers_(false)
  {
    const Int n_id_in_suplev = caas_in_suplev ? 1 : nsublev;
    if (Alg::is_qlt(alg)) {
      tree = make_tree(p, ncell, gid_data, rank_data, n_id_in_suplev, use_sgi,
                       cdr_over_super_levels, nsuplev);
      cedr::CDR::Options options;
      options.prefer_numerical_mass_conservation_to_numerical_bounds = true;
      Int nleaf = ncell*n_id_in_suplev;
      if (cdr_over_super_levels) nleaf *= nsuplev;
      cdr = std::make_shared<QLTT>(p, nleaf, tree, options,
                                   threed ? nsuplev : 0);
      tree = nullptr;
    } else if (Alg::is_caas(alg)) {
      const auto caas = std::make_shared<CAAST>(
        p, nlclcell*n_id_in_suplev*(cdr_over_super_levels ? nsuplev : 1),
        std::make_shared<ReproSumReducer>(fcomm));
      cdr = caas;
    } else {
      cedr_throw_if(true, "Invalid semi_lagrange_cdr_alg " << alg);
    }
    ie2gci.resize(nlclcell);
  }

  void init_tracers (const Int qsize, const bool need_conservation) {
    nonneg.resize(qsize, hard_zero);
    typedef cedr::ProblemType PT;
    const Int nt = cdr_over_super_levels ? qsize : nsuplev*qsize;
    for (Int ti = 0; ti < nt; ++ti)
      cdr->declare_tracer(PT::shapepreserve |
                          (need_conservation ? PT::conserve : 0), 0);
    cdr->end_tracer_declarations();
  }

  void get_buffers_sizes (size_t& s1, size_t &s2) {
    cdr->get_buffers_sizes(s1, s2);
  }

  void set_buffers (Real* b1, Real* b2) {
    cdr->set_buffers(b1, b2);
    cdr->finish_setup();
  }

private:
  bool inited_tracers_;
};

void set_ie2gci (CDR& q, const Int ie, const Int gci) { q.ie2gci[ie] = gci; }

void init_ie2lci (CDR& q) {
  const Int n_id_in_suplev = q.caas_in_suplev ? 1 : q.nsublev;
  const Int nleaf =
    n_id_in_suplev*
    q.ie2gci.size()*
    (q.cdr_over_super_levels ? q.nsuplev : 1);
  q.ie2lci.resize(nleaf);
  if (CDR::Alg::is_qlt(q.alg)) {
    auto qlt = std::static_pointer_cast<CDR::QLTT>(q.cdr);
    if (q.cdr_over_super_levels) {
      const auto nlevwrem = q.nsuplev*n_id_in_suplev;
      for (size_t ie = 0; ie < q.ie2gci.size(); ++ie)
        for (Int spli = 0; spli < q.nsuplev; ++spli)
          for (Int sbli = 0; sbli < n_id_in_suplev; ++sbli)
            //       local indexing is fastest over the whole column
            q.ie2lci[nlevwrem*ie + n_id_in_suplev*spli + sbli] =
              //           but global indexing is organized according to the tree
              qlt->gci2lci(n_id_in_suplev*(q.ncell*spli + q.ie2gci[ie]) + sbli);
    } else {
      for (size_t ie = 0; ie < q.ie2gci.size(); ++ie)
        for (Int sbli = 0; sbli < n_id_in_suplev; ++sbli)
          q.ie2lci[n_id_in_suplev*ie + sbli] =
            qlt->gci2lci(n_id_in_suplev*q.ie2gci[ie] + sbli);
    }
  } else {
    if (q.cdr_over_super_levels) {
      const auto nlevwrem = q.nsuplev*n_id_in_suplev;
      for (size_t ie = 0; ie < q.ie2gci.size(); ++ie)
        for (Int spli = 0; spli < q.nsuplev; ++spli)
          for (Int sbli = 0; sbli < n_id_in_suplev; ++sbli) {
            const Int id = nlevwrem*ie + n_id_in_suplev*spli + sbli;
            q.ie2lci[id] = id;
          }
    } else {
      for (size_t ie = 0; ie < q.ie2gci.size(); ++ie)
        for (Int sbli = 0; sbli < n_id_in_suplev; ++sbli) {
          const Int id = n_id_in_suplev*ie + sbli;
          q.ie2lci[id] = id;
        }
    }
  }
}

void init_tracers (CDR& q, const Int nlev, const Int qsize,
                   const bool need_conservation) {
  q.init_tracers(qsize, need_conservation);
}

namespace sl { // For sl_advection.F90
// Fortran array wrappers.
template <typename T> using FA2 =
  Kokkos::View<T**,    Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA4 =
  Kokkos::View<T****,  Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <typename T> using FA5 =
  Kokkos::View<T*****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Following are naming conventions in element_state and sl_advection:
//     elem(ie)%state%Q(:,:,k,q) is tracer mixing ratio.
//     elem(ie)%state%dp3d(:,:,k,tl%np1) is essentially total density.
//     elem(ie)%state%Qdp(:,:,k,q,n0_qdp) is Q*dp3d.
//     rho(:,:,k,ie) is spheremp*dp3d, essentially total mass at a GLL point.
//     Hence Q*rho = Q*spheremp*dp3d is tracer mass at a GLL point.
// We need to get pointers to some of these; elem can't be given the bind(C)
// attribute, so we can't take the elem array directly. We get these quantities
// at a mix of previous and current time steps.
//   In the code that follows, _p is previous and _c is current time step. Q is
// renamed to q, and Q is tracer mass at a GLL point.
struct Data {
  typedef std::shared_ptr<Data> Ptr;
  const Int np, nlev, qsize, qsize_d, timelevels;
  Int n0_qdp, n1_qdp, tl_np1;
  std::vector<const Real*> spheremp, dp3d_c;
  std::vector<Real*> q_c, qdp_pc;
  const Real* dp0;

  struct Check {
    Kokkos::View<Real**, Kokkos::Serial>
      mass_p, mass_c, mass_lo, mass_hi,
      q_lo, q_hi, q_min_l, q_max_l, qd_lo, qd_hi;
    Check (const Int nlev, const Int qsize)
      : mass_p("mass_p", nlev, qsize), mass_c("mass_c", nlev, qsize),
        mass_lo("mass_lo", nlev, qsize), mass_hi("mass_hi", nlev, qsize),
        q_lo("q_lo", nlev, qsize), q_hi("q_hi", nlev, qsize),
        q_min_l("q_min_l", nlev, qsize), q_max_l("q_max_l", nlev, qsize),
        qd_lo("qd_lo", nlev, qsize), qd_hi("qd_hi", nlev, qsize)
    {}
  };
  std::shared_ptr<Check> check;

  Data (Int lcl_ncell, Int np_, Int nlev_, Int qsize_, Int qsize_d_, Int timelevels_)
    : np(np_), nlev(nlev_), qsize(qsize_), qsize_d(qsize_d_), timelevels(timelevels_),
      spheremp(lcl_ncell, nullptr), dp3d_c(lcl_ncell, nullptr), q_c(lcl_ncell, nullptr),
      qdp_pc(lcl_ncell, nullptr)
  {}
};

static void check (const CDR& q, const Data& d) {
  cedr_assert(q.nlclcell == static_cast<Int>(d.spheremp.size()));
}

template <typename T>
void insert (std::vector<T*>& r, const Int i, T* v) {
  cedr_assert(i >= 0 && i < static_cast<int>(r.size()));
  r[i] = v;
}

void insert (const Data::Ptr& d, const Int ie, const Int ptridx, Real* array,
             const Int i0 = 0, const Int i1 = 0) {
  cedr_assert(d);
  switch (ptridx) {
  case 0: insert<const double>(d->spheremp, ie, array); break;
  case 1: insert<      double>(d->qdp_pc,   ie, array); d->n0_qdp = i0; d->n1_qdp = i1; break;
  case 2: insert<const double>(d->dp3d_c,   ie, array); d->tl_np1 = i0; break;
  case 3: insert<      double>(d->q_c,      ie, array); break;
  case 4: d->dp0 = array; break;
  default: cedr_throw_if(true, "Invalid pointer index " << ptridx);
  }
}

static void run_cdr (CDR& q) {
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
  q.cdr->run();
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
#endif
}

void accum_values (const Int ie, const Int k, const Int q, const Int tl_np1,
                   const Int n0_qdp, const Int np, const bool nonneg,
                   const FA2<const Real>& spheremp, const FA4<const Real>& dp3d_c,
                   const FA5<Real>& q_min, const FA5<const Real>& q_max,
                   const FA5<const Real>& qdp_p, const FA4<const Real>& q_c,
                   Real& volume, Real& rhom, Real& Qm, Real& Qm_prev,
                   Real& Qm_min, Real& Qm_max) {
  for (Int j = 0; j < np; ++j) {
    for (Int i = 0; i < np; ++i) {
      volume += spheremp(i,j); // * dp0[k];
      const Real rhomij = dp3d_c(i,j,k,tl_np1) * spheremp(i,j);
      rhom += rhomij;
      Qm += q_c(i,j,k,q) * rhomij;
      if (nonneg) q_min(i,j,k,q,ie) = std::max<Real>(q_min(i,j,k,q,ie), 0);
      Qm_min += q_min(i,j,k,q,ie) * rhomij;
      Qm_max += q_max(i,j,k,q,ie) * rhomij;
      Qm_prev += qdp_p(i,j,k,q,n0_qdp) * spheremp(i,j);
    }
  }
}

void run (CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
          const Int nets, const Int nete) {
  static constexpr Int max_np = 4;
  const Int np = d.np, nlev = d.nlev, qsize = d.qsize,
    nlevwrem = cdr.nsuplev*cdr.nsublev;
  cedr_assert(np <= max_np);
  
  FA5<      Real> q_min(q_min_r, np, np, nlev, qsize, nete+1);
  FA5<const Real> q_max(q_max_r, np, np, nlev, qsize, nete+1);

  for (Int ie = nets; ie <= nete; ++ie) {
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<const Real> qdp_p(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<const Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
#ifdef COMPOSE_COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const bool nonneg = cdr.nonneg[q];
        const Int ti = cdr.cdr_over_super_levels ? q : spli*qsize + q;
        Real Qm = 0, Qm_min = 0, Qm_max = 0, Qm_prev = 0, rhom = 0, volume = 0;
        Int ie_idx;
        if (cdr.caas_in_suplev)
          ie_idx = cdr.cdr_over_super_levels ?
            cdr.nsuplev*ie + spli :
            ie;
        for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
          const auto k = k0 + sbli;
          if ( ! cdr.caas_in_suplev)
            ie_idx = cdr.cdr_over_super_levels ?
              nlevwrem*ie + k :
              cdr.nsublev*ie + sbli;
          const auto lci = cdr.ie2lci[ie_idx];
          if ( ! cdr.caas_in_suplev) {
            Qm = 0; Qm_min = 0; Qm_max = 0; Qm_prev = 0;
            rhom = 0;
            volume = 0;
          }
          if (k < nlev)
            accum_values(ie, k, q, d.tl_np1, d.n0_qdp, np, nonneg,
                         spheremp, dp3d_c, q_min, q_max, qdp_p, q_c,
                         volume, rhom, Qm, Qm_prev, Qm_min, Qm_max);
          const bool write = ! cdr.caas_in_suplev || sbli == cdr.nsublev-1;
          if (write) {
            // For now, handle just one rhom. For feasible global problems,
            // it's used only as a weight vector in QLT, so it's fine. In fact,
            // use just the cell geometry, rather than total density, since in QLT
            // this field is used as a weight vector.
            //todo Generalize to one rhom field per level. Until then, we're not
            // getting QLT's safety benefit.
            if (ti == 0) cdr.cdr->set_rhom(lci, 0, volume);
            cdr.cdr->set_Qm(lci, ti, Qm, Qm_min, Qm_max, Qm_prev);
            if (Qm_prev < -0.5) {
              static bool first = true;
              if (first) {
                first = false;
                std::stringstream ss;
                ss << "Qm_prev < -0.5: Qm_prev = " << Qm_prev
                   << " on rank " << cdr.p->rank()
                   << " at (ie,gid,spli,k0,q,ti,sbli,lci,k,n0_qdp,tl_np1) = ("
                   << ie << "," << cdr.ie2gci[ie] << "," << spli << "," << k0 << ","
                   << q << "," << ti << "," << sbli << "," << lci << "," << k << ","
                   << d.n0_qdp << "," << d.tl_np1 << ")\n";
                ss << "Qdp(:,:,k,q,n0_qdp) = [";
                for (Int j = 0; j < np; ++j)
                  for (Int i = 0; i < np; ++i)
                    ss << " " << qdp_p(i,j,k,q,d.n0_qdp);
                ss << "]\n";
                ss << "dp3d(:,:,k,tl_np1) = [";
                for (Int j = 0; j < np; ++j)
                  for (Int i = 0; i < np; ++i)
                    ss << " " << dp3d_c(i,j,k,d.tl_np1);
                ss << "]\n";
                pr(ss.str());
              }
            }
          }
        }
      }
    }
  }

  run_cdr(cdr);
}

void solve_local (const Int ie, const Int k, const Int q,
                  const Int tl_np1, const Int n1_qdp, const Int np, 
                  const bool scalar_bounds, const Int limiter_option,
                  const FA2<const Real>& spheremp, const FA4<const Real>& dp3d_c,
                  const FA5<const Real>& q_min, const FA5<const Real>& q_max,
                  const Real Qm, FA5<Real>& qdp_c, FA4<Real>& q_c) {
  static constexpr Int max_np = 4, max_np2 = max_np*max_np;
  const Int np2 = np*np;
  cedr_assert(np <= max_np);

  Real wa[max_np2], qlo[max_np2], qhi[max_np2], y[max_np2], x[max_np2];
  Real rhom = 0;
  for (Int j = 0, cnt = 0; j < np; ++j)
    for (Int i = 0; i < np; ++i, ++cnt) {
      const Real rhomij = dp3d_c(i,j,k,tl_np1) * spheremp(i,j);
      rhom += rhomij;
      wa[cnt] = rhomij;
      y[cnt] = q_c(i,j,k,q);
      x[cnt] = y[cnt];
    }

  //todo Replace with ReconstructSafely.
  if (scalar_bounds) {
    qlo[0] = q_min(0,0,k,q,ie);
    qhi[0] = q_max(0,0,k,q,ie);
    const Int N = std::min(max_np2, np2);
    for (Int i = 1; i < N; ++i) qlo[i] = qlo[0];
    for (Int i = 1; i < N; ++i) qhi[i] = qhi[0];
    // We can use either 2-norm minimization or ClipAndAssuredSum for
    // the local filter. CAAS is the faster. It corresponds to limiter
    // = 0. 2-norm minimization is the same in spirit as limiter = 8,
    // but it assuredly achieves the first-order optimality conditions
    // whereas limiter 8 does not.
    if (limiter_option == 8)
      cedr::local::solve_1eq_bc_qp(np2, wa, wa, Qm, qlo, qhi, y, x);
    else {
      // We need to use *some* limiter; if 8 isn't chosen, default to
      // CAAS.
      cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x);
    }
  } else {
    const Int N = std::min(max_np2, np2);
    for (Int j = 0, cnt = 0; j < np; ++j)
      for (Int i = 0; i < np; ++i, ++cnt) {
        qlo[cnt] = q_min(i,j,k,q,ie);
        qhi[cnt] = q_max(i,j,k,q,ie);
      }
    for (Int trial = 0; trial < 3; ++trial) {
      int info;
      if (limiter_option == 8) {
        info = cedr::local::solve_1eq_bc_qp(
          np2, wa, wa, Qm, qlo, qhi, y, x);
        if (info == 1) info = 0;
      } else {
        info = 0;
        cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x, false /* clip */);
        // Clip for numerics against the cell extrema.
        Real qlo_s = qlo[0], qhi_s = qhi[0];
        for (Int i = 1; i < N; ++i) {
          qlo_s = std::min(qlo_s, qlo[i]);
          qhi_s = std::max(qhi_s, qhi[i]);
        }
        for (Int i = 0; i < N; ++i)
          x[i] = cedr::impl::max(qlo_s, cedr::impl::min(qhi_s, x[i]));
      }
      if (info == 0 || trial == 1) break;
      switch (trial) {
      case 0: {
        Real qlo_s = qlo[0], qhi_s = qhi[0];
        for (Int i = 1; i < N; ++i) {
          qlo_s = std::min(qlo_s, qlo[i]);
          qhi_s = std::max(qhi_s, qhi[i]);
        }
        const Int N = std::min(max_np2, np2);
        for (Int i = 0; i < N; ++i) qlo[i] = qlo_s;
        for (Int i = 0; i < N; ++i) qhi[i] = qhi_s;
      } break;
      case 1: {
        const Real q = Qm / rhom;
        for (Int i = 0; i < N; ++i) qlo[i] = std::min(qlo[i], q);
        for (Int i = 0; i < N; ++i) qhi[i] = std::max(qhi[i], q);                
      } break;
      }
    }
  }
        
  for (Int j = 0, cnt = 0; j < np; ++j)
    for (Int i = 0; i < np; ++i, ++cnt) {
      q_c(i,j,k,q) = x[cnt];
      qdp_c(i,j,k,q,n1_qdp) = q_c(i,j,k,q) * dp3d_c(i,j,k,tl_np1);
    }
}

Int vertical_caas_backup (const Int n, Real* rhom,
                          const Real q_min, const Real q_max,
                          Real Qmlo_tot, Real Qmhi_tot, const Real Qm_tot,
                          Real* Qmlo, Real* Qmhi, Real* Qm) {
  Int status = 0;
  if (Qm_tot < Qmlo_tot || Qm_tot > Qmhi_tot) {
    if (Qm_tot < Qmlo_tot) {
      status = -2;
      for (Int i = 0; i < n; ++i) Qmhi[i] = Qmlo[i];
      for (Int i = 0; i < n; ++i) Qmlo[i] = q_min*rhom[i];
      Qmlo_tot = 0;
      for (Int i = 0; i < n; ++i) Qmlo_tot += Qmlo[i];
      if (Qm_tot < Qmlo_tot) status = -4;
    } else {
      status = -1;
      for (Int i = 0; i < n; ++i) Qmlo[i] = Qmhi[i];
      for (Int i = 0; i < n; ++i) Qmhi[i] = q_max*rhom[i];
      Qmhi_tot = 0;
      for (Int i = 0; i < n; ++i) Qmhi_tot += Qmhi[i];
      if (Qm_tot > Qmhi_tot) status = -3;
    }
    if (status < -2) {
      Real rhom_tot = 0;
      for (Int i = 0; i < n; ++i) rhom_tot += rhom[i];
      const Real q = Qm_tot/rhom_tot;
      for (Int i = 0; i < n; ++i) Qm[i] = q*rhom[i];
      return status;
    }
  }
  for (Int i = 0; i < n; ++i) rhom[i] = 1;
  cedr::local::caas(n, rhom, Qm_tot, Qmlo, Qmhi, Qm, Qm, false);
  return status;
}

void run_local (CDR& cdr, const Data& d, Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete, const bool scalar_bounds,
                const Int limiter_option) {
  const Int np = d.np, nlev = d.nlev, qsize = d.qsize,
    nlevwrem = cdr.nsuplev*cdr.nsublev;

  FA5<      Real> q_min(q_min_r, np, np, nlev, qsize, nete+1);
  FA5<const Real> q_max(q_max_r, np, np, nlev, qsize, nete+1);

  for (Int ie = nets; ie <= nete; ++ie) {
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<      Real> qdp_c(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<      Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
#ifdef COMPOSE_COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const Int ti = cdr.cdr_over_super_levels ? q : spli*qsize + q;
        if (cdr.caas_in_suplev) {
          const auto ie_idx = cdr.cdr_over_super_levels ?
            cdr.nsuplev*ie + spli :
            ie;
          const auto lci = cdr.ie2lci[ie_idx];
          const Real Qm_tot = cdr.cdr->get_Qm(lci, ti);
          Real Qm_min_tot = 0, Qm_max_tot = 0;
          Real rhom[CDR::nsublev_per_suplev], Qm[CDR::nsublev_per_suplev],
            Qm_min[CDR::nsublev_per_suplev], Qm_max[CDR::nsublev_per_suplev];
          // Redistribute mass in the vertical direction of the super level.
          Int n = cdr.nsublev;
          for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
            const Int k = k0 + sbli;
            if (k >= nlev) {
              n = sbli;
              break;
            }
            rhom[sbli] = 0; Qm[sbli] = 0; Qm_min[sbli] = 0; Qm_max[sbli] = 0;
            Real Qm_prev = 0, volume = 0;
            accum_values(ie, k, q, d.tl_np1, d.n0_qdp, np,
                         false /* nonneg already applied */,
                         spheremp, dp3d_c, q_min, q_max, qdp_c, q_c,
                         volume, rhom[sbli], Qm[sbli], Qm_prev,
                         Qm_min[sbli], Qm_max[sbli]);
            Qm_min_tot += Qm_min[sbli];
            Qm_max_tot += Qm_max[sbli];
          }
          if (Qm_tot >= Qm_min_tot && Qm_tot <= Qm_max_tot) {
            for (Int i = 0; i < n; ++i) rhom[i] = 1;
            cedr::local::caas(n, rhom, Qm_tot, Qm_min, Qm_max, Qm, Qm, false);
          } else {
            Real q_min_s, q_max_s;
            bool first = true;
            for (Int sbli = 0; sbli < n; ++sbli) {
              const Int k = k0 + sbli;
              for (Int j = 0; j < np; ++j) {
                for (Int i = 0; i < np; ++i) {
                  if (first) {
                    q_min_s = q_min(i,j,k,q,ie);
                    q_max_s = q_max(i,j,k,q,ie);
                    first = false;
                  } else {
                    q_min_s = std::min(q_min_s, q_min(i,j,k,q,ie));
                    q_max_s = std::max(q_max_s, q_max(i,j,k,q,ie));
                  }
                }
              }
            }
            vertical_caas_backup(n, rhom, q_min_s, q_max_s,
                                 Qm_min_tot, Qm_max_tot, Qm_tot,
                                 Qm_min, Qm_max, Qm);
          }
          // Redistribute mass in the horizontal direction of each level.
          for (Int i = 0; i < n; ++i) {
            const Int k = k0 + i;
            solve_local(ie, k, q, d.tl_np1, d.n1_qdp, np,
                        scalar_bounds, limiter_option,
                        spheremp, dp3d_c, q_min, q_max, Qm[i], qdp_c, q_c);
          }
        } else {
          for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
            const Int k = k0 + sbli;
            if (k >= nlev) break;
            const auto ie_idx = cdr.cdr_over_super_levels ?
              nlevwrem*ie + k :
              cdr.nsublev*ie + sbli;
            const auto lci = cdr.ie2lci[ie_idx];
            const Real Qm = cdr.cdr->get_Qm(lci, ti);
            solve_local(ie, k, q, d.tl_np1, d.n1_qdp, np,
                        scalar_bounds, limiter_option,
                        spheremp, dp3d_c, q_min, q_max, Qm, qdp_c, q_c);
          }
        }
      }
    }
  }
}

void check (CDR& cdr, Data& d, const Real* q_min_r, const Real* q_max_r,
            const Int nets, const Int nete) {
  using cedr::mpi::reduce;

  const Int np = d.np, nlev = d.nlev, nsuplev = cdr.nsuplev, qsize = d.qsize,
    nprob = cdr.threed ? 1 : nsuplev;

  Kokkos::View<Real**, Kokkos::Serial>
    mass_p("mass_p", nprob, qsize), mass_c("mass_c", nprob, qsize),
    mass_lo("mass_lo", nprob, qsize), mass_hi("mass_hi", nprob, qsize),
    q_lo("q_lo", nprob, qsize), q_hi("q_hi", nprob, qsize),
    q_min_l("q_min_l", nprob, qsize), q_max_l("q_max_l", nprob, qsize),
    qd_lo("qd_lo", nprob, qsize), qd_hi("qd_hi", nprob, qsize);
  FA5<const Real>
    q_min(q_min_r, np, np, nlev, qsize, nete+1),
    q_max(q_max_r, np, np, nlev, qsize, nete+1);
  Kokkos::deep_copy(q_lo,  1e200);
  Kokkos::deep_copy(q_hi, -1e200);
  Kokkos::deep_copy(q_min_l,  1e200);
  Kokkos::deep_copy(q_max_l, -1e200);
  Kokkos::deep_copy(qd_lo, 0);
  Kokkos::deep_copy(qd_hi, 0);

  Int iprob = 0;

  bool fp_issue = false; // Limit output once the first issue is seen.
  for (Int ie = nets; ie <= nete; ++ie) {
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<const Real> qdp_pc(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<const Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
    for (Int spli = 0; spli < nsuplev; ++spli) {
      if (nprob > 1) iprob = spli;
      for (Int k = spli*cdr.nsublev; k < (spli+1)*cdr.nsublev; ++k) {
        if (k >= nlev) continue;
        if ( ! fp_issue) {
          for (Int j = 0; j < np; ++j)
            for (Int i = 0; i < np; ++i) {
              // FP issues.
              if (std::isnan(dp3d_c(i,j,k,d.tl_np1)))
              { pr("dp3d NaN:" pu(k) pu(i) pu(j)); fp_issue = true; }
              if (std::isinf(dp3d_c(i,j,k,d.tl_np1)))
              { pr("dp3d Inf:" pu(k) pu(i) pu(j)); fp_issue = true; }
            }
        }
        for (Int q = 0; q < qsize; ++q) {
          Real qlo_s = q_min(0,0,k,q,ie), qhi_s = q_max(0,0,k,q,ie);
          for (Int j = 0; j < np; ++j)
            for (Int i = 0; i < np; ++i) {
              qlo_s = std::min(qlo_s, q_min(i,j,k,q,ie));
              qhi_s = std::max(qhi_s, q_max(i,j,k,q,ie));
            }
          for (Int j = 0; j < np; ++j)
            for (Int i = 0; i < np; ++i) {
              // FP issues.
              if ( ! fp_issue) {
                for (Int i_qdp : {0, 1}) {
                  const Int n_qdp = i_qdp == 0 ? d.n0_qdp : d.n1_qdp;
                  if (std::isnan(qdp_pc(i,j,k,q,n_qdp)))
                  { pr("qdp NaN:" puf(i_qdp) pu(q) pu(k) pu(i) pu(j)); fp_issue = true; }
                  if (std::isinf(qdp_pc(i,j,k,q,n_qdp)))
                  { pr("qdp Inf:" puf(i_qdp) pu(q) pu(k) pu(i) pu(j)); fp_issue = true; }
                }
                if (std::isnan(q_c(i,j,k,q)))
                { pr("q NaN:" pu(q) pu(k) pu(i) pu(j)); fp_issue = true; }
                if (std::isinf(q_c(i,j,k,q)))
                { pr("q Inf:" pu(q) pu(k) pu(i) pu(j)); fp_issue = true; }
              }
              // Mass conservation.
              mass_p(iprob,q) += qdp_pc(i,j,k,q,d.n0_qdp) * spheremp(i,j);
              mass_c(iprob,q) += qdp_pc(i,j,k,q,d.n1_qdp) * spheremp(i,j);
              // Local bound constraints w.r.t. cell-local extrema.
              if (q_c(i,j,k,q) < qlo_s)
                qd_lo(iprob,q) = std::max(qd_lo(iprob,q), qlo_s - q_c(i,j,k,q));
              if (q_c(i,j,k,q) > qhi_s)
                qd_hi(iprob,q) = std::max(qd_hi(iprob,q), q_c(i,j,k,q) - qhi_s);
              // Safety problem bound constraints.
              mass_lo(iprob,q) += (q_min(i,j,k,q,ie) * dp3d_c(i,j,k,d.tl_np1) *
                                  spheremp(i,j));
              mass_hi(iprob,q) += (q_max(i,j,k,q,ie) * dp3d_c(i,j,k,d.tl_np1) *
                                  spheremp(i,j));
              q_lo(iprob,q) = std::min(q_lo(iprob,q), q_min(i,j,k,q,ie));
              q_hi(iprob,q) = std::max(q_hi(iprob,q), q_max(i,j,k,q,ie));
              q_min_l(iprob,q) = std::min(q_min_l(iprob,q), q_min(i,j,k,q,ie));
              q_max_l(iprob,q) = std::max(q_max_l(iprob,q), q_max(i,j,k,q,ie));
            }
        }
      }
    }
  }

#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    if ( ! d.check)
      d.check = std::make_shared<Data::Check>(nprob, qsize);
    auto& c = *d.check;
    Kokkos::deep_copy(c.mass_p, 0);
    Kokkos::deep_copy(c.mass_c, 0);
    Kokkos::deep_copy(c.mass_lo, 0);
    Kokkos::deep_copy(c.mass_hi, 0);
    Kokkos::deep_copy(c.q_lo,  1e200);
    Kokkos::deep_copy(c.q_hi, -1e200);
    Kokkos::deep_copy(c.q_min_l,  1e200);
    Kokkos::deep_copy(c.q_max_l, -1e200);
    Kokkos::deep_copy(c.qd_lo, 0);
    Kokkos::deep_copy(c.qd_hi, 0);
  }

#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp critical
#endif
  {
    auto& c = *d.check;
    for (Int spli = 0; spli < nprob; ++spli) {
      if (nprob > 1) iprob = spli;
      for (Int q = 0; q < qsize; ++q) {
        c.mass_p(iprob,q) += mass_p(iprob,q);
        c.mass_c(iprob,q) += mass_c(iprob,q);
        c.qd_lo(iprob,q) = std::max(c.qd_lo(iprob,q), qd_lo(iprob,q));
        c.qd_hi(iprob,q) = std::max(c.qd_hi(iprob,q), qd_hi(iprob,q));
        c.mass_lo(iprob,q) += mass_lo(iprob,q);
        c.mass_hi(iprob,q) += mass_hi(iprob,q);
        c.q_lo(iprob,q) = std::min(c.q_lo(iprob,q), q_lo(iprob,q));
        c.q_hi(iprob,q) = std::max(c.q_hi(iprob,q), q_hi(iprob,q));
        c.q_min_l(iprob,q) = std::min(c.q_min_l(iprob,q), q_min_l(iprob,q));
        c.q_max_l(iprob,q) = std::max(c.q_max_l(iprob,q), q_max_l(iprob,q));
      }
    }
  }

#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    Kokkos::View<Real**, Kokkos::Serial>
      mass_p_g("mass_p_g", nprob, qsize), mass_c_g("mass_c_g", nprob, qsize),
      mass_lo_g("mass_lo_g", nprob, qsize), mass_hi_g("mass_hi_g", nprob, qsize),
      q_lo_g("q_lo_g", nprob, qsize), q_hi_g("q_hi_g", nprob, qsize),
      q_min_g("q_min_g", nprob, qsize), q_max_g("q_max_g", nprob, qsize),
      qd_lo_g("qd_lo_g", nprob, qsize), qd_hi_g("qd_hi_g", nprob, qsize);

    const auto& p = *cdr.p;
    const auto& c = *d.check;
    const auto root = cdr.p->root();
    const auto N = nprob*qsize;

    reduce(p, c.mass_p.data(), mass_p_g.data(), N, MPI_SUM, root);
    reduce(p, c.mass_c.data(), mass_c_g.data(), N, MPI_SUM, root);
    reduce(p, c.qd_lo.data(), qd_lo_g.data(), N, MPI_MAX, root);
    reduce(p, c.qd_hi.data(), qd_hi_g.data(), N, MPI_MAX, root);
    // Safety problem.
    reduce(p, c.mass_lo.data(), mass_lo_g.data(), N, MPI_SUM, root);
    reduce(p, c.mass_hi.data(), mass_hi_g.data(), N, MPI_SUM, root);
    reduce(p, c.q_lo.data(), q_lo_g.data(), N, MPI_MIN, root);
    reduce(p, c.q_hi.data(), q_hi_g.data(), N, MPI_MAX, root);
    reduce(p, c.q_min_l.data(), q_min_g.data(), N, MPI_MIN, root);
    reduce(p, c.q_max_l.data(), q_max_g.data(), N, MPI_MAX, root);

    if (cdr.p->amroot()) {
      const Real tol = 1e4*std::numeric_limits<Real>::epsilon();
      for (Int k = 0; k < nprob; ++k)
        for (Int q = 0; q < qsize; ++q) {
          const Real rd = cedr::util::reldif(mass_p_g(k,q), mass_c_g(k,q));
          if (rd > tol)
            pr(puf(k) pu(q) pu(mass_p_g(k,q)) pu(mass_c_g(k,q)) pu(rd));
          if (mass_lo_g(k,q) <= mass_c_g(k,q) && mass_c_g(k,q) <= mass_hi_g(k,q)) {
            // Local problems should be feasible.
            if (qd_lo_g(k,q) > 0)
              pr(puf(k) pu(q) pu(qd_lo_g(k,q)));
            if (qd_hi_g(k,q) > 0)
              pr(puf(k) pu(q) pu(qd_hi_g(k,q)));
          } else {
            // Safety problem must hold.
            if (q_lo_g(k,q) < q_min_g(k,q))
              pr(puf(k) pu(q) pu(q_lo_g(k,q) - q_min_g(k,q)) pu(q_min_g(k,q)));
            if (q_hi_g(k,q) > q_max_g(k,q))
              pr(puf(k) pu(q) pu(q_max_g(k,q) - q_hi_g(k,q)) pu(q_max_g(k,q)));
          }
        }
    }
  }
}

} // namespace sl
} // namespace homme

static homme::CDR::Ptr g_cdr;

extern "C" void
cedr_init_impl (const homme::Int fcomm, const homme::Int cdr_alg, const bool use_sgi,
                const homme::Int* gid_data, const homme::Int* rank_data,
                const homme::Int gbl_ncell, const homme::Int lcl_ncell,
                const homme::Int nlev, const bool independent_time_steps, const bool hard_zero,
                const homme::Int, const homme::Int) {
  const auto p = cedr::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  g_cdr = std::make_shared<homme::CDR>(
    cdr_alg, gbl_ncell, lcl_ncell, nlev, use_sgi, independent_time_steps, hard_zero,
    gid_data, rank_data, p, fcomm);
}

extern "C" void cedr_query_bufsz (homme::Int* sendsz, homme::Int* recvsz) {
  cedr_assert(g_cdr);
  size_t s1, s2;
  g_cdr->get_buffers_sizes(s1, s2);
  *sendsz = static_cast<homme::Int>(s1);
  *recvsz = static_cast<homme::Int>(s2);
}

extern "C" void cedr_set_bufs (homme::Real* sendbuf, homme::Real* recvbuf,
                               homme::Int, homme::Int) {
  g_cdr->set_buffers(sendbuf, recvbuf);
}

extern "C" void cedr_unittest (const homme::Int fcomm, homme::Int* nerrp) {
#if 0
  cedr_assert(g_cdr);
  cedr_assert(g_cdr->tree);
  auto p = cedr::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  if (homme::CDR::Alg::is_qlt(g_cdr->alg))
    *nerrp = cedr::qlt::test::test_qlt(p, g_cdr->tree, g_cdr->nsublev*g_cdr->ncell,
                                       1, false, false, true, false);
  else
    *nerrp = cedr::caas::test::unittest(p);
#endif
  *nerrp += homme::test_tree_maker();
  *nerrp += homme::CDR::QLTT::unittest();
}

extern "C" void cedr_set_ie2gci (const homme::Int ie, const homme::Int gci) {
  cedr_assert(g_cdr);
  // Now is a good time to drop the tree, whose persistence was used for unit
  // testing if at all.
  g_cdr->tree = nullptr;
  homme::set_ie2gci(*g_cdr, ie - 1, gci - 1);
}

static homme::sl::Data::Ptr g_sl;

extern "C" homme::Int cedr_sl_init (
  const homme::Int np, const homme::Int nlev, const homme::Int qsize,
  const homme::Int qsized, const homme::Int timelevels,
  const homme::Int need_conservation)
{
  cedr_assert(g_cdr);
  g_sl = std::make_shared<homme::sl::Data>(g_cdr->nlclcell, np, nlev, qsize,
                                           qsized, timelevels);
  homme::init_ie2lci(*g_cdr);
  homme::init_tracers(*g_cdr, nlev, qsize, need_conservation);
  homme::sl::check(*g_cdr, *g_sl);
  return 1;
}

extern "C" void cedr_sl_set_pointers_begin (homme::Int nets, homme::Int nete) {}
extern "C" void cedr_sl_set_spheremp (homme::Int ie, homme::Real* v)
{ homme::sl::insert(g_sl, ie - 1, 0, v); }
extern "C" void cedr_sl_set_qdp (homme::Int ie, homme::Real* v, homme::Int n0_qdp,
                                 homme::Int n1_qdp)
{ homme::sl::insert(g_sl, ie - 1, 1, v, n0_qdp - 1, n1_qdp - 1); }
extern "C" void cedr_sl_set_dp3d (homme::Int ie, homme::Real* v, homme::Int tl_np1)
{ homme::sl::insert(g_sl, ie - 1, 2, v, tl_np1 - 1); }
extern "C" void cedr_sl_set_dp (homme::Int ie, homme::Real* v)
{ homme::sl::insert(g_sl, ie - 1, 2, v, 0); }
extern "C" void cedr_sl_set_q (homme::Int ie, homme::Real* v)
{ homme::sl::insert(g_sl, ie - 1, 3, v); }
extern "C" void cedr_sl_set_dp0 (homme::Real* v)
{ homme::sl::insert(g_sl, 0, 4, v); }
extern "C" void cedr_sl_set_pointers_end () {}

// Run QLT.
extern "C" void cedr_sl_run (homme::Real* minq, const homme::Real* maxq,
                             homme::Int nets, homme::Int nete) {
  cedr_assert(minq != maxq);
  cedr_assert(g_cdr);
  cedr_assert(g_sl);
  homme::sl::run(*g_cdr, *g_sl, minq, maxq, nets-1, nete-1);
}

// Run the cell-local limiter problem.
extern "C" void cedr_sl_run_local (homme::Real* minq, const homme::Real* maxq,
                                   homme::Int nets, homme::Int nete, homme::Int use_ir,
                                   homme::Int limiter_option) {
  cedr_assert(minq != maxq);
  cedr_assert(g_cdr);
  cedr_assert(g_sl);
  homme::sl::run_local(*g_cdr, *g_sl, minq, maxq, nets-1, nete-1, use_ir,
                       limiter_option);
}

// Check properties for this transport step.
extern "C" void cedr_sl_check (const homme::Real* minq, const homme::Real* maxq,
                               homme::Int nets, homme::Int nete) {
  cedr_assert(g_cdr);
  cedr_assert(g_sl);
  homme::sl::check(*g_cdr, *g_sl, minq, maxq, nets-1, nete-1);
}

extern "C" void cedr_finalize () {
  g_sl = nullptr;
  g_cdr = nullptr;
}
