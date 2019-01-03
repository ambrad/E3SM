// Includes if using compose library.
#include "compose/cedr.hpp"
#include "compose/cedr_util.hpp"
// Use these when rewriting each CDR's run() function to interact nicely with
// Homme's nested OpenMP and top-level horizontal threading scheme.
#include "compose/cedr_qlt.hpp"
#include "compose/cedr_caas.hpp"

#define THREAD_QLT_RUN
#ifndef QLT_MAIN
# ifdef HAVE_CONFIG_H
#  include "config.h.c"
# endif
#endif

namespace homme {
namespace compose {

template <typename ES>
struct QLT : public cedr::qlt::QLT<ES> {
  QLT (const cedr::mpi::Parallel::Ptr& p, const cedr::Int& ncells,
       const cedr::qlt::tree::Node::Ptr& tree)
    : cedr::qlt::QLT<ES>(p, ncells, tree)
  {}

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
#if ! defined THREAD_QLT_RUN && defined HORIZ_OPENMP
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
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
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
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
#         pragma omp barrier
#endif
        }

        // Combine kids' data.
        if (lvl.nodes.size()) {
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
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
#if defined THREAD_QLT_RUN && defined COLUMN_OPENMP
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
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
#       pragma omp master
#endif
        {
          for (size_t i = 0; i < lvl.me.size(); ++i) {
            const auto& mmd = lvl.me[i];
            mpi::isend(*p_, &bd_.l2r_data(mmd.offset*l2rndps), mmd.size*l2rndps,
                       mmd.rank, mpitag);
          }
        }

#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
#       pragma omp barrier
#endif
      }

      // Root.
      if ( ! (ns_->levels.empty() || ns_->levels.back().nodes.size() != 1 ||
              ns_->node_h(ns_->levels.back().nodes[0])->parent >= 0)) {
        const auto n = ns_->node_h(ns_->levels.back().nodes[0]);
        for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
          const Int problem_type = md_.get_problem_type(pti);
          const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP && defined COLUMN_OPENMP
#         pragma omp parallel
#endif
#if defined THREAD_QLT_RUN && (defined HORIZ_OPENMP || defined COLUMN_OPENMP)
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
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
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
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
#         pragma omp barrier
#endif
        }

        // Solve QP for kids' values.
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
#       pragma omp for
#endif
        for (size_t ni = 0; ni < lvl.nodes.size(); ++ni) {
          const auto lvlidx = lvl.nodes[ni];
          const auto n = ns_->node_h(lvlidx);
          if ( ! n->nkids) continue;
          for (Int pti = 0; pti < md_.nprobtypes; ++pti) {
            const Int problem_type = md_.get_problem_type(pti);
            const Int bis = md_.a_d.prob2trcrptr[pti], bie = md_.a_d.prob2trcrptr[pti+1];
#if defined THREAD_QLT_RUN && defined COLUMN_OPENMP
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
                 bd_.r2l_data(k1->offset*r2lndps + r2lbdi));
            }
          }
        }

        // Send.
        if (lvl.kids.size())
#if defined THREAD_QLT_RUN && defined HORIZ_OPENMP
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
#if ! defined THREAD_QLT_RUN && defined HORIZ_OPENMP
    }
#endif
  }
};

// We explicitly use Kokkos::Serial here so we can run the Kokkos kernels in the
// super class w/o triggering an expecution-space initialization error in
// Kokkos. This complication results from the interaction of Homme's
// HORIZ_OPENMP threading with Kokkos kernels.
struct CAAS : public cedr::caas::CAAS<Kokkos::Serial> {
  typedef cedr::caas::CAAS<Kokkos::Serial> Super;

  CAAS (const cedr::mpi::Parallel::Ptr& p, const cedr::Int nlclcells,
        const typename Super::UserAllReducer::Ptr& uar)
    : Super(p, nlclcells, uar)
  {}

  void run () override {
#if defined HORIZ_OPENMP
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
  if (node->nkids)
    for (Int k = 0; k < node->nkids; ++k)
      renumber(nrank, nelem, my_rank, owned_ids, rank2sfc, node->kids[k]);
  else {
    const Int sfc = node->cellidx;
    node->rank = rank2sfc_search(rank2sfc, nrank, sfc);
    cedr_assert(node->rank >= 0 && node->rank < nrank);
    node->cellidx = node->rank == my_rank ? owned_ids[sfc - rank2sfc[my_rank]] : -1;
    cedr_assert((node->rank != my_rank && node->cellidx == -1) ||
                (node->rank == my_rank && node->cellidx >= 0 && node->cellidx < nelem));
  }
}

// Build a subtree over [0, nsublev).
void add_sub_levels (const qlt::tree::Node::Ptr& node, const Int nsublev,
                     const Int gci, const Int rank,
                     const Int slb, const Int sle) {
  if (slb+1 == sle) {
    node->cellidx = nsublev*gci + slb;
  } else {
    node->nkids = 2;
    for (Int k = 0; k < 2; ++k) {
      auto kid = std::make_shared<qlt::tree::Node>();
      kid->parent = node.get();
      kid->rank = rank;
      node->kids[k] = kid;
    }
    const Int mid = slb + (sle - slb)/2;
    add_sub_levels(node->kids[0], nsublev, gci, rank, slb, mid);
    add_sub_levels(node->kids[1], nsublev, gci, rank, mid, sle);
  }
}

// Recurse to each leaf and call add_sub_levels above.
void add_sub_levels (const Int my_rank, const qlt::tree::Node::Ptr& node,
                     const Int nsublev) {
  if (node->nkids)
    for (Int k = 0; k < node->nkids; ++k)
      add_sub_levels(my_rank, node->kids[k], nsublev);
  else {
    const Int gci = node->cellidx;
    const Int rank = node->rank;
    add_sub_levels(node, nsublev, gci, rank, 0, nsublev);
  }
}

qlt::tree::Node::Ptr
make_tree (const cedr::mpi::Parallel::Ptr& p, const Int nelem,
           const Int* owned_ids, const Int* rank2sfc, const Int nsublev) {
  // Partition 0:nelem-1, the space-filling curve space.
  auto tree = qlt::tree::make_tree_over_1d_mesh(p, nelem);
  // Renumber so that node->cellidx records the global element number, and
  // associate the correct rank with the element.
  const auto my_rank = p->rank();
  renumber(p->size(), nelem, my_rank, owned_ids, rank2sfc, tree);
  add_sub_levels(my_rank, tree, nsublev);
  return tree;
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
    enum Enum { qlt, qlt_super_level, caas, caas_super_level };
    static Enum convert (Int cdr_alg) {
      switch (cdr_alg) {
      case 2:  return qlt;
      case 20: return qlt_super_level;
      case 3:  return caas;
      case 30: return caas_super_level;
      case 42: return caas_super_level; // actually none
      default: cedr_throw_if(true,  "cdr_alg " << cdr_alg << " is invalid.");
      }
    }
    static bool is_qlt (Enum e) { return e == qlt || e == qlt_super_level; }
    static bool is_caas (Enum e) { return e == caas || e == caas_super_level; }
    static bool is_suplev (Enum e) {
      return e == qlt_super_level || e == caas_super_level;
    }
  };

  enum { nsublev_per_suplev = 8 };
  
  const Alg::Enum alg;
  const Int ncell, nlclcell, nlev, nsublev, nsuplev;
  const cedr::mpi::Parallel::Ptr p;
  qlt::tree::Node::Ptr tree; // Don't need this except for unit testing.
  cedr::CDR::Ptr cdr;
  std::vector<Int> ie2gci; // Map Homme ie to Homme global cell index.
  std::vector<Int> ie2lci; // Map Homme ie to CDR local cell index (lclcellidx).

  CDR (Int cdr_alg_, Int ngblcell_, Int nlclcell_, Int nlev_,
       const Int* owned_ids, const Int* rank2sfc,
       const cedr::mpi::Parallel::Ptr& p_, Int fcomm)
    : alg(Alg::convert(cdr_alg_)), ncell(ngblcell_), nlclcell(nlclcell_),
      nlev(nlev_),
      nsublev(Alg::is_suplev(alg) ? nsublev_per_suplev : 1),
      nsuplev((nlev + nsublev - 1) / nsublev),
      p(p_), inited_tracers_(false)
  {
    if (Alg::is_qlt(alg)) {
      tree = make_tree(p, ncell, owned_ids, rank2sfc, nsublev);
      cdr = std::make_shared<QLTT>(p, ncell*nsublev, tree);
    } else if (Alg::is_caas(alg)) {
      const auto caas = std::make_shared<CAAST>(
        p, nlclcell*nsublev, std::make_shared<ReproSumReducer>(fcomm));
      cdr = caas;
    } else {
      cedr_throw_if(true, "Invalid semi_lagrange_cdr_alg " << alg);
    }
    ie2gci.resize(nlclcell);
  }

  void init_tracers (const Int qsize, const bool need_conservation) {
    typedef cedr::ProblemType PT;
    for (Int ti = 0, nt = nsuplev*qsize; ti < nt; ++ti)
      cdr->declare_tracer(PT::shapepreserve |
                          (need_conservation ? PT::conserve : 0), 0);
    cdr->end_tracer_declarations();
  }

private:
  bool inited_tracers_;
};

void set_ie2gci (CDR& q, const Int ie, const Int gci) { q.ie2gci[ie] = gci; }

void init_ie2lci (CDR& q) {
  q.ie2lci.resize(q.nsublev*q.ie2gci.size());
  if (CDR::Alg::is_qlt(q.alg)) {
    auto qlt = std::static_pointer_cast<CDR::QLTT>(q.cdr);
    for (size_t ie = 0; ie < q.ie2gci.size(); ++ie) {
      for (Int sbli = 0; sbli < q.nsublev; ++sbli)
        q.ie2lci[q.nsublev*ie + sbli] = qlt->gci2lci(q.nsublev*q.ie2gci[ie] + sbli);
    }
  } else {
    for (size_t ie = 0; ie < q.ie2gci.size(); ++ie)
      for (Int sbli = 0; sbli < q.nsublev; ++sbli) {
        const Int id = q.nsublev*ie + sbli;
        q.ie2lci[id] = id;
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
  default: cedr_throw_if(true, "Invalid pointer index " << ptridx);
  }
}

static void run_cdr (CDR& q) {
#ifdef HORIZ_OPENMP
# pragma omp barrier
#endif
  q.cdr->run();
#ifdef HORIZ_OPENMP
# pragma omp barrier
#endif
}

void run (CDR& cdr, const Data& d, const Real* q_min_r, const Real* q_max_r,
          const Int nets, const Int nete) {
  static constexpr Int max_np = 4;
  const Int np = d.np, nlev = d.nlev, qsize = d.qsize, ncell = nete - nets + 1;
  cedr_assert(np <= max_np);

  FA5<const Real>
    q_min(q_min_r, np, np, nlev, qsize, ncell),
    q_max(q_max_r, np, np, nlev, qsize, ncell);

  for (Int ie = nets; ie <= nete; ++ie) {
    const Int ie0 = ie - nets;
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<const Real> qdp_p(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<const Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
#ifdef COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const Int ti = spli*qsize + q;
        for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
          const Int lci = cdr.ie2lci[cdr.nsublev*ie + sbli];
          const Int k = k0 + sbli;
          if (k >= nlev) {
            cdr.cdr->set_Qm(lci, ti, 0, 0, 0, 0);
            break;
          }
          Real Qm = 0, Qm_min = 0, Qm_max = 0, Qm_prev = 0, rhom = 0, volume = 0;
          for (Int j = 0; j < np; ++j) {
            for (Int i = 0; i < np; ++i) {
              volume += spheremp(i,j);
              const Real rhomij = dp3d_c(i,j,k,d.tl_np1) * spheremp(i,j);
              rhom += rhomij;
              Qm += q_c(i,j,k,q) * rhomij;
              Qm_min += q_min(i,j,k,q,ie0) * rhomij;
              Qm_max += q_max(i,j,k,q,ie0) * rhomij;
              Qm_prev += qdp_p(i,j,k,q,d.n0_qdp) * spheremp(i,j);
            }
          }
          //kludge For now, handle just one rhom. For feasible global problems,
          // it's used only as a weight vector in QLT, so it's fine. In fact,
          // use just the cell geometry, rather than total density, since in QLT
          // this field is used as a weight vector.
          //todo Generalize to one rhom field per level. Until then, we're not
          // getting QLT's safety benefit.
          if (ti == 0) cdr.cdr->set_rhom(lci, 0, volume);
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
          cdr.cdr->set_Qm(lci, ti, Qm, Qm_min, Qm_max, Qm_prev);
        }
      }
    }
  }

  run_cdr(cdr);
}

void run_local (CDR& cdr, const Data& d, const Real* q_min_r, const Real* q_max_r,
                const Int nets, const Int nete, const bool scalar_bounds,
                const Int limiter_option) {
  static constexpr Int max_np = 4, max_np2 = max_np*max_np;
  const Int np = d.np, np2 = np*np, nlev = d.nlev, qsize = d.qsize,
    ncell = nete - nets + 1;
  cedr_assert(np <= max_np);

  FA5<const Real>
    q_min(q_min_r, np, np, nlev, qsize, ncell),
    q_max(q_max_r, np, np, nlev, qsize, ncell);

  for (Int ie = nets; ie <= nete; ++ie) {
    const Int ie0 = ie - nets;
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<      Real> qdp_c(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<      Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
#ifdef COLUMN_OPENMP
#   pragma omp parallel for
#endif
    for (Int spli = 0; spli < cdr.nsuplev; ++spli) {
      const Int k0 = cdr.nsublev*spli;
      for (Int q = 0; q < qsize; ++q) {
        const Int ti = spli*qsize + q;
        for (Int sbli = 0; sbli < cdr.nsublev; ++sbli) {
          const Int k = k0 + sbli;
          const Int lci = cdr.ie2lci[cdr.nsublev*ie + sbli];
          if (k >= nlev) break;
          Real wa[max_np2], qlo[max_np2], qhi[max_np2], y[max_np2], x[max_np2];
          Real rhom = 0;
          for (Int j = 0, cnt = 0; j < np; ++j)
            for (Int i = 0; i < np; ++i, ++cnt) {
              const Real rhomij = dp3d_c(i,j,k,d.tl_np1) * spheremp(i,j);
              rhom += rhomij;
              wa[cnt] = rhomij;
              y[cnt] = q_c(i,j,k,q);
              x[cnt] = y[cnt];
            }
          const Real Qm = cdr.cdr->get_Qm(lci, ti);

          //todo Replace with ReconstructSafely.
          if (scalar_bounds) {
            qlo[0] = q_min(0,0,k,q,ie0);
            qhi[0] = q_max(0,0,k,q,ie0);
            const Int N = std::min(max_np2, np2);
            for (Int i = 1; i < N; ++i) qlo[i] = qlo[0];
            for (Int i = 1; i < N; ++i) qhi[i] = qhi[0];
            // We can use either 2-norm minimization or ClipAndAssuredSum for the
            // local filter. CAAS is the faster. It corresponds to limiter =
            // 0. 2-norm minimization is the same in spirit as limiter = 8, but it
            // assuredly achieves the first-order optimality conditions whereas
            // limiter 8 does not.
            if (limiter_option == 8)
              cedr::local::solve_1eq_bc_qp(np2, wa, wa, Qm, qlo, qhi, y, x);
            else {
              // We need to use *some* limiter; if 8 isn't chosen, default to CAAS.
              cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x);
            }
          } else {
            for (Int j = 0, cnt = 0; j < np; ++j)
              for (Int i = 0; i < np; ++i, ++cnt) {
                qlo[cnt] = q_min(i,j,k,q,ie0);
                qhi[cnt] = q_max(i,j,k,q,ie0);
              }
            for (int trial = 0; trial < 2; ++trial) {
              int info;
              if (limiter_option == 8) {
                info = cedr::local::solve_1eq_bc_qp(
                  np2, wa, wa, Qm, qlo, qhi, y, x);
                if (info == 1) info = 0;
              } else {
                info = 0;
                cedr::local::caas(np2, wa, Qm, qlo, qhi, y, x);
              }
              if (info == 0 || trial == 1) break;
              const Real q = Qm / rhom;
              const Int N = std::min(max_np2, np2);
              for (Int i = 0; i < N; ++i) qlo[i] = std::min(qlo[i], q);
              for (Int i = 0; i < N; ++i) qhi[i] = std::max(qhi[i], q);
            }          
          }
        
          for (Int j = 0, cnt = 0; j < np; ++j)
            for (Int i = 0; i < np; ++i, ++cnt) {
              q_c(i,j,k,q) = x[cnt];
              qdp_c(i,j,k,q,d.n1_qdp) = q_c(i,j,k,q) * dp3d_c(i,j,k,d.tl_np1);
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
    ncell = nete - nets + 1;

  Kokkos::View<Real**, Kokkos::Serial>
    mass_p("mass_p", nsuplev, qsize), mass_c("mass_c", nsuplev, qsize),
    mass_lo("mass_lo", nsuplev, qsize), mass_hi("mass_hi", nsuplev, qsize),
    q_lo("q_lo", nsuplev, qsize), q_hi("q_hi", nsuplev, qsize),
    q_min_l("q_min_l", nsuplev, qsize), q_max_l("q_max_l", nsuplev, qsize),
    qd_lo("qd_lo", nsuplev, qsize), qd_hi("qd_hi", nsuplev, qsize);
  FA5<const Real>
    q_min(q_min_r, np, np, nlev, qsize, ncell),
    q_max(q_max_r, np, np, nlev, qsize, ncell);
  Kokkos::deep_copy(q_lo,  1e200);
  Kokkos::deep_copy(q_hi, -1e200);
  Kokkos::deep_copy(q_min_l,  1e200);
  Kokkos::deep_copy(q_max_l, -1e200);
  Kokkos::deep_copy(qd_lo, 0);
  Kokkos::deep_copy(qd_hi, 0);

  bool fp_issue = false; // Limit output once the first issue is seen.
  for (Int ie = nets; ie <= nete; ++ie) {
    const Int ie0 = ie - nets;
    FA2<const Real> spheremp(d.spheremp[ie], np, np);
    FA5<const Real> qdp_pc(d.qdp_pc[ie], np, np, nlev, d.qsize_d, 2);
    FA4<const Real> dp3d_c(d.dp3d_c[ie], np, np, nlev, d.timelevels);
    FA4<const Real> q_c(d.q_c[ie], np, np, nlev, d.qsize_d);
    for (Int spli = 0; spli < nsuplev; ++spli) {
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
        for (Int q = 0; q < qsize; ++q)
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
              mass_p(spli,q) += qdp_pc(i,j,k,q,d.n0_qdp) * spheremp(i,j);
              mass_c(spli,q) += qdp_pc(i,j,k,q,d.n1_qdp) * spheremp(i,j);
              // Local bound constraints.
              if (q_c(i,j,k,q) < q_min(i,j,k,q,ie0))
                qd_lo(spli,q) = std::max(qd_lo(spli,q),
                                         q_min(i,j,k,q,ie0) - q_c(i,j,k,q));
              if (q_c(i,j,k,q) > q_max(i,j,k,q,ie0))
                qd_hi(spli,q) = std::max(qd_hi(spli,q),
                                         q_c(i,j,k,q) - q_max(i,j,k,q,ie0));
              // Safety problem bound constraints.
              mass_lo(spli,q) += (q_min(i,j,k,q,ie0) * dp3d_c(i,j,k,d.tl_np1) *
                                  spheremp(i,j));
              mass_hi(spli,q) += (q_max(i,j,k,q,ie0) * dp3d_c(i,j,k,d.tl_np1) *
                                  spheremp(i,j));
              q_lo(spli,q) = std::min(q_lo(spli,q), q_min(i,j,k,q,ie0));
              q_hi(spli,q) = std::max(q_hi(spli,q), q_max(i,j,k,q,ie0));
              q_min_l(spli,q) = std::min(q_min_l(spli,q), q_min(i,j,k,q,ie0));
              q_max_l(spli,q) = std::max(q_max_l(spli,q), q_max(i,j,k,q,ie0));
            }
      }
    }
  }

#ifdef HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    if ( ! d.check)
      d.check = std::make_shared<Data::Check>(nsuplev, qsize);
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

#ifdef HORIZ_OPENMP
# pragma omp barrier
# pragma omp critical
#endif
  {
    auto& c = *d.check;
    for (Int spli = 0; spli < nsuplev; ++spli)
      for (Int q = 0; q < qsize; ++q) {
        c.mass_p(spli,q) += mass_p(spli,q);
        c.mass_c(spli,q) += mass_c(spli,q);
        c.qd_lo(spli,q) = std::max(c.qd_lo(spli,q), qd_lo(spli,q));
        c.qd_hi(spli,q) = std::max(c.qd_hi(spli,q), qd_hi(spli,q));
        c.mass_lo(spli,q) += mass_lo(spli,q);
        c.mass_hi(spli,q) += mass_hi(spli,q);
        c.q_lo(spli,q) = std::min(c.q_lo(spli,q), q_lo(spli,q));
        c.q_hi(spli,q) = std::max(c.q_hi(spli,q), q_hi(spli,q));
        c.q_min_l(spli,q) = std::min(c.q_min_l(spli,q), q_min_l(spli,q));
        c.q_max_l(spli,q) = std::max(c.q_max_l(spli,q), q_max_l(spli,q));
      }
  }

#ifdef HORIZ_OPENMP
# pragma omp barrier
# pragma omp master
#endif
  {
    Kokkos::View<Real**, Kokkos::Serial>
      mass_p_g("mass_p_g", nsuplev, qsize), mass_c_g("mass_c_g", nsuplev, qsize),
      mass_lo_g("mass_lo_g", nsuplev, qsize), mass_hi_g("mass_hi_g", nsuplev, qsize),
      q_lo_g("q_lo_g", nsuplev, qsize), q_hi_g("q_hi_g", nsuplev, qsize),
      q_min_g("q_min_g", nsuplev, qsize), q_max_g("q_max_g", nsuplev, qsize),
      qd_lo_g("qd_lo_g", nsuplev, qsize), qd_hi_g("qd_hi_g", nsuplev, qsize);

    const auto& p = *cdr.p;
    const auto& c = *d.check;
    const auto root = cdr.p->root();
    const auto N = nsuplev*qsize;

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
      for (Int k = 0; k < nsuplev; ++k)
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

extern "C" {
void kokkos_init () {
  Kokkos::InitArguments args;
  args.disable_warnings = true;
  Kokkos::initialize(args);
}

void kokkos_finalize () { Kokkos::finalize_all(); }

extern "C" void cedr_unittest (const homme::Int fcomm, homme::Int* nerrp) {
}

extern "C" void cedr_set_ie2gci (const homme::Int ie, const homme::Int gci) {
}

extern "C" homme::Int cedr_forcing_init (
  const homme::Int np, const homme::Int nlev, const homme::Int qsize,
  const homme::Int qsized, const homme::Int timelevels,
  const homme::Int need_conservation)
{
  return 1;
}

extern "C" void cedr_forcing_set_qdp (homme::Int ie, homme::Real* v, homme::Int n0_qdp,
                                      homme::Int n1_qdp)
{}

extern "C" void cedr_forcing_run (const homme::Real* minq, const homme::Real* maxq,
                                  homme::Int nets, homme::Int nete) {
  cedr_assert(minq != maxq);
}

extern "C" void cedr_forcing_run_local (const homme::Real* minq, const homme::Real* maxq,
                                        homme::Int nets, homme::Int nete, homme::Int use_ir,
                                        homme::Int limiter_option) {
  cedr_assert(minq != maxq);
}

extern "C" void cedr_forcing_check (const homme::Real* minq, const homme::Real* maxq,
                                    homme::Int nets, homme::Int nete) {
}
} // extern "C"
