#include "compose_cedr_cdr.hpp"
#include "compose_cedr_sl.hpp"
#include "compose_kokkos.hpp"

namespace ko = Kokkos;

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

template <typename MT>
CDR<MT>::CDR (Int cdr_alg_, Int ngblcell_, Int nlclcell_, Int nlev_, bool use_sgi,
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

template <typename MT>
void CDR<MT>::init_tracers (const Int qsize, const bool need_conservation) {
  nonneg.resize(qsize, hard_zero);
  typedef cedr::ProblemType PT;
  const Int nt = cdr_over_super_levels ? qsize : nsuplev*qsize;
  for (Int ti = 0; ti < nt; ++ti)
    cdr->declare_tracer(PT::shapepreserve |
                        (need_conservation ? PT::conserve : 0), 0);
  cdr->end_tracer_declarations();
}

template <typename MT>
void CDR<MT>::get_buffers_sizes (size_t& s1, size_t &s2) {
  cdr->get_buffers_sizes(s1, s2);
}

template <typename MT>
void CDR<MT>::set_buffers (Real* b1, Real* b2) {
  cdr->set_buffers(b1, b2);
  cdr->finish_setup();
}

template <typename MT>
void set_ie2gci (CDR<MT>& q, const Int ie, const Int gci) { q.ie2gci[ie] = gci; }

template <typename MT>
void init_ie2lci (CDR<MT>& q) {
  const Int n_id_in_suplev = q.caas_in_suplev ? 1 : q.nsublev;
  const Int nleaf =
    n_id_in_suplev*
    q.ie2gci.size()*
    (q.cdr_over_super_levels ? q.nsuplev : 1);
  q.ie2lci.resize(nleaf);
  if (Alg::is_qlt(q.alg)) {
    auto qlt = std::static_pointer_cast<typename CDR<MT>::QLTT>(q.cdr);
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

template <typename MT>
void init_tracers (CDR<MT>& q, const Int nlev, const Int qsize,
                   const bool need_conservation) {
  q.init_tracers(qsize, need_conservation);
}

namespace sl { // For sl_advection.F90

template <typename T>
void insert (std::vector<T*>& r, const Int i, T* v) {
  cedr_assert(i >= 0 && i < static_cast<int>(r.size()));
  r[i] = v;
}

void insert (const Data::Ptr& d, const Int ie, const Int ptridx, Real* array,
             const Int i0 = 0, const Int i1 = 0) {
  cedr_assert(d);
  switch (ptridx) {
  case 0: d->ta->pspheremp.set_ie_ptr(ie, array); break;
  case 1: d->ta->pqdp.set_ie_ptr(ie, array); d->ta->n0_qdp = i0; d->ta->n1_qdp = i1; break;
  case 2: d->ta->pdp3d.set_ie_ptr(ie, array); d->ta->np1 = i0; break;
  case 3: d->ta->pq.set_ie_ptr(ie, array); break;
  case 4: /* unused */; break;
  default: cedr_throw_if(true, "Invalid pointer index " << ptridx);
  }
}

} // namespace sl
} // namespace homme

static homme::CDR<ko::MachineTraits>::Ptr g_cdr;

extern "C" void
cedr_init_impl (const homme::Int fcomm, const homme::Int cdr_alg, const bool use_sgi,
                const homme::Int* gid_data, const homme::Int* rank_data,
                const homme::Int gbl_ncell, const homme::Int lcl_ncell,
                const homme::Int nlev, const bool independent_time_steps, const bool hard_zero,
                const homme::Int, const homme::Int) {
  const auto p = cedr::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  g_cdr = std::make_shared<homme::CDR<ko::MachineTraits> >(
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
  *nerrp += homme::CDR<ko::MachineTraits>::QLTT::unittest();
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
  const auto tracer_arrays = homme::get_tracer_arrays();
  g_sl = std::make_shared<homme::sl::Data>(tracer_arrays);
  homme::init_ie2lci(*g_cdr);
  homme::init_tracers(*g_cdr, nlev, qsize, need_conservation);
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
extern "C" void cedr_sl_run_global (homme::Real* minq, const homme::Real* maxq,
                                    homme::Int nets, homme::Int nete) {
  cedr_assert(minq != maxq);
  cedr_assert(g_cdr);
  cedr_assert(g_sl);
  homme::cedr_h2d(*g_sl->ta);
  homme::sl::run_global<ko::MachineTraits>(*g_cdr, *g_sl, minq, maxq, nets-1, nete-1);
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
  homme::cedr_d2h(*g_sl->ta);
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

namespace homme {
template class CDR<Kokkos::MachineTraits>;
} // namespace homme
