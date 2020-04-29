#ifndef INCLUDE_COMPOSE_SLMM_ISLMPI_HPP
#define INCLUDE_COMPOSE_SLMM_ISLMPI_HPP

#include "compose.hpp"
#include "compose_homme.hpp"
#include "compose_slmm.hpp"
#include "compose_slmm_advecter.hpp"

#include <mpi.h>

#include <memory>

namespace homme {
namespace mpi { //todo Share with cedr.

class Parallel {
  MPI_Comm comm_;
public:
  typedef std::shared_ptr<Parallel> Ptr;
  Parallel(MPI_Comm comm) : comm_(comm) {}
  MPI_Comm comm () const { return comm_; }
  Int size() const {
    int sz = 0;
    MPI_Comm_size(comm_, &sz);
    return sz;
  }
  Int rank() const {
    int pid = 0;
    MPI_Comm_rank(comm_, &pid);
    return pid;
  }
  Int root () const { return 0; }
  bool amroot () const { return rank() == root(); }
};

inline Parallel::Ptr make_parallel (MPI_Comm comm) {
  return std::make_shared<Parallel>(comm);
}

struct Request {
  MPI_Request request;

#ifdef COMPOSE_DEBUG_MPI
  int unfreed;
  Request();
  ~Request();
#endif
};

template <typename T> MPI_Datatype get_type();
template <> inline MPI_Datatype get_type<int>() { return MPI_INT; }
template <> inline MPI_Datatype get_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype get_type<long>() { return MPI_LONG_INT; }

template <typename T>
int isend (const Parallel& p, const T* buf, int count, int dest, int tag,
           Request* ireq) {
  MPI_Datatype dt = get_type<T>();
  MPI_Request ureq;
  MPI_Request* req = ireq ? &ireq->request : &ureq;
  int ret = MPI_Isend(const_cast<T*>(buf), count, dt, dest, tag, p.comm(), req);
  if ( ! ireq) MPI_Request_free(req);
#ifdef COMPOSE_DEBUG_MPI
  else ireq->unfreed++;
#endif
  return ret;
}

template <typename T>
int irecv (const Parallel& p, T* buf, int count, int src, int tag, Request* ireq) {
  MPI_Datatype dt = get_type<T>();
  MPI_Request ureq;
  MPI_Request* req = ireq ? &ireq->request : &ureq;
  int ret = MPI_Irecv(buf, count, dt, src, tag, p.comm(), req);
  if ( ! ireq) MPI_Request_free(req);
#ifdef COMPOSE_DEBUG_MPI
  else ireq->unfreed++;
#endif
  return ret;
}

int waitany(int count, Request* reqs, int* index, MPI_Status* stats = nullptr);
int waitall(int count, Request* reqs, MPI_Status* stats = nullptr);
int wait(Request* req, MPI_Status* stat = nullptr);
} // namespace mpi

namespace islmpi {
// Meta and bulk data for the interpolation SL method with special comm pattern.

#ifdef COMPOSE_BOUNDS_CHECK
# define slmm_assert_high(condition) slmm_assert(condition)
# define slmm_kernel_assert_high(condition) slmm_kernel_assert(condition)
#else
# define slmm_assert_high(condition)
# define slmm_kernel_assert_high(condition)
#endif

// FixedCapList, ListOfLists, and BufferLayoutArray are simple and somewhat
// problem-specific array data structures for use in IslMpi.
template <typename T, typename ES>
struct FixedCapList {
  typedef ko::View<T*, ES> Array;
  typedef FixedCapList<T, typename Array::host_mirror_space> Mirror;

  // Empty ctor b/c we need to permit view of view construction.
  SLMM_KIF FixedCapList () {}

  FixedCapList (const Int& cap) {
    slmm_assert_high(cap >= 0);
    reset_capacity(cap);
  }

  void reset_capacity (const Int& cap, const bool also_size = false) {
    slmm_assert(cap >= 0);
    if (d_.size() == 0) init_n();
    ko::resize(d_, cap);
    set_n_from_host(also_size ? cap : 0);
  }

  SLMM_KIF Int capacity () const { return d_.size(); }

  // If empty ctor was called, nothing below here is valid until reset_capacity
  // is called.

  SLMM_KIF void clear () const { set_n(0); }
  SLMM_KIF Int size () const { return n(); }
  SLMM_KIF T& operator() (const Int& i) const { slmm_kernel_assert_high(i >= 0 && i < n()); return d_[i]; }
  SLMM_KIF void inc () const { ++get_n_ref(); slmm_kernel_assert_high(n() <= static_cast<Int>(d_.size())); }
  SLMM_KIF void inc (const Int& dn) const { get_n_ref() += dn; slmm_kernel_assert_high(n() <= static_cast<Int>(d_.size())); }
  SLMM_KIF T& atomic_inc_and_return_next () const {
#ifdef COMPOSE_PORT
    volatile Int* n_vol = &n_();
    Int n_read = -1;
    for (;;) {
      n_read = *n_vol;
      if (ko::atomic_compare_exchange_strong(n_vol, n_read, n_read+1))
        break;
    }
    return d_[n_read];
#else
    inc();
    return back();
#endif
  }

  SLMM_KIF T* data () const { return d_.data(); }  
  SLMM_KIF T& back () const { slmm_kernel_assert_high(n() > 0); return d_[n()-1]; }
  SLMM_KIF T* begin () const { return d_.data(); }
  SLMM_KIF T* end () const { return d_.data() + n(); }

  void zero () { if (d_.size()) ko::deep_copy(d_, 0); }

  // Copy from s to this.
  template <typename ESS>
  void copy (const FixedCapList<T, ESS>& s) {
    siqk::resize_and_copy(d_, s.view());
    init_n();
    set_n(s);
  }

  // Use everything that follows only for low-level host-device things.

  SLMM_KIF void set_n (const Int& n0) const { get_n_ref() = n0; }

  template <typename ESS>
  void set_n (const ESS& s) const {
#ifdef COMPOSE_PORT
    ko::deep_copy(n_, s.n());
#else
    get_n_ref() = s.n();
#endif
  }

#ifdef COMPOSE_PORT
  typedef ko::View<Int, ES> NT;

  SLMM_KIF Int n () const { return n_(); }

  void init_n () { n_ = NT("FixedCapList::n_"); }
  void set_n_from_host (const Int& n0) { ko::deep_copy(n_, n0); }

  // Create a FixedCapList whose View is a mirror view of this.
  Mirror mirror () const {
    Mirror v;
    v.set_view(ko::create_mirror_view(d_));
    v.set_n_view(ko::create_mirror_view(n_));
    ko::deep_copy(v.n_view(), n_);
    return v;
  }

  const ko::View<Int, ES>& n_view () const { return n_; }
  void set_n_view (const NT& v) { n_ = v; }
#else
  typedef Int NT;

  SLMM_KIF Int n () const { return n_; }

  void init_n () {}
  void set_n_from_host (const Int& n0) { set_n(n0); }

  Mirror mirror () const {
    Mirror v;
    v.set_view(ko::create_mirror_view(d_));
    v.set_n(n_);
    return v;
  }
#endif

  const Array& view () const { return d_; }
  void set_view (const Array v) { d_ = v; }

private:
  Array d_;

#ifndef COMPOSE_PORT
  // You'll notice in a number of spots that there is strange const/mutable
  // stuff going on. This is driven by what Kokkos needs.
  mutable
#endif
  NT n_;

#ifdef COMPOSE_PORT
  SLMM_KIF Int& get_n_ref () const { return n_(); }
#else
  SLMM_KIF Int& get_n_ref () const { return n_; }
#endif
};

template <typename T, typename ESD, typename ESS>
void deep_copy (FixedCapList<T, ESD>& d, const FixedCapList<T, ESS>& s) {
  slmm_assert_high(d.capacity() == s.capacity());
  if (d.view().size() > 0) ko::deep_copy(d.view(), s.view());
#ifdef COMPOSE_PORT
  ko::deep_copy(d.n_view(), s.n_view());
#else
  d.set_n(s.n());
#endif
}

template <typename ES> struct BufferLayoutArray;

template <typename T, typename ES>
struct ListOfLists {
  template <typename T1> using Array = ko::View<T1*, ES>;
  typedef ListOfLists<T, typename Array<T>::host_mirror_space> Mirror;

  struct List {
    SLMM_KIF Int n () const { return n_; }

    SLMM_KIF T& operator() (const Int& i) const {
      slmm_kernel_assert_high(i >= 0 && i < n_); return d_[i];
    }

    SLMM_KIF T* data () const { return d_; }
    SLMM_KIF T* begin () const { return d_; }
    SLMM_KIF T* end () const { return d_ + n_; }

  private:
    friend class ListOfLists<T, ES>;
    SLMM_KIF List (T* d, const Int& n) : d_(d), n_(n) { slmm_kernel_assert_high(n_ >= 0); }
    T* const d_;
    const Int n_;
  };

  ListOfLists () {}
  ListOfLists (const Int nlist, const Int* nlist_per_list) { init(nlist, nlist_per_list); }
  void init (const Int nlist, const Int* nlist_per_list, T* buf = nullptr) {
    slmm_assert(nlist >= 0);
    ptr_ = Array<Int>("ptr_", nlist+1);
    ptr_h_ = ko::create_mirror_view(ptr_);
    ptr_h_[0] = 0;
    for (Int i = 0; i < nlist; ++i) {
      slmm_assert(nlist_per_list[i] >= 0);
      ptr_h_[i+1] = ptr_h_[i] + nlist_per_list[i];
    }
    ko::deep_copy(ptr_, ptr_h_);
    if (buf) {
      d_ = Array<T>(buf, ptr_h_[nlist]);
    } else {
      d_ = Array<T>("d_", ptr_h_[nlist]);
    }
  }

  SLMM_KIF Int n () const { return static_cast<Int>(ptr_.size()) - 1; }
  SLMM_KIF List operator() (const Int& i) const {
    slmm_kernel_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1);
    return List(const_cast<T*>(&d_[ptr_[i]]), ptr_[i+1] - ptr_[i]);
  }
  List get_h (const Int& i) const {
    slmm_kernel_assert_high(i >= 0 && i < static_cast<Int>(ptr_h_.size()) - 1);
    return List(const_cast<T*>(d_.data() + ptr_h_[i]), ptr_h_[i+1] - ptr_h_[i]);
  }
  SLMM_KIF T& operator() (const Int& i, const Int& j) const {
    slmm_kernel_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1 &&
                            j >= 0 && j < ptr_[i+1] - ptr_[i]);
    return d_[ptr_[i] + j];
  }

  void zero () {
#ifdef COMPOSE_HORIZ_OPENMP
#   pragma omp for
    for (Int i = 0; i < ptr_[n()]; ++i)
      d_[i] = 0;
#else
    ko::deep_copy(d_, 0);
#endif
  }

  SLMM_KIF T* data () const { return d_.data(); }

  // For device-host stuff:

  void set_views (const Array<T>& d, const Array<Int>& ptr,
                  const typename Array<Int>::HostMirror& ptr_h) {
    d_ = d; ptr_ = ptr; ptr_h_ = ptr_h;
  }

  Mirror mirror () const {
    Mirror v;
    const auto ptr = ko::create_mirror_view(ptr_);
    ko::deep_copy(ptr, ptr_h_);
    v.set_views(ko::create_mirror_view(d_), ptr, ptr_h_);
    return v;
  }

  const Array<T>& d_view () const { return d_; }
  const Array<Int>& ptr_view () const { return ptr_; }
  const typename Array<Int>::HostMirror& ptr_h_view () const { return ptr_h_; }

private:
  friend class BufferLayoutArray<ES>;
  Array<T> d_;
  Array<Int> ptr_;
  typename Array<Int>::HostMirror ptr_h_;
};

struct LayoutTriple {
  Int xptr, qptr, cnt;
  SLMM_KIF LayoutTriple () : LayoutTriple(0) {}
  SLMM_KIF LayoutTriple (const Int& val) { xptr = qptr = cnt = 0; }
};

template <typename T, typename ESD, typename ESS>
void deep_copy (ListOfLists<T, ESD>& d, const ListOfLists<T, ESS>& s) {
  ko::deep_copy(d.d_view(), s.d_view());
  ko::deep_copy(d.ptr_view(), s.ptr_view());
  ko::deep_copy(d.ptr_h_view(), s.ptr_h_view());
}

template <typename ES>
struct BufferLayoutArray {
  struct BufferRankLayoutArray {
    SLMM_KIF LayoutTriple& operator() (const Int& lidi, const Int& lev) const {
      slmm_kernel_assert_high(lidi >= 0 && lev >= 0 && lidi*nlev_ + lev < d_.n());
      return d_(lidi*nlev_ + lev);
    }

  private:
    friend class BufferLayoutArray;
    SLMM_KIF BufferRankLayoutArray (const typename ListOfLists<LayoutTriple, ES>::List& d,
                                    const Int& nlev)
      : d_(d), nlev_(nlev) {}
    typename ListOfLists<LayoutTriple, ES>::List d_;
    Int nlev_;
  };

  BufferLayoutArray () : nlev_(0) {}
  BufferLayoutArray (const Int& nrank, const Int* nlid_per_rank, const Int& nlev)
  { init(nrank, nlid_per_rank, nlev); }

  void init (const Int& nrank, const Int* nlid_per_rank, const Int& nlev) {
    slmm_assert(nrank >= 0 && nlev > 0);
    nlev_ = nlev;
    std::vector<Int> ns(nrank);
    for (Int i = 0; i < nrank; ++i) {
      slmm_assert(nlid_per_rank[i] > 0);
      ns[i] = nlid_per_rank[i] * nlev;
    }
    d_.init(nrank, ns.data());
  }

  Int nlev () const { return nlev_; }

  void zero () { d_.zero(); }

  SLMM_KIF LayoutTriple& operator() (const Int& ri, const Int& lidi, const Int& lev) const {
    slmm_kernel_assert_high(ri >= 0 && ri < d_.n() &&
                            lidi >= 0 && lev >= 0 &&
                            lidi*nlev_ + lev < d_(ri).n());
    return d_.data()[d_.ptr_[ri] + lidi*nlev_ + lev];
  }

  SLMM_KIF BufferRankLayoutArray operator() (const Int& ri) const {
    slmm_kernel_assert_high(ri >= 0 && ri < d_.n());
    return BufferRankLayoutArray(d_(ri), nlev_);
  }

  // For device-host stuff:

  typedef BufferLayoutArray<typename ko::View<Int,ES>::host_mirror_space> Mirror;

  ListOfLists<LayoutTriple, ES>& get_lol () { return d_; }
  const ListOfLists<LayoutTriple, ES>& get_lol () const { return d_; }
  void set_lol (const ListOfLists<LayoutTriple, ES>& d) { d_ = d; }
  void set_nlev (const Int& nlev) { nlev_ = nlev; }

  Mirror mirror () const {
    Mirror v;
    const auto m = d_.mirror();
    v.set_lol(m);
    v.set_nlev(nlev_);
    return v;
  }

private:
  ListOfLists<LayoutTriple, ES> d_;
  Int nlev_;
};

template <typename ESD, typename ESS>
void deep_copy (BufferLayoutArray<ESD>& d, const BufferLayoutArray<ESS>& s) {
  slmm_assert(d.nlev() == s.nlev());
  deep_copy(d.get_lol(), s.get_lol());
}

struct GidRank {
  Int
  gid,      // cell global ID
    rank,     // the rank that owns the cell
    rank_idx, // index into list of ranks with whom I communicate, including me
    lid_on_rank,     // the local ID of the cell on the owning rank
    lid_on_rank_idx; // index into list of LIDs on the rank
};
struct OwnItem {
  short lev;   // level index
  short k;     // linearized GLL index
};
struct RemoteItem {
  Int q_extrema_ptr, q_ptr; // pointers into recvbuf
  short lev, k;
};

// Meta and bulk data for the interpolation SL MPI communication pattern.
template <typename MT = ko::MachineTraits>
struct IslMpi {
  typedef typename MT::HES HES;
  typedef typename MT::DES DES;

  using Advecter = slmm::Advecter<MT>;

  typedef std::shared_ptr<IslMpi> Ptr;

  template <typename Datatype, typename ES>
  using Array = ko::View<Datatype, siqk::Layout, ES>;
  template <typename Datatype>
  using ArrayH = ko::View<Datatype, siqk::Layout, HES>;
  template <typename Datatype>
  using ArrayD = ko::View<Datatype, siqk::Layout, DES>;

  // The comm and real data associated with an element patch, the set of
  // elements surrounding an owned cell.
  template <typename ES>
  struct ElemData {
    GidRank* me;                      // the owned cell
    FixedCapList<GidRank, ES> nbrs;   // the cell's neighbors (but including me)
    Int nin1halo;                     // nbrs[0:n]
    FixedCapList<OwnItem, ES> own;    // points whose q are computed with own rank's data
    FixedCapList<RemoteItem, ES> rmt; // points computed by a remote rank's data
    Array<Int**, ES> src;             // src(lev,k) = get_src_cell
    Array<Real**[2], ES> q_extrema;
    const Real* qdp, * dp;  // the owned cell's data
    Real* q;
  };

  typedef ElemData<HES> ElemDataH;
  typedef ElemData<DES> ElemDataD;
  typedef FixedCapList<ElemDataH, HES> ElemDataListH;
  typedef FixedCapList<ElemDataD, DES> ElemDataListD;

  const mpi::Parallel::Ptr p;
  const typename Advecter::ConstPtr advecter;
  const Int np, np2, nlev, qsize, qsized, nelemd, halo;

  ElemDataListH ed_h; // this rank's owned cells, indexed by LID
  ElemDataListD ed_d;
  typename ElemDataListD::Mirror ed_m; // handle managed allocs

  //typename TracerArrays<MT>::Ptr tracer_arrays;
  TracerArrays<MT> tracer_arrays;

  // IDs.
  FixedCapList<Int, HES> ranks, mylid_with_comm_tid_ptr_h;
  FixedCapList<Int, DES> nx_in_rank, mylid_with_comm_d;
  ListOfLists <Int, DES> nx_in_lid, lid_on_rank;
  BufferLayoutArray<DES> bla;

  // MPI comm data.
  FixedCapList<mpi::Request, HES> sendreq, recvreq;
  ListOfLists<Real, DES> sendbuf, recvbuf;
  FixedCapList<Int, DES> sendcount, x_bulkdata_offset;
  ListOfLists<Real, HES> sendbuf_meta_h, recvbuf_meta_h; // not mirrors
  FixedCapList<Int, DES> rmt_xs, rmt_qs_extrema;
  Int nrmt_xs, nrmt_qs_extrema;

  // Mirror views.
  typename FixedCapList<Int, DES>::Mirror nx_in_rank_h, sendcount_h,
    x_bulkdata_offset_h, rmt_xs_h, rmt_qs_extrema_h, mylid_with_comm_h;
  typename ListOfLists <Int, DES>::Mirror nx_in_lid_h, lid_on_rank_h;
  typename BufferLayoutArray<DES>::Mirror bla_h;

  bool horiz_openmp;
#ifdef COMPOSE_HORIZ_OPENMP
  ListOfLists<omp_lock_t, HES> ri_lidi_locks;
#endif

  // temporary work space
  std::vector<Int> nlid_per_rank, sendsz, recvsz, sendmetasz, recvmetasz;
  Array<Real**,DES> rwork;

  IslMpi (const mpi::Parallel::Ptr& ip, const typename Advecter::ConstPtr& advecter,
          Int inp, Int inlev, Int iqsize, Int iqsized, Int inelemd, Int ihalo)
    : p(ip), advecter(advecter),
      np(inp), np2(np*np), nlev(inlev), qsize(iqsize), qsized(iqsized), nelemd(inelemd),
      halo(ihalo), tracer_arrays(nelemd, nlev, np2, qsize)
  {}

  IslMpi(const IslMpi&) = delete;
  IslMpi& operator=(const IslMpi&) = delete;

  ~IslMpi () {
#ifdef COMPOSE_HORIZ_OPENMP
    const Int nrmtrank = static_cast<Int>(ranks.n()) - 1;
    for (Int ri = 0; ri < nrmtrank; ++ri) {
      auto&& locks = ri_lidi_locks(ri);
      for (auto& lock: locks) {
        // The following call is causing a seg fault on at least one Cray KNL
        // system. It doesn't kill the run b/c it occurs after main exits. Not
        // calling this is a memory leak, but it's innocuous in this case. Thus,
        // comment it out:
        //omp_destroy_lock(&lock);
        // It's possible that something in the OpenMP runtime shuts down before
        // this call, and that's the problem. If that turns out to be it, I
        // should make a compose_finalize() call somewhere.
      }
    }
#endif
  }
};

inline int get_tid () {
#ifdef COMPOSE_HORIZ_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

inline int get_num_threads () {
#ifdef COMPOSE_HORIZ_OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

template <typename MT>
void alloc_mpi_buffers (IslMpi<MT>& cm, Real* sendbuf = nullptr, Real* recvbuf = nullptr);

template <typename MT>
void setup_comm_pattern(IslMpi<MT>& cm, const Int* nbr_id_rank, const Int* nirptr);

namespace extend_halo {
template <typename MT>
void extend_local_meshes(const mpi::Parallel& p,
                         const typename IslMpi<MT>::ElemDataListH& eds,
                         typename IslMpi<MT>::Advecter& advecter);
} // namespace extend_halo

template <typename MT>
void analyze_dep_points(IslMpi<MT>& cm, const Int& nets, const Int& nete,
                        const DepPoints<MT>& dep_points);

template <typename MT>
void init_mylid_with_comm_threaded(IslMpi<MT>& cm, const Int& nets, const Int& nete);
template <typename MT>
void setup_irecv(IslMpi<MT>& cm, const bool skip_if_empty = false);
template <typename MT>
void isend(IslMpi<MT>& cm, const bool want_req = true, const bool skip_if_empty = false);
template <typename MT>
void recv_and_wait_on_send(IslMpi<MT>& cm);
template <typename MT>
void wait_on_send (IslMpi<MT>& cm, const bool skip_if_empty = false);
template <typename MT>
void recv(IslMpi<MT>& cm, const bool skip_if_empty = false);

const int nreal_per_2int = (2*sizeof(Int) + sizeof(Real) - 1) / sizeof(Real);

template <typename MT>
void pack_dep_points_sendbuf_pass1(IslMpi<MT>& cm);
template <typename MT>
void pack_dep_points_sendbuf_pass2(IslMpi<MT>& cm, const DepPoints<MT>& dep_points);

template <typename MT>
void calc_q_extrema(IslMpi<MT>& cm, const Int& nets, const Int& nete);

template <typename MT>
void calc_rmt_q(IslMpi<MT>& cm);
template <typename MT>
void calc_own_q(IslMpi<MT>& cm, const Int& nets, const Int& nete,
                const DepPoints<MT>& dep_points,
                const QExtrema<MT>& q_min, const QExtrema<MT>& q_max);
template <typename MT>
void copy_q(IslMpi<MT>& cm, const Int& nets,
            const QExtrema<MT>& q_min, const QExtrema<MT>& q_max);

/* dep_points is const in principle, but if lev <=
   semi_lagrange_nearest_point_lev, a departure point may be altered if the
   winds take it outside of the comm halo.
*/
template <typename MT = ko::MachineTraits>
void step(
  IslMpi<MT>& cm, const Int nets, const Int nete,
  Real* dep_points_r,            // dep_points(1:3, 1:np, 1:np)
  Real* q_min_r, Real* q_max_r); // q_{min,max}(1:np, 1:np, lev, 1:qsize, ie-nets+1)

} // namespace islmpi
} // namespace homme

#endif
