#ifndef INCLUDE_COMPOSE_SLMM_ISLMPI_HPP
#define INCLUDE_COMPOSE_SLMM_ISLMPI_HPP

#include "compose_slmm.hpp"

#include <mpi.h>

#include <memory>

// Uncomment this to look for MPI-related memory leaks.
//#define COMPOSE_DEBUG_MPI

namespace homme {
typedef slmm::Int Int;
typedef slmm::Real Real;

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

Parallel::Ptr make_parallel (MPI_Comm comm) {
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

#ifdef COMPOSE_DEBUG_MPI
Request::Request () : unfreed(0) {}
Request::~Request () {
  if (unfreed) {
    std::stringstream ss;
    ss << "Request is being deleted with unfreed = " << unfreed;
    int fin;
    MPI_Finalized(&fin);
    if (fin) {
      ss << "\n";
      std::cerr << ss.str();
    } else {
      pr(ss.str());
    }
  }
}
#endif

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

int waitany (int count, Request* reqs, int* index, MPI_Status* stats = nullptr) {
#ifdef COMPOSE_DEBUG_MPI
  std::vector<MPI_Request> vreqs(count);
  for (int i = 0; i < count; ++i) vreqs[i] = reqs[i].request;
  const auto out = MPI_Waitany(count, vreqs.data(), index,
                               stats ? stats : MPI_STATUS_IGNORE);
  for (int i = 0; i < count; ++i) reqs[i].request = vreqs[i];
  reqs[*index].unfreed--;
  return out;
#else
  return MPI_Waitany(count, reinterpret_cast<MPI_Request*>(reqs), index,
                     stats ? stats : MPI_STATUS_IGNORE);
#endif
}

int waitall (int count, Request* reqs, MPI_Status* stats = nullptr) {
#ifdef COMPOSE_DEBUG_MPI
  std::vector<MPI_Request> vreqs(count);
  for (int i = 0; i < count; ++i) vreqs[i] = reqs[i].request;
  const auto out = MPI_Waitall(count, vreqs.data(),
                               stats ? stats : MPI_STATUS_IGNORE);
  for (int i = 0; i < count; ++i) {
    reqs[i].request = vreqs[i];
    reqs[i].unfreed--;
  }
  return out;
#else
  return MPI_Waitall(count, reinterpret_cast<MPI_Request*>(reqs),
                     stats ? stats : MPI_STATUS_IGNORE);
#endif
}
} // namespace mpi

namespace islmpi {
// Meta and bulk data for the interpolation SL method with special comm pattern.

#define SLMM_BOUNDS_CHECK
#ifdef SLMM_BOUNDS_CHECK
# define slmm_assert_high(condition) slmm_assert(condition)
#else
# define slmm_assert_high(condition)
#endif

// FixedCapList, ListOfLists, and BufferLayoutArray are simple and somewhat
// problem-specific array data structures for use in CslMpi.
template <typename T>
struct FixedCapList {
  FixedCapList () : n_(0) {}
  FixedCapList (const Int& cap) { slmm_assert_high(cap >= 0); reset_capacity(cap); }
  void reset_capacity (const Int& cap, const bool also_size = false) {
    slmm_assert(cap >= 0);
    d_.resize(cap);
    n_ = also_size ? cap : 0;
  }
  void clear () { n_ = 0; }

  Int n () const { return n_; }
  Int size () const { return n_; }
  Int capacity () const { return d_.size(); }
  const T& operator() (const Int& i) const { slmm_assert_high(i >= 0 && i < n_); return d_[i]; }
  T& operator() (const Int& i) { slmm_assert_high(i >= 0 && i < n_); return d_[i]; }
  void inc () { ++n_; slmm_assert_high(n_ <= static_cast<Int>(d_.size())); }
  void inc (const Int& dn) { n_ += dn; slmm_assert_high(n_ <= static_cast<Int>(d_.size())); }

  const T* data () const { return d_.data(); }
  T* data () { return d_.data(); }  
  const T& back () const { slmm_assert_high(n_ > 0); return d_[n_-1]; }
  T& back () { slmm_assert_high(n_ > 0); return d_[n_-1]; }
  const T* begin () const { return d_.data(); }
  T* begin () { return d_.data(); }
  const T* end () const { return d_.data() + n_; }
  T* end () { return d_.data() + n_; }

 private:
  std::vector<T> d_;
  Int n_;
};

template <typename T>
struct ListOfLists {
  struct List {
    Int n () const { return n_; }

    T& operator() (const Int& i) { slmm_assert_high(i >= 0 && i < n_); return d_[i]; }
    const T& operator() (const Int& i) const { slmm_assert_high(i >= 0 && i < n_); return d_[i]; }

    const T* data () const { return d_; }
    T* data () { return d_; }
    const T* begin () const { return d_; }
    T* begin () { return d_; }
    const T* end () const { return d_ + n_; }
    T* end () { return d_ + n_; }

    void zero () { for (Int i = 0; i < n_; ++i) d_[i] = 0; }

  private:
    friend class ListOfLists<T>;
    List (T* d, const Int& n) : d_(d), n_(n) { slmm_assert_high(n_ >= 0); }
    T* const d_;
    const Int n_;
  };

  ListOfLists () {}
  ListOfLists (const Int nlist, const Int* nlist_per_list) { init(nlist, nlist_per_list); }
  void init (const Int nlist, const Int* nlist_per_list) {
    slmm_assert(nlist >= 0); 
    ptr_.resize(nlist+1);
    ptr_[0] = 0;
    for (Int i = 0; i < nlist; ++i) {
      slmm_assert(nlist_per_list[i] >= 0);
      ptr_[i+1] = ptr_[i] + nlist_per_list[i];
    }
    d_.resize(ptr_.back());
  }

  Int n () const { return static_cast<Int>(ptr_.size()) - 1; }
  List operator() (const Int& i) {
    slmm_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1);
    return List(&d_[ptr_[i]], ptr_[i+1] - ptr_[i]);
  }
  const List operator() (const Int& i) const {
    slmm_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1);
    return List(const_cast<T*>(&d_[ptr_[i]]), ptr_[i+1] - ptr_[i]);
  }
  T& operator() (const Int& i, const Int& j) {
    slmm_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1 &&
                     j >= 0 && j < ptr_[i+1] - ptr_[i]);
    return d_[ptr_[i] + j];
  }
  const T& operator() (const Int& i, const Int& j) const {
    slmm_assert_high(i >= 0 && i < static_cast<Int>(ptr_.size()) - 1 &&
                     j >= 0 && j < ptr_[i+1] - ptr_[i]);
    return d_[ptr_[i] + j];
  }

private:
  friend class BufferLayoutArray;
  std::vector<T> d_;
  std::vector<Int> ptr_;
  T* data () { return d_.data(); }
  const T* data () const { return d_.data(); }
};

struct LayoutTriple {
  Int xptr, qptr, cnt;
  LayoutTriple () : LayoutTriple(0) {}
  LayoutTriple (const Int& val) { xptr = qptr = cnt = 0; }
};

struct BufferLayoutArray {
  struct BufferRankLayoutArray {
    LayoutTriple& operator() (const Int& lidi, const Int& lev) {
      slmm_assert_high(lidi >= 0 && lev >= 0 && lidi*nlev_ + lev < d_.n());
      return d_(lidi*nlev_ + lev);
    }
    const LayoutTriple& operator() (const Int& lidi, const Int& lev) const {
      slmm_assert_high(lidi >= 0 && lev >= 0 && lidi*nlev_ + lev < d_.n());
      return d_(lidi*nlev_ + lev);
    }

  private:
    friend class BufferLayoutArray;
    BufferRankLayoutArray (const ListOfLists<LayoutTriple>::List& d, const Int& nlev)
      : d_(d), nlev_(nlev) {}
    ListOfLists<LayoutTriple>::List d_;
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

  void zero () {
    const Int ni = d_.n();
#ifdef HORIZ_OPENMP
#   pragma omp for
#endif
    for (Int i = 0; i < ni; ++i) {
      auto&& l = d_(i);
      for (Int j = 0, nj = l.n(); j < nj; ++j)
        l(j) = 0;
    }
  }

  LayoutTriple& operator() (const Int& ri, const Int& lidi, const Int& lev) {
    slmm_assert_high(ri >= 0 && ri < d_.n() &&
                     lidi >= 0 && lev >= 0 &&
                     lidi*nlev_ + lev < d_(ri).n());
    return d_.data()[d_.ptr_[ri] + lidi*nlev_ + lev];
  }
  const LayoutTriple& operator() (const Int& ri, const Int& lidi, const Int& lev) const {
    return const_cast<BufferLayoutArray*>(this)->operator()(ri, lidi, lev);
  }
  BufferRankLayoutArray operator() (const Int& ri) {
    slmm_assert_high(ri >= 0 && ri < d_.n());
    return BufferRankLayoutArray(d_(ri), nlev_);
  }
  const BufferRankLayoutArray operator() (const Int& ri) const {
    slmm_assert_high(ri >= 0 && ri < d_.n());
    return BufferRankLayoutArray(d_(ri), nlev_);
  }

private:
  ListOfLists<LayoutTriple> d_;
  Int nlev_;
};

} // namespace islmpi
} // namespace homme

#endif
