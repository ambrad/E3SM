#ifndef INCLUDE_COMPOSE_HOMME_HPP
#define INCLUDE_COMPOSE_HOMME_HPP

#include "compose.hpp"

namespace homme {
typedef int Int;
typedef double Real;

namespace ko = Kokkos;

// Fortran array wrappers with Fortran index order.
template <typename T> using FA2 = ko::View<T**,    ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA3 = ko::View<T***,   ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA4 = ko::View<T****,  ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA5 = ko::View<T*****, ko::LayoutLeft, ko::HostSpace>;

struct Cartesian3D { Real x, y, z; };

template <typename T, int rank_>
struct HommeFormatArray {
  enum : int { rank = rank_ };
  typedef T value_type;

  HommeFormatArray (Int nelemd, Int np2_, Int nlev_, Int qsize_ = -1)
    : nlev(nlev_), np2(np2_), qsize(qsize_)
  { ie_data_ptr.resize(nelemd); }

  void set_ie_ptr (const Int ie, T* ptr) {
    check(ie);
    ie_data_ptr[ie] = ptr;
  }

  T& operator() (const Int& ie, const Int& k, const Int& lev) const {
    static_assert(rank == 3, "rank 3 array");
    check(ie, k, lev);
    return *(ie_data_ptr[ie] + lev*np2 + k);
  }
  T& operator() (const Int& ie, const Int& q, const Int& k, const Int& lev) const {
    static_assert(rank == 4, "rank 4 array");
    check(ie, k, lev, q);
    return *(ie_data_ptr[ie] + (q*nlev + lev)*np2 + k);
  }

private:
  std::vector<T*> ie_data_ptr;
  const Int nlev, np2, qsize;

  void check (Int ie, Int k = -1, Int lev = -1, Int q = -1) const {
#ifdef COMPOSE_BOUNDS_CHECK
    assert(ie >= 0 && ie < static_cast<Int>(ie_data_ptr.size()));
    if (k >= 0) assert(k < np2);
    if (lev >= 0) assert(lev < nlev);
    if (q >= 0) assert(q < qsize);
#endif    
  }
};

// Qdp, dp, Q
template <typename MT>
struct TracerArrays {
#if defined COMPOSE_PORT_DEV
# if defined COMPOSE_PORT_DEV_VIEWS
  template <typename Datatype>
  using View = ko::View<Datatype, ko::LayoutRight, typename MT::DES>;
  View<Real****> qdp;
  View<Real***> dp;
  View<Real****> q;
  HommeFormatArray<const Real,4> pqdp;
  HommeFormatArray<const Real,3> pdp;
  HommeFormatArray<Real,4> pq;
# else
  HommeFormatArray<const Real,4> & qdp, pqdp;
  HommeFormatArray<const Real,3> & dp, pdp;
  HommeFormatArray<Real,4> & q, pq;
# endif
#else
#endif

  TracerArrays (Int nelemd, Int nlev, Int np2, Int qsize)
#if defined COMPOSE_PORT_DEV
    : pqdp(nelemd, np2, nlev, qsize), pdp(nelemd, np2, nlev), pq(nelemd, np2, nlev, qsize),
# if defined COMPOSE_PORT_DEV_VIEWS
      qdp("qdp", nelemd, qsize, np2, nlev), dp("dp", nelemd, np2, nlev), q("q", nelemd, qsize, np2, nlev)
# else
      qdp(pqdp), dp(pdp), q(pq)
# endif
#endif
  {}
};

} // namespace homme

#endif
