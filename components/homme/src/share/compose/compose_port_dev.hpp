#ifndef INCLUDE_COMPOSE_PORT_DEV_HPP
#define INCLUDE_COMPOSE_PORT_DEV_HPP

#include "compose_homme.hpp"

namespace homme {

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

  T& operator() (const Int& ie, const Int& lev, const Int& k) const {
    static_assert(rank == 3, "rank 3 array");
    check(ie, k, nlev);
    return *(ie_data_ptr[ie] + lev*np2 + k);
  }
  T& operator() (const Int& q, const Int& ie, const Int& lev, const Int& k) const {
    static_assert(rank == 4, "rank 4 array");
    check(ie, k, nlev, q);
    return *(ie_data_ptr[ie] + (q*nlev + lev)*np2 + k);
  }

private:
  std::vector<Real*> ie_data_ptr;
  const Int nlev, np2, qsize;

  bool check (Int ie, Int k = -1, Int lev = -1, Int q = -1) {
#ifdef COMPOSE_BOUNDS_CHECK
    assert(ie >= 0 && ie < static_cast<Int>(ie_data_ptr.size()));
    if (k >= 0) assert(k < np2);
    if (lev >= 0) assert(lev < nlev);
    if (q >= 0) assert(q < qsize);
#endif    
  }
};

} // namespace homme

#endif
