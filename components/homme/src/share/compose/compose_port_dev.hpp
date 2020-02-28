#ifndef INCLUDE_COMPOSE_PORT_DEV_HPP
#define INCLUDE_COMPOSE_PORT_DEV_HPP

#include "compose_homme.hpp"

namespace homme {

template <typename T, int rank_>
struct HommeFormatArray {
  enum : int { rank = rank_ };
  typedef T value_type;

  HommeFormatArray (Int nelemd, Int nlev_, Int np2_, Int qsize_ = -1)
    : nlev(nlev_), np2(np2_), qsize(qsize_)
  {
    ie_data_ptr.resize(nelemd);
  }

  void set_ie_ptr (const Int ie, T* ptr) {
    slmm_assert_high(ie >= 0 && ie < static_cast<Int>(ie_data_ptr.size()));
    ie_data_ptr[ie] = ptr;
  }

  T& operator() (const Int& k, const Int& lev, const Int& ie,
                 typename std::enable_if<rank == 3>::type* = 0) const
  { return *(ie_data_ptr[ie] + lev*np2 + k); }
  T& operator() (const Int& k, const Int& lev, const Int& ie, const Int& q,
                 typename std::enable_if<rank == 4>::type* = 0) const
  { return *(ie_data_ptr[ie] + (q*nlev + lev)*np2 + k); }

private:
  std::vector<Real*> ie_data_ptr;
  const Int nlev, np2, qsize;
  
};

} // namespace homme

#endif
