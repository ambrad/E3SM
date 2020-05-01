#ifndef INCLUDE_COMPOSE_HOMME_HPP
#define INCLUDE_COMPOSE_HOMME_HPP

#include "compose.hpp"
#include "compose_kokkos.hpp"

#include <memory>

namespace homme {
typedef int Int;
typedef double Real;

namespace ko = Kokkos;

// Fortran array wrappers with Fortran index order.
template <typename T> using FA2 = ko::View<T**,    ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA3 = ko::View<T***,   ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA4 = ko::View<T****,  ko::LayoutLeft, ko::HostSpace>;
template <typename T> using FA5 = ko::View<T*****, ko::LayoutLeft, ko::HostSpace>;

template <typename MT> using DepPoints =
  ko::View<Real***[3], ko::LayoutRight, typename MT::DES>;
template <typename MT> using QExtrema =
  ko::View<Real****, ko::LayoutRight, typename MT::DES>;
template <typename MT> using DepPointsH = typename DepPoints<MT>::HostMirror;
template <typename MT> using QExtremaH = typename QExtrema<MT>::HostMirror;

struct Cartesian3D { Real x, y, z; };

template <typename T, int rank_>
struct HommeFormatArray {
  enum : int { rank = rank_ };
  typedef T value_type;

  HommeFormatArray (Int nelemd, Int np2_, Int nlev_, Int qsize_ = -1, Int ntimelev_ = -1)
    : nlev(nlev_), np2(np2_), qsize(qsize_), ntimelev(ntimelev_)
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
  T& operator() (const Int& ie, const Int& q_or_timelev, const Int& k, const Int& lev) const {
    static_assert(rank == 4, "rank 4 array");
    check(ie, k, lev, q_or_timelev);
    return *(ie_data_ptr[ie] + (q_or_timelev*nlev + lev)*np2 + k);
  }
  T& operator() (const Int& ie, const Int& timelev, const Int& q, const Int& k, const Int& lev) const {
    static_assert(rank == 4, "rank 4 array");
    check(ie, k, lev, q, timelev);
    return *(ie_data_ptr[ie] + ((timelev*qsize + q)*nlev + lev)*np2 + k);
  }

private:
  std::vector<T*> ie_data_ptr;
  const Int nlev, np2, qsize, ntimelev;

  void check (Int ie, Int k = -1, Int lev = -1, Int q_or_timelev = -1, Int timelev = -1) const {
#ifdef COMPOSE_BOUNDS_CHECK
    assert(ie >= 0 && ie < static_cast<Int>(ie_data_ptr.size()));
    if (k >= 0) assert(k < np2);
    if (lev >= 0) assert(lev < nlev);
    if (q_or_timelev >= 0) {
      if (qsize < 0)
        assert(q_or_timelev < ntimelev);
    }
    if (timelev >= 0) assert(timelev < ntimelev);
#endif    
  }
};

// Qdp, dp, Q
template <typename MT>
struct TracerArrays {
  typedef std::shared_ptr<TracerArrays<MT> > Ptr;

#if defined COMPOSE_PORT_DEV
# if defined COMPOSE_PORT_DEV_VIEWS
  template <typename Datatype>
  using View = ko::View<Datatype, ko::LayoutRight, typename MT::DES>;
  View<Real*****> qdps; // elem%state%Qdp(:,:,:,:,:)
  View<Real****>  qdp;  // elem%state%Qdp(:,:,:,:,n0_qdp)
  View<Real****>  q;    // elem%state%Q
  View<Real***>   dp;   // elem%derived%dp
  View<Real***>   dp3d; // elem%state%dp3d or the sl3d equivalent
  HommeFormatArray<const Real,4> pqdp;
  HommeFormatArray<const Real,3> pdp, pdp3d;
  HommeFormatArray<Real,4> pq;
  DepPoints<MT> dep_points;
  QExtrema<MT> q_min, q_max;
# else
  HommeFormatArray<const Real,4> & qdp, pqdp;
  HommeFormatArray<const Real,3> & dp, pdp, & dp3d, pdp3d;
  HommeFormatArray<Real,4> & q, pq;
# endif
#endif

  TracerArrays(Int nelemd, Int nlev, Int np2, Int qsize);
  TracerArrays(const TracerArrays<MT>&) = delete;
  TracerArrays& operator=(const TracerArrays<MT>&) = delete;
};

template <typename MT>
void sl_h2d(const TracerArrays<MT>& ta, Cartesian3D* dep_points);

template <typename MT>
void sl_d2h(const TracerArrays<MT>& ta, Cartesian3D* dep_points, Real* minq, Real* maxq);

TracerArrays<ko::MachineTraits>::Ptr init_tracer_arrays(Int nelemd, Int nlev, Int np2, Int qsize);
TracerArrays<ko::MachineTraits>::Ptr get_tracer_arrays();
void delete_tracer_arrays();

} // namespace homme

#endif
