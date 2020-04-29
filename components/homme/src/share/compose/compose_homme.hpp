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
  View<Real*****> qdps; // elem%state%Qdp(:,:,:,:,:)
  View<Real****>  qdp;  // elem%state%Qdp(:,:,:,:,n0_qdp)
  View<Real****>  q;    // elem%state%Q
  View<Real***>   dp;   // elem%derived%dp
  View<Real***>   dp3d; // elem%state%dp3d or the sl3d equivalent
  HommeFormatArray<const Real,4> pqdp;
  HommeFormatArray<const Real,3> pdp;
  HommeFormatArray<Real,4> pq;
  DepPoints<MT> dep_points;
  QExtrema<MT> q_min, q_max;
# else
  HommeFormatArray<const Real,4> & qdp, pqdp;
  HommeFormatArray<const Real,3> & dp, pdp;
  HommeFormatArray<Real,4> & q, pq;
# endif
#endif

  TracerArrays (Int nelemd, Int nlev, Int np2, Int qsize)
#if defined COMPOSE_PORT_DEV
    : pqdp(nelemd, np2, nlev, qsize), pdp(nelemd, np2, nlev), pq(nelemd, np2, nlev, qsize),
# if defined COMPOSE_PORT_DEV_VIEWS
      qdps("qdps", 2, nelemd, qsize, np2, nlev), qdp(qdps.data(), nelemd, qsize, np2, nlev),
      q("q", nelemd, qsize, np2, nlev),
      dp("dp", nelemd, np2, nlev), dp3d("dp3d", nelemd, np2, nlev),
      dep_points("dep_points", nelemd, nlev, np2),
      q_min("q_min", nelemd, qsize, nlev, np2), q_max("q_max", nelemd, qsize, nlev, np2)
# else
      qdp(pqdp), dp(pdp), q(pq)
# endif
#endif
  {}
};

template <typename MT>
void sl_h2d (const TracerArrays<MT>& ta, Cartesian3D* dep_points) {
#if defined COMPOSE_PORT_DEV_VIEWS
  const auto qdp_m = ko::create_mirror_view(ta.qdp);
  const auto dp_m = ko::create_mirror_view(ta.dp);
  const auto q_m = ko::create_mirror_view(ta.q);
  const Int nelemd = q_m.extent_int(0), qsize = q_m.extent_int(1), np2 = q_m.extent_int(2),
    nlev = q_m.extent_int(3);
  for (Int ie = 0; ie < nelemd; ++ie)
    for (Int iq = 0; iq < qsize; ++iq)
      for (Int k = 0; k < np2; ++k)
        for (Int lev = 0; lev < nlev; ++lev) {
          qdp_m(ie,iq,k,lev) = ta.pqdp(ie,iq,k,lev);
          q_m(ie,iq,k,lev) = ta.pq(ie,iq,k,lev);
        }
  for (Int ie = 0; ie < q_m.extent_int(0); ++ie)
    for (Int k = 0; k < q_m.extent_int(2); ++k)
      for (Int lev = 0; lev < q_m.extent_int(3); ++lev)
        dp_m(ie,k,lev) = ta.pdp(ie,k,lev);
  ko::deep_copy(ta.qdp, qdp_m);
  ko::deep_copy(ta.dp, dp_m);
  ko::deep_copy(ta.q, q_m);
  const DepPointsH<MT> dep_points_h(reinterpret_cast<Real*>(dep_points), nelemd, nlev, np2);
  ko::deep_copy(ta.dep_points, dep_points_h);
#endif
}

template <typename MT>
void sl_d2h (const TracerArrays<MT>& ta, Cartesian3D* dep_points, Real* minq, Real* maxq) {
#if defined COMPOSE_PORT_DEV_VIEWS
  ko::fence();
  const auto q_m = ko::create_mirror_view(ta.q);
  const Int nelemd = q_m.extent_int(0), qsize = q_m.extent_int(1), np2 = q_m.extent_int(2),
    nlev = q_m.extent_int(3);
  ko::deep_copy(q_m, ta.q);
  for (Int ie = 0; ie < nelemd; ++ie)
    for (Int iq = 0; iq < qsize; ++iq)
      for (Int k = 0; k < np2; ++k)
        for (Int lev = 0; lev < nlev; ++lev)
          ta.pq(ie,iq,k,lev) = q_m(ie,iq,k,lev);
  const DepPointsH<MT> dep_points_h(reinterpret_cast<Real*>(dep_points), nelemd, nlev, np2);
  const QExtremaH<MT>
    q_min_h(minq, nelemd, qsize, nlev, np2),
    q_max_h(maxq, nelemd, qsize, nlev, np2);
  ko::deep_copy(dep_points_h, ta.dep_points);
  ko::deep_copy(q_min_h, ta.q_min);
  ko::deep_copy(q_max_h, ta.q_max);
#endif  
}

} // namespace homme

#endif
