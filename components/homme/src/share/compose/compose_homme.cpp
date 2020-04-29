#include "compose_homme.hpp"

namespace homme {

template <typename MT>
TracerArrays<MT>::TracerArrays (Int nelemd, Int nlev, Int np2, Int qsize)
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

TracerArrays<ko::MachineTraits>::Ptr& get_instance () {
  static typename TracerArrays<ko::MachineTraits>::Ptr p;
  return p;
}

TracerArrays<ko::MachineTraits>::Ptr init_tracer_arrays (Int nelemd, Int nlev, Int np2, Int qsize) {
  auto& p = get_instance();
  if (p == nullptr)
    p = std::make_shared<TracerArrays<ko::MachineTraits> >(nelemd, nlev, np2, qsize);
  return p;
}

TracerArrays<ko::MachineTraits>::Ptr get_tracer_arrays () {
  return get_instance();
}

void delete_tracer_arrays () {
  auto& p = get_instance();
  p = nullptr;
}

template struct TracerArrays<ko::MachineTraits>;
template void sl_h2d(const TracerArrays<ko::MachineTraits>& ta, Cartesian3D* dep_points);
template void sl_d2h(const TracerArrays<ko::MachineTraits>& ta, Cartesian3D* dep_points,
                     Real* minq, Real* maxq);

} // namespace homme
