#include "compose_slmm.hpp"
#include "compose_slmm_siqk.hpp"
#include "compose_slmm_advecter.hpp"
#include "compose.hpp"
#include "compose_slmm_islmpi.hpp"

#include <sys/time.h>
#include <mpi.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>

namespace homme {
namespace islmpi {

#ifdef COMPOSE_PORT
template <typename MT, typename ESD, typename ESS>
void deep_copy (typename IslMpi<MT>::template ElemData<ESD>& d,
                const typename IslMpi<MT>::template ElemData<ESS>& s) {
  d.nbrs.copy(s.nbrs);
  const ptrdiff_t me_os = s.me - s.nbrs.data();
  d.me = d.nbrs.data() + me_os;
  d.nin1halo = s.nin1halo;
  d.own.copy(s.own);
  d.rmt.copy(s.rmt);
  siqk::resize_and_copy(d.src, s.src);
  d.q_extrema = typename IslMpi<MT>::template Array<Real**[2], ESD>(
    "q_extrema", s.q_extrema.extent_int(0), s.q_extrema.extent_int(1));
  ko::deep_copy(d.q_extrema, s.q_extrema);
  d.qdp = s.qdp;
  d.dp = s.dp;
  d.q = s.q;
}
#endif

template <typename MT>
void deep_copy (typename IslMpi<MT>::ElemDataListD& d,
                const typename IslMpi<MT>::ElemDataListH& s) {
#ifdef COMPOSE_PORT
  const Int ned = s.size();
  // device view of device views
  d = typename IslMpi<MT>::ElemDataListD(ned);
  // host view of device views
  auto m = d.mirror();
  m.inc(ned);
  for (Int i = 0; i < ned; ++i)
    deep_copy<MT>(m(i), s(i));
  deep_copy(d, m);
#endif
}

template <typename MT> slmm::EnableIfDiffSpace<MT>
sync_to_device (IslMpi<MT>& cm) { deep_copy<MT>(cm.ed_d, cm.ed_h); }

template <typename MT> slmm::EnableIfSameSpace<MT>
sync_to_device (IslMpi<MT>& cm) { cm.ed_d = cm.ed_h; }

template <typename MT>
typename IslMpi<MT>::Ptr
init (const typename IslMpi<MT>::Advecter::ConstPtr& advecter,
      const mpi::Parallel::Ptr& p,
      Int np, Int nlev, Int qsize, Int qsized, Int nelemd,
      const Int* nbr_id_rank, const Int* nirptr,
      Int halo) {
  slmm_throw_if(halo < 1 || halo > 2, "halo must be 1 (default) or 2.");
  auto cm = std::make_shared<IslMpi<MT> >(p, advecter, np, nlev, qsize, qsized,
                                          nelemd, halo);
  setup_comm_pattern(*cm, nbr_id_rank, nirptr);
  return cm;
}

// For const clarity, take the non-const advecter as an arg, even though cm
// already has a ref to the const'ed one.
template <typename MT>
void finalize_init_phase (IslMpi<MT>& cm, typename IslMpi<MT>::Advecter& advecter) {
  if (cm.halo == 2)
    extend_halo::extend_local_meshes<MT>(*cm.p, cm.ed_h, advecter);
  advecter.fill_nearest_points_if_needed();
  advecter.sync_to_device();
  sync_to_device(cm);
}

// Set pointers to HOMME data arrays.
template <typename MT>
void set_elem_data (IslMpi<MT>& cm, const Int ie, const Real* qdp,
                    const Real* dp, Real* q, const Int nelem_in_patch) {
  slmm_assert(ie < cm.ed_h.size());
  slmm_assert(cm.halo > 1 || cm.ed_h(ie).nbrs.size() == nelem_in_patch);
  auto& e = cm.ed_h(ie);
#if defined COMPOSE_PORT_DEV
  cm.tracer_arrays.pqdp.set_ie_ptr(ie, qdp);
  cm.tracer_arrays.pdp.set_ie_ptr(ie, dp);
  cm.tracer_arrays.pq.set_ie_ptr(ie, q);
  e.qdp = e.dp = e.q = nullptr;
#else
  e.qdp = qdp;
  e.dp = dp;
  e.q = q;
#endif
}

template <typename MT>
void h2d (const TracerArrays<MT>& ta) {
#if defined COMPOSE_PORT_DEV_VIEWS
  const auto qdp_m = ko::create_mirror_view(ta.qdp);
  const auto dp_m = ko::create_mirror_view(ta.dp);
  const auto q_m = ko::create_mirror_view(ta.q);
  for (Int ie = 0; ie < q_m.extent_int(0); ++ie)
    for (Int iq = 0; iq < q_m.extent_int(1); ++iq)
      for (Int k = 0; k < q_m.extent_int(2); ++k)
        for (Int lev = 0; lev < q_m.extent_int(3); ++lev) {
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
#endif
}

template <typename MT>
void d2h (const TracerArrays<MT>& ta) {
#if defined COMPOSE_PORT_DEV_VIEWS
  const auto q_m = ko::create_mirror_view(ta.q);
  ko::deep_copy(q_m, ta.q);
  for (Int ie = 0; ie < q_m.extent_int(0); ++ie)
    for (Int iq = 0; iq < q_m.extent_int(1); ++iq)
      for (Int k = 0; k < q_m.extent_int(2); ++k)
        for (Int lev = 0; lev < q_m.extent_int(3); ++lev)
          ta.pq(ie,iq,k,lev) = q_m(ie,iq,k,lev);
#endif  
}
} // namespace islmpi

typedef slmm::MachineTraits HommeMachineTraits;
typedef islmpi::IslMpi<HommeMachineTraits> HommeIslMpi;

static HommeIslMpi::Advecter::Ptr g_advecter;

void slmm_init (const Int np, const Int nelem, const Int nelemd,
                const Int transport_alg, const Int cubed_sphere_map,
                const Int sl_nearest_point_lev, const Int* lid2facenum) {
  g_advecter = std::make_shared<HommeIslMpi::Advecter>(
    np, nelemd, transport_alg, cubed_sphere_map, sl_nearest_point_lev);
  g_advecter->init_meta_data(nelem, lid2facenum);
}
} // namespace homme

namespace amb {
template <typename T> T strto(const char* s);
template <> inline int strto (const char* s) { return std::atoi(s); }
template <> inline bool strto (const char* s) { return std::atoi(s); }
template <> inline double strto (const char* s) { return std::atof(s); }
template <> inline std::string strto (const char* s) { return std::string(s); }

template <typename T>
bool getenv (const std::string& varname, T& var) {
  const char* var_s = std::getenv(varname.c_str());
  if ( ! var_s) return false;
  var = strto<T>(var_s);
  return true;
}

void dev_init_threads () {
#if defined COMPOSE_MIMIC_GPU
  static int nthr = -1;
  slmm_assert(omp_get_thread_num() == 0);
  if (nthr < 0) {
    nthr = 1;
    getenv("OMP_NUM_THREADS", nthr);
  }
  omp_set_num_threads(nthr);
  static_assert(std::is_same<slmm::MachineTraits::DES, Kokkos::OpenMP>::value,
                "in this dev code, should have OpenMP exe space on");
#endif
}

void dev_fin_threads () {
#if defined COMPOSE_MIMIC_GPU
  omp_set_num_threads(1);
#endif
}
} // namespace amb

// Valid after slmm_init_local_mesh_ is called.
int slmm_unittest () {
  amb::dev_init_threads();
  int nerr = 0, ne;
  {
    ne = 0;
    for (int i = 0; i < homme::g_advecter->nelem(); ++i) {
      auto& m = homme::g_advecter->local_mesh_host(i);
      ne += slmm::unittest(m, m.tgt_elem);
    }
    if (ne)
      fprintf(stderr, "slmm_unittest: slmm::unittest returned %d\n", ne);
    nerr += ne;
  }
  amb::dev_fin_threads();
  return nerr;
}

#include <cstdlib>

static homme::HommeIslMpi::Ptr g_csl_mpi;

extern "C" {
// Interface for Homme, through compose_mod.F90.
void kokkos_init () {
  amb::dev_init_threads();
  Kokkos::InitArguments args;
  args.disable_warnings = true;
  Kokkos::initialize(args);
  // Test these initialize correctly.
  Kokkos::View<int> v("hi");
  Kokkos::deep_copy(v, 0);
  homme::islmpi::FixedCapList<int,slmm::MachineTraits::DES> fcl, fcl1(2);
  amb::dev_fin_threads();
}

void kokkos_finalize () {
  amb::dev_init_threads();
  Kokkos::finalize();
  amb::dev_fin_threads();
}

void slmm_init_impl (
  homme::Int fcomm, homme::Int transport_alg, homme::Int np,
  homme::Int nlev, homme::Int qsize, homme::Int qsized, homme::Int nelem,
  homme::Int nelemd, homme::Int cubed_sphere_map,
  const homme::Int* lid2gid, const homme::Int* lid2facenum,
  const homme::Int* nbr_id_rank, const homme::Int* nirptr,
  homme::Int sl_nearest_point_lev, homme::Int, homme::Int, homme::Int,
  homme::Int)
{
  amb::dev_init_threads();
  homme::slmm_init(np, nelem, nelemd, transport_alg, cubed_sphere_map,
                   sl_nearest_point_lev - 1, lid2facenum);
  slmm_throw_if(homme::g_advecter->is_cisl(), "CISL code was removed.");
  const auto p = homme::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  g_csl_mpi = homme::islmpi::init<homme::HommeMachineTraits>(
    homme::g_advecter, p, np, nlev, qsize, qsized, nelemd,
    nbr_id_rank, nirptr, 2 /* halo */);
  amb::dev_fin_threads();
}

void slmm_query_bufsz (homme::Int* sendsz, homme::Int* recvsz) {
  slmm_assert(g_csl_mpi);
  homme::Int s = 0, r = 0;
  for (const auto e : g_csl_mpi->sendsz) s += e;
  for (const auto e : g_csl_mpi->recvsz) r += e;
  *sendsz = s;
  *recvsz = r;
}

void slmm_set_bufs (homme::Real* sendbuf, homme::Real* recvbuf,
                    homme::Int, homme::Int) {
  amb::dev_init_threads();
  slmm_assert(g_csl_mpi);
  homme::islmpi::alloc_mpi_buffers(*g_csl_mpi, sendbuf, recvbuf);
  amb::dev_fin_threads();
}

void slmm_get_mpi_pattern (homme::Int* sl_mpi) {
  *sl_mpi = g_csl_mpi ? 1 : 0;
}

void slmm_init_local_mesh (
  homme::Int ie, homme::Cartesian3D* neigh_corners, homme::Int nnc,
  homme::Cartesian3D* p_inside, homme::Int)
{
  amb::dev_init_threads();
  homme::g_advecter->init_local_mesh_if_needed(
    ie - 1, homme::FA3<const homme::Real>(
      reinterpret_cast<const homme::Real*>(neigh_corners), 3, 4, nnc),
    reinterpret_cast<const homme::Real*>(p_inside));
  amb::dev_fin_threads();
}

void slmm_init_finalize () {
  amb::dev_init_threads();
  if (g_csl_mpi)
    homme::islmpi::finalize_init_phase(*g_csl_mpi, *homme::g_advecter);
  amb::dev_fin_threads();
}

void slmm_check_ref2sphere (homme::Int ie, homme::Cartesian3D* p) {
  amb::dev_init_threads();
  homme::g_advecter->check_ref2sphere(
    ie - 1, reinterpret_cast<const homme::Real*>(p));
  amb::dev_fin_threads();
}

void slmm_csl_set_elem_data (
  homme::Int ie, homme::Real* metdet, homme::Real* qdp, homme::Real* dp,
  homme::Real* q, homme::Int nelem_in_patch)
{
  amb::dev_init_threads();
  slmm_assert(g_csl_mpi);
  homme::islmpi::set_elem_data(*g_csl_mpi, ie - 1, qdp, dp, q,
                               nelem_in_patch);
  amb::dev_fin_threads();
}

void slmm_csl (
  homme::Int nets, homme::Int nete, homme::Cartesian3D* dep_points,
  homme::Real* minq, homme::Real* maxq, homme::Int* info)
{
  amb::dev_init_threads();
  slmm_assert(g_csl_mpi);
  slmm_assert(g_csl_mpi->sendsz.empty()); // alloc_mpi_buffers was called
  homme::islmpi::h2d(g_csl_mpi->tracer_arrays);
  *info = 0;
#if 0
#pragma message "RM TRY-CATCH WHILE DEV'ING"
  try {
    homme::islmpi::step(*g_csl_mpi, nets - 1, nete - 1, dep_points, minq, maxq);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    *info = -1;
  }
#else
  homme::islmpi::step(*g_csl_mpi, nets - 1, nete - 1, dep_points, minq, maxq);
#endif
  homme::islmpi::d2h(g_csl_mpi->tracer_arrays);
  amb::dev_fin_threads();
}

void slmm_finalize () {
  g_csl_mpi = nullptr;
  homme::g_advecter = nullptr;
}
} // extern "C"
