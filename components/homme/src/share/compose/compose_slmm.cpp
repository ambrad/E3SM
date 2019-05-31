#include "compose_slmm.hpp"
#include "compose_slmm_siqk.hpp"
#include "compose_slmm_advecter.hpp"
#include "compose_slmm_homme.hpp"
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

IslMpi::Ptr init (const slmm::Advecter::ConstPtr& advecter,
                  const mpi::Parallel::Ptr& p,
                  Int np, Int nlev, Int qsize, Int qsized, Int nelemd,
                  const Int* nbr_id_rank, const Int* nirptr,
                  Int halo) {
  slmm_throw_if(halo < 1 || halo > 2, "halo must be 1 (default) or 2.");
  auto cm = std::make_shared<IslMpi>(p, advecter, np, nlev, qsize, qsized,
                                     nelemd, halo);
  setup_comm_pattern(*cm, nbr_id_rank, nirptr);
  return cm;
}

// For const clarity, take the non-const advecter as an arg, even though cm
// already has a ref to the const'ed one.
void finalize_local_meshes (IslMpi& cm, slmm::Advecter& advecter) {
  if (cm.halo == 2) extend_halo::extend_local_meshes(*cm.p, cm.ed, advecter);
}

// Set pointers to HOMME data arrays.
void set_elem_data (IslMpi& cm, const Int ie, const Real* metdet, const Real* qdp,
                    const Real* dp, Real* q, const Int nelem_in_patch) {
  slmm_assert(ie < cm.ed.size());
  slmm_assert(cm.halo > 1 || cm.ed(ie).nbrs.size() == nelem_in_patch);
  auto& e = cm.ed(ie);
  e.metdet = metdet;
  e.qdp = qdp;
  e.dp = dp;
  e.q = q;
}
} // namespace islmpi

static slmm::Advecter::Ptr g_advecter;

void slmm_init (const Int np, const Int nelem, const Int nelemd,
                const Int transport_alg, const Int cubed_sphere_map,
                const Int sl_nearest_point_lev, const Int* lid2facenum) {
  g_advecter = std::make_shared<slmm::Advecter>(
    np, nelemd, transport_alg, cubed_sphere_map, sl_nearest_point_lev);
  g_advecter->init_meta_data(nelem, lid2facenum);
}
} // namespace homme

// Valid after slmm_init_local_mesh_ is called.
int slmm_unittest () {
  int nerr = 0, ne;
  {
    ne = 0;
    for (int i = 0; i < homme::g_advecter->nelem(); ++i) {
      const auto& m = homme::g_advecter->local_mesh(i);
      ne += slmm::unittest(m, m.tgt_elem);
    }
    if (ne)
      fprintf(stderr, "slmm_unittest: slmm::unittest returned %d\n", ne);
    nerr += ne;
  }
  return nerr;
}

#include <cstdlib>

static homme::islmpi::IslMpi::Ptr g_csl_mpi;

extern "C" {
void slmm_init_impl (
  homme::Int fcomm, homme::Int transport_alg, homme::Int np,
  homme::Int nlev, homme::Int qsize, homme::Int qsized, homme::Int nelem,
  homme::Int nelemd, homme::Int cubed_sphere_map,
  const homme::Int** lid2gid, const homme::Int** lid2facenum,
  const homme::Int** nbr_id_rank, const homme::Int** nirptr,
  homme::Int sl_nearest_point_lev)
{
  homme::slmm_init(np, nelem, nelemd, transport_alg, cubed_sphere_map,
                   sl_nearest_point_lev - 1, *lid2facenum);
  slmm_throw_if(homme::g_advecter->is_cisl(),
                "CISL code was removed.");
  const auto p = homme::mpi::make_parallel(MPI_Comm_f2c(fcomm));
  g_csl_mpi = homme::islmpi::init(homme::g_advecter, p, np, nlev, qsize,
                                  qsized, nelemd, *nbr_id_rank, *nirptr,
                                  2 /* halo */);
}

void slmm_get_mpi_pattern (homme::Int* sl_mpi) {
  *sl_mpi = g_csl_mpi ? 1 : 0;
}

void slmm_init_local_mesh (
  homme::Int ie, homme::Cartesian3D** neigh_corners, homme::Int nnc,
  homme::Cartesian3D* p_inside)
{
  homme::g_advecter->init_local_mesh_if_needed(
    ie - 1, homme::FA3<const homme::Real>(
      reinterpret_cast<const homme::Real*>(*neigh_corners), 3, 4, nnc),
    reinterpret_cast<const homme::Real*>(p_inside));
}

void slmm_init_finalize () {
  if (g_csl_mpi)
    homme::islmpi::finalize_local_meshes(*g_csl_mpi, *homme::g_advecter);
}

void slmm_check_ref2sphere (homme::Int ie, homme::Cartesian3D* p) {
  homme::g_advecter->check_ref2sphere(
    ie - 1, reinterpret_cast<const homme::Real*>(p));
}

void slmm_csl_set_elem_data (
  homme::Int ie, homme::Real* metdet, homme::Real* qdp, homme::Real* dp,
  homme::Real* q, homme::Int nelem_in_patch)
{
  slmm_assert(g_csl_mpi);
  homme::islmpi::set_elem_data(*g_csl_mpi, ie - 1, metdet, qdp, dp, q,
                               nelem_in_patch);
}

void slmm_csl (
  homme::Int nets, homme::Int nete, homme::Cartesian3D* dep_points,
  homme::Real* minq, homme::Real* maxq, homme::Int* info)
{
  slmm_assert(g_csl_mpi);
  *info = 0;
  try {
    homme::islmpi::step(*g_csl_mpi, nets - 1, nete - 1, dep_points, minq, maxq);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    *info = -1;
  }
}
} // extern "C"
