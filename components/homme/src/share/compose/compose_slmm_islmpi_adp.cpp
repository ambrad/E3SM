#include "compose_slmm_islmpi.hpp"
#include "compose_slmm_departure_point.hpp"

#ifdef COMPOSE_MIMIC_GPU
# pragma message "COMPOSE_MIMIC_GPU"
#endif
#ifdef COMPOSE_HORIZ_OPENMP
# pragma message "COMPOSE_HORIZ_OPENMP"
#endif
#ifdef COMPOSE_COLUMN_OPENMP
# pragma message "COMPOSE_COLUMN_OPENMP"
#endif
#ifdef COMPOSE_PORT
# pragma message "COMPOSE_PORT"
#endif
/*
  x get COMPOSE_PORT stuff going
  x get other containers in cm portable
  x make ListOfLists::init and BufferLayoutArray::init portable
  - kernelize analyze_dep_points, including use of atomics
  - pull qdp, dp, q out of ElemData; use idx() routines for them
*/

namespace slmm {
template <typename ES>
int get_nearest_point (const LocalMesh<ES>& m,
                       Real* v, const Int my_ic) {
  nearest_point::calc(m, v);
  return get_src_cell(m, v, my_ic);
}
} // namespace slmm

namespace homme {
namespace islmpi {

template <typename MT>
SLMM_KF void throw_on_sci_error (
  const IslMpi<MT>& cm, const FA4<Real>& dep_points, Int k, Int lev, Int tci,
  typename std::enable_if< ! slmm::OnGpu<typename MT::DES>::value>::type* = 0)
{
  const auto& mesh = cm.advecter->local_mesh(tci);
  const auto tgt_idx = mesh.tgt_elem;
  auto& ed = cm.ed_d(tci);
  std::stringstream ss;
  ss.precision(17);
  const auto* v = &dep_points(0,k,lev,tci);
  ss << "Departure point is outside of halo:\n"
     << "  nearest point permitted: "
     << cm.advecter->nearest_point_permitted(lev)
     << "\n  elem LID " << tci
     << " elem GID " << ed.me->gid
     << " (lev, k) (" << lev << ", " << k << ")"
     << " v " << v[0] << " " << v[1] << " " << v[2]
     << "\n  tgt_idx " << tgt_idx
     << " local mesh:\n  " << slmm::to_string(mesh) << "\n";
  slmm_throw_if(true, ss.str());
}

template <typename MT>
SLMM_KF void throw_on_sci_error (
  const IslMpi<MT>& cm, const FA4<Real>& dep_points, Int k, Int lev, Int tci,
  typename std::enable_if<slmm::OnGpu<typename MT::DES>::value>::type* = 0)
{
  ko::abort("throw_on_sci_error");
}

// Find where each departure point is.
template <typename MT>
void analyze_dep_points (IslMpi<MT>& cm, const Int& nets, const Int& nete,
                         const FA4<Real>& dep_points) {
  const auto myrank = cm.p->rank();
  const Int nrmtrank = static_cast<Int>(cm.ranks.size()) - 1;
  cm.bla.zero();
  cm.nx_in_lid.zero();
  {
    auto ed = cm.ed_d;
    ko::parallel_for(ko::RangePolicy<typename MT::DES>(nets, nete+1),
                     KOKKOS_LAMBDA (const Int& tci) { ed(tci).own.clear(); });
  }
  {
    const Int np2 = cm.np2, nlev = cm.nlev;
    const Int nearest_point_permitted_lev_bdy =
      cm.advecter->nearest_point_permitted_lev_bdy();
    const auto local_meshes = cm.advecter->local_meshes();
    const auto eds = cm.ed_d;
    const auto nx_in_lid = cm.nx_in_lid;
    const auto bla = cm.bla;
    // N.B. Currently a race condition in COMPOSE_PORT build.
    const auto f = KOKKOS_LAMBDA (const Int& ki) {
      const Int tci = nets + ki/(nlev*np2);
      const Int lev = (ki/np2) % nlev;
      const Int k = ki % np2;
      const auto& mesh = local_meshes(tci);
      const auto tgt_idx = mesh.tgt_elem;
      auto& ed = eds(tci);
      Int sci = slmm::get_src_cell(mesh, &dep_points(0,k,lev,tci), tgt_idx);
      if (sci == -1 &&
          slmm::Advecter<MT>::nearest_point_permitted(
            nearest_point_permitted_lev_bdy, lev))
        sci = slmm::get_nearest_point(mesh, &dep_points(0,k,lev,tci), tgt_idx);
      if (sci == -1)
        throw_on_sci_error(cm, dep_points, k, lev, tci);
      ed.src(lev,k) = sci;
      if (ed.nbrs(sci).rank == myrank) {
        auto& t = ed.own.atomic_inc_and_return_next();
        t.lev = lev;
        t.k = k;
      } else {
        const auto ri = ed.nbrs(sci).rank_idx;
        const auto lidi = ed.nbrs(sci).lid_on_rank_idx;
#ifdef COMPOSE_HORIZ_OPENMP
        omp_lock_t* lock;
        if (cm.horiz_openmp) {
          lock = &cm.ri_lidi_locks(ri,lidi);
          omp_set_lock(lock);
        }
#endif
        {
#ifdef COMPOSE_PORT
          ko::atomic_increment(static_cast<volatile Int*>(&nx_in_lid(ri,lidi)));
          ko::atomic_increment(static_cast<volatile Int*>(&bla(ri,lidi,lev).xptr));
#else
          ++nx_in_lid(ri,lidi);
          ++bla(ri,lidi,lev).xptr;
#endif
        }
#ifdef COMPOSE_HORIZ_OPENMP
        if (cm.horiz_openmp) omp_unset_lock(lock);
#endif
      }
    };
    ko::parallel_for(
      ko::RangePolicy<typename MT::DES>(0, (nete - nets +1)*nlev*np2), f);
  }
#ifdef COMPOSE_HORIZ_OPENMP
# pragma omp barrier
# pragma omp for
#endif
  for (Int ri = 0; ri < nrmtrank; ++ri) {
    auto& nx_in_rank = cm.nx_in_rank(ri);
    nx_in_rank = 0;
    for (Int i = 0, n = cm.lid_on_rank(ri).n(); i < n; ++i)
      nx_in_rank += cm.nx_in_lid(ri,i);
  }
}

template void analyze_dep_points(IslMpi<slmm::MachineTraits>& cm, const Int& nets,
                                 const Int& nete, const FA4<Real>& dep_points);

} // namespace islmpi
} // namespace homme
