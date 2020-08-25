#include "compose_hommexx.hpp"
#include "compose_homme.hpp"
#include "compose_slmm_islmpi.hpp"

namespace homme {

islmpi::IslMpi<>::Ptr get_isl_mpi_singleton();

void cedr_sl_run_global();
void cedr_sl_run_local(const int limiter_option);

void slmm_finalize();
void cedr_finalize();

namespace compose {

void set_views (const Kokkos::View<double***>& spheremp,
                const Kokkos::View<double****>& dp, const Kokkos::View<double*****>& dp3d,
                const Kokkos::View<double******>& qdp, const Kokkos::View<double*****>& q,
                const Kokkos::View<double*****>& dep_points) {
  using Kokkos::View;
  auto& ta = *get_tracer_arrays();
  const auto nel = spheremp.extent_int(0);
  const auto np2 = spheremp.extent_int(1)*spheremp.extent_int(1);
  const auto nlev = dp.extent_int(3);
  const auto qsize_d = qdp.extent_int(2);
  ta.spheremp = View<Real**>(spheremp.data(), nel, np2);
  ta.dp = View<Real***>(dp.data(), nel, np2, nlev);
  ta.dp3d = View<Real****>(dp3d.data(), nel, dp3d.extent_int(1), np2, nlev);
  ta.qdp = View<Real*****>(qdp.data(), nel, qdp.extent_int(1), qsize_d, np2, nlev);
  ta.q = View<Real****>(q.data(), nel, qsize_d, np2, nlev);
  ta.dep_points = View<Real***[3]>(dep_points.data(), nel, dep_points.extent_int(1), np2);
}

void advect (const int np1, const int n0_qdp, const int np1_qdp) {
  auto& cm = *get_isl_mpi_singleton();
  cm.tracer_arrays->np1 = np1;
  cm.tracer_arrays->n0_qdp = n0_qdp;
  cm.tracer_arrays->n1_qdp = np1_qdp;
  islmpi::step<>(cm, 0, cm.nelemd - 1, nullptr, nullptr, nullptr);
}

void property_preserve (const int limiter_option) {
  homme::cedr_sl_run_global();
  homme::cedr_sl_run_local(limiter_option);
}

void finalize () {
  slmm_finalize();
  cedr_finalize();
}

} // namespace compose
} // namespace homme
