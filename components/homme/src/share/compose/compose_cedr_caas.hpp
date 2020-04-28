#ifndef INCLUDE_COMPOSE_CEDR_CAAS_HPP
#define INCLUDE_COMPOSE_CEDR_CAAS_HPP

#include "cedr_caas.hpp"

namespace homme {
namespace compose {

// We explicitly use Kokkos::Serial here so we can run the Kokkos kernels in the
// super class w/o triggering an expecution-space initialization error in
// Kokkos. This complication results from the interaction of Homme's
// COMPOSE_HORIZ_OPENMP threading with Kokkos kernels.
struct CAAS : public cedr::caas::CAAS<Kokkos::Serial> {
  typedef cedr::caas::CAAS<Kokkos::Serial> Super;

  CAAS (const cedr::mpi::Parallel::Ptr& p, const cedr::Int nlclcells,
        const typename Super::UserAllReducer::Ptr& uar)
    : Super(p, nlclcells, uar)
  {}

  void run () override {
#if defined COMPOSE_HORIZ_OPENMP
#   pragma omp master
#endif
    {
      Super::run();
    }
  }
};

} // namespace compose
} // namespace homme

#endif
