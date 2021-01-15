#ifndef INCLUDE_COMPOSE_CEDR_CAAS_HPP
#define INCLUDE_COMPOSE_CEDR_CAAS_HPP

#include "cedr_caas.hpp"

namespace homme {
namespace compose {

template <typename ES>
struct CAAS : public cedr::caas::CAAS<ES> {
  typedef cedr::caas::CAAS<ES> Super;

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
