#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module compose_forcing_mod

implicit none

interface
   subroutine kokkos_init() bind(c)
   end subroutine kokkos_init

   subroutine kokkos_finalize() bind(c)
   end subroutine kokkos_finalize

   subroutine cedr_forcing_set_ie2gci(ie, gci) bind(c)
     integer, value, intent(in) :: ie, gci
   end subroutine cedr_forcing_set_ie2gci
   
   subroutine cedr_forcing_init(np, nlev, qsize, qsize_d, timelevels, &
        need_conservation) bind(c)
     integer, value, intent(in) :: np, nlev, qsize, qsize_d, timelevels, &
          need_conservation
   end subroutine cedr_forcing_init

   subroutine cedr_forcing_init_impl(comm, cdr_alg, sc2gci, sc2rank, &
        ncell, nlclcell, nlev) bind(c)
     integer, value, intent(in) :: comm, cdr_alg, ncell, nlclcell, nlev
     integer, intent(in) :: sc2gci(:), sc2rank(:)
   end subroutine cedr_forcing_init_impl

   subroutine cedr_unittest(comm, nerr) bind(c)
     integer, value, intent(in) :: comm
     integer, intent(out) :: nerr
   end subroutine cedr_unittest

   subroutine cedr_forcing_set_Qdp(ie, Qdp, n0_qdp, np1_qdp) bind(c)
     use kinds         , only : real_kind
     use dimensions_mod, only : nlev, np, qsize_d
     integer, value, intent(in) :: ie, n0_qdp, np1_qdp
     real(kind=real_kind), intent(in) :: Qdp(np,np,nlev,qsize_d,2)
   end subroutine cedr_forcing_set_Qdp

   subroutine cedr_forcing_run(minq, maxq, nets, nete) bind(c)
     use kinds         , only : real_kind
     use dimensions_mod, only : nlev, np, qsize
     real(kind=real_kind), intent(in) :: minq(np,np,nlev,qsize,nets:nete)
     real(kind=real_kind), intent(in) :: maxq(np,np,nlev,qsize,nets:nete)
     integer, value, intent(in) :: nets, nete
   end subroutine cedr_forcing_run

   subroutine cedr_forcing_run_local(minq, maxq, nets, nete, use_ir, limiter_option) bind(c)
     use kinds         , only : real_kind
     use dimensions_mod, only : nlev, np, qsize
     real(kind=real_kind), intent(in) :: minq(np,np,nlev,qsize,nets:nete)
     real(kind=real_kind), intent(in) :: maxq(np,np,nlev,qsize,nets:nete)
     integer, value, intent(in) :: nets, nete, use_ir, limiter_option
   end subroutine cedr_forcing_run_local

   subroutine cedr_forcing_check(minq, maxq, nets, nete) bind(c)
     use kinds         , only : real_kind
     use dimensions_mod, only : nlev, np, qsize
     real(kind=real_kind), intent(in) :: minq(np,np,nlev,qsize,nets:nete)
     real(kind=real_kind), intent(in) :: maxq(np,np,nlev,qsize,nets:nete)
     integer, value, intent(in) :: nets, nete
   end subroutine cedr_forcing_check
end interface

contains

  subroutine compose_forcing_init(par, elem, GridVertex)
    use parallel_mod, only: parallel_t, abortmp
    use dimensions_mod, only : np, nlev, qsize, qsize_d, nelem, nelemd
    use element_mod, only : element_t
    use gridgraph_mod, only : GridVertex_t
    use perf_mod, only: t_startf, t_stopf

    type (parallel_t), intent(in) :: par
    type (element_t), intent(in) :: elem(:)
    type (GridVertex_t), intent(in), target :: GridVertex(:)
  end subroutine compose_forcing_init

end module compose_forcing_mod
