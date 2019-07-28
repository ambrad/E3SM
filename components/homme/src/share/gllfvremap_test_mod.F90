#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

module gllfvremap_test_mod
  ! Test gllfvremap's main API.

  use hybrid_mod, only: hybrid_t
  use kinds, only: real_kind
  use dimensions_mod, only: np, qsize, nelemd
  use element_mod, only: element_t

  implicit none

  private

  real(kind=real_kind), parameter :: &
       zero = 0.0_real_kind, one = 1.0_real_kind, eps = epsilon(1.0_real_kind)

  type :: PhysgridData_t
     integer :: nphys
     real(kind=real_kind), allocatable :: ps(:,:), zs(:,:), T(:,:,:), uv(:,:,:,:), &
          omega_p(:,:,:), q(:,:,:,:)
  end type PhysgridData_t

  type (PhysgridData_t), private :: pg_data

  public :: gfr_check_api

contains
  
  subroutine check_init(nphys)
    use dimensions_mod, only: nelemd, nlev, qsize

    integer, intent(in) :: nphys

    integer :: ncol

    ncol = nphys*nphys
    pg_data%nphys = nphys
    allocate(pg_data%ps(ncol,nelemd), pg_data%zs(ncol,nelemd), pg_data%T(ncol,nlev,nelemd), &
         pg_data%omega_p(ncol,nlev,nelemd), pg_data%uv(ncol,2,nlev,nelemd), &
         pg_data%q(ncol,nlev,qsize,nelemd))
  end subroutine check_init

  subroutine check_finish()
    deallocate(pg_data%ps, pg_data%zs, pg_data%T, pg_data%uv, pg_data%omega_p, pg_data%q)
  end subroutine check_finish

  subroutine check_api(hybrid, hvcoord, elem, nets, nete, nphys, tendency)
    use hybvcoord_mod, only: hvcoord_t
    use dimensions_mod, only: nlev, qsize
    use element_ops, only: set_thermostate, get_temperature
    use coordinate_systems_mod, only: cartesian3D_t, change_coordinates
    use prim_driver_base, only: applyCAMforcing_tracers
    use prim_advance_mod, only: applyCAMforcing_dynamics
    use parallel_mod, only: global_shared_buf, global_shared_sum
    use global_norms_mod, only: wrap_repro_sum
    use reduction_mod, only: ParallelMin, ParallelMax
    use gllfvremap_mod

    type (hybrid_t), intent(in) :: hybrid
    type (hvcoord_t) , intent(in) :: hvcoord
    type (element_t), intent(inout) :: elem(:)
    integer, intent(in) :: nets, nete, nphys
    logical, intent(in) :: tendency

    real(kind=real_kind) :: wr(np,np,nlev), tend(np,np,nlev), f, a, b, rd
    integer :: nf, ncol, nt1, nt2, ie, i, j, k, d, q, tl, col
    type (cartesian3D_t) :: p

    nf = nphys
    ncol = nf*nf
    nt1 = 1
    nt2 = 2
    
    ! Set analytical GLL values. nt1 contains the original, true
    ! values for later use.
    do ie = nets,nete
       do j = 1,np
          do i = 1,np
             p = change_coordinates(elem(ie)%spherep(i,j))
             do k = 1,nlev
                do q = 1,qsize
                   elem(ie)%state%Q(i,j,k,q) = 1 + &
                        0.5*sin((0.5 + modulo(q,2))*p%x)* &
                        sin((0.5 + modulo(q,3))*1.5*p%y)* &
                        sin((-2.3 + modulo(q,5))*2.5*p%z)
                end do
             end do
             elem(ie)%state%ps_v(i,j,nt1:nt2) = &
                  1.0d3*(1 + 0.05*sin(2*p%x+0.5)*sin(p%y+1.5)*sin(3*p%z+2.5))
             elem(ie)%state%phis(i,j) = &
                  1 + 0.5*sin(p%x-0.5)*sin(0.5*p%y+2.5)*sin(2*p%z-2.5)
             do d = 1,2
                elem(ie)%state%v(i,j,d,k,nt1:nt2) = &
                     sin(0.5*d*p%x+d-0.5)*sin(1.5*p%y-d++2.5)*sin(d*p%z+d-2.5)
             end do
             ! Set omega_p to a v component so we know its true value later.
             elem(ie)%derived%omega_p(i,j,k) = elem(ie)%state%v(i,j,1,k,nt1)
             do k = 1,nlev
                wr(i,j,k) = &
                     1 + 0.5*sin(p%x+1.5)*sin(1.5*p%y+0.5)*sin(2*p%z-0.5)
             end do
          end do
       end do
       call set_thermostate(elem(ie), elem(ie)%state%ps_v(:,:,nt1), wr, hvcoord)
       do q = 1,qsize
          do tl = nt1,nt2
             elem(ie)%state%Qdp(:,:,:,q,tl) = &
                  elem(ie)%state%Q(:,:,:,q)*elem(ie)%state%dp3d(:,:,:,nt1)
          end do
       end do
    end do

    ! GLL -> FV.
    call gfr_dyn_to_fv_phys(hybrid, nt2, hvcoord, elem, pg_data%ps, pg_data%zs, pg_data%T, &
         pg_data%uv, pg_data%omega_p, pg_data%q, nets, nete)
    
    ! Set FV tendencies.
    if (tendency) then
       do ie = nets,nete
          do j = 1,nf
             do i = 1,nf
                col = nf*(j-1) + i
                call gfr_f_get_cartesian3d(ie, i, j, p)
                f = 0.25_real_kind*sin(3.2*p%x)*sin(4.2*p%y)*sin(2.3*p%z)
                pg_data%uv(col,:,:,ie) = f
                pg_data%T(col,:,ie) = f
                ! no moisture adjustment => no dp3d adjustment
                pg_data%q(col,:,2:qsize,ie) = pg_data%q(col,:,2:qsize,ie) + f
             end do
          end do
       end do
    else
       ! Test that if tendencies are 0, then the original fields are unchanged.
       pg_data%T = zero
       pg_data%uv = zero
    end if

    ! FV -> GLL.
    call gfr_fv_phys_to_dyn(hybrid, nt2, hvcoord, elem, pg_data%T, pg_data%uv, pg_data%q, &
         nets, nete)
    call gfr_f2g_dss(hybrid, elem, nets, nete)
    if (nf == 1) then
       call gfr_fv_phys_to_dyn_boost_pg1(hybrid, nt2, hvcoord, elem, pg_data%T, pg_data%uv, &
            pg_data%q, nets, nete)
       call gfr_f2g_dss(hybrid, elem, nets, nete)
    end if

    ! Apply the tendencies.
    do ie = nets,nete
       call applyCAMforcing_tracers(elem(ie), hvcoord, nt2, nt2, one, .true.)
    end do
    call applyCAMforcing_dynamics(elem, hvcoord, nt2, one, nets, nete)

    ! Test GLL state nt2 vs the original state nt1.
    if (hybrid%masterthread) print '(a,i3)', 'gfrt> tendency', tendency
    tend = zero
    do q = 1,qsize
       do ie = nets,nete
          do k = 1,nlev
             wr(:,:,k) = elem(ie)%spheremp
          end do
          if (tendency .and. q > 1) then
             do j = 1,np
                do i = 1,np
                   p = change_coordinates(elem(ie)%spherep(i,j))
                   tend(i,j,:) = 0.25_real_kind*sin(3.2*p%x)*sin(4.2*p%y)*sin(2.3*p%z)
                end do
             end do
          end if
          global_shared_buf(ie,1) = &
               sum(wr*( &
               elem(ie)%state%Q(:,:,:,q) - &
               (elem(ie)%state%Qdp(:,:,:,q,nt1)/elem(ie)%state%dp3d(:,:,:,nt1) + tend))**2)
          global_shared_buf(ie,2) = &
               sum(wr*( &
               elem(ie)%state%Qdp(:,:,:,q,nt1)/elem(ie)%state%dp3d(:,:,:,nt1) + tend)**2)
       end do
       call wrap_repro_sum(nvars=2, comm=hybrid%par%comm)
       if (hybrid%masterthread) then
          rd = sqrt(global_shared_sum(1)/global_shared_sum(2))
          print '(a,i3,es12.4)', 'gfrt> q l2', q, rd
       end if
    end do
    do ie = nets,nete
    end do
    call wrap_repro_sum(nvars=3, comm=hybrid%par%comm)
  end subroutine check_api

  subroutine gfr_check_api(hybrid, nets, nete, hvcoord, deriv, elem)
    use derivative_mod, only: derivative_t
    use hybvcoord_mod, only: hvcoord_t
    use gllfvremap_mod

    type (hybrid_t), intent(in) :: hybrid
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(inout) :: elem(:)
    type (hvcoord_t) , intent(in) :: hvcoord
    integer, intent(in) :: nets, nete

    integer :: nphys

    do nphys = 1, np
       ! This is meant to be called before threading starts.
       if (hybrid%ithr == 0) then
          call gfr_init(hybrid, elem, nphys)
          call check_init(nphys)
       end if
       !$omp barrier

       call check_api(hybrid, hvcoord, elem, nets, nete, nphys, .false.)
       call check_api(hybrid, hvcoord, elem, nets, nete, nphys, .true.)

       ! This is meant to be called after threading ends.
       !$omp barrier
       if (hybrid%ithr == 0) then
          call gfr_finish()
          call check_finish()
       end if
       !$omp barrier
    end do
  end subroutine gfr_check_api
end module gllfvremap_test_mod
