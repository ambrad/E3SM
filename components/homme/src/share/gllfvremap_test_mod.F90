#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!todo
! - global mass checks
! - global extrema checks
! - area correction: alpha
! - test vector_dp routines: conservation
! - topo roughness
! - ftype other than 2,4
! - np4-np2 instead of np4-pg1
! - halo exchange buffers
! - impl original pg2 to compare

module gllfvremap_test_mod
  ! Test gllfvremap's main API.

  use hybrid_mod, only: hybrid_t
  use kinds, only: real_kind
  use dimensions_mod, only: nelemd, np, nlev, nlevp, qsize
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

  type :: State_t
     real(kind=real_kind), dimension(np,np,nlev) :: u, v, w, T, p, dp, rho, z
     real(kind=real_kind), dimension(np,np,nlevp) :: zi, wi
     real(kind=real_kind), dimension(np,np) :: ps, phis
  end type State_t

  type (PhysgridData_t), private :: pg_data

  public :: gfr_check_api

contains
  
  subroutine init(nphys)
    integer, intent(in) :: nphys

    integer :: ncol

    ncol = nphys*nphys
    pg_data%nphys = nphys
    allocate(pg_data%ps(ncol,nelemd), pg_data%zs(ncol,nelemd), pg_data%T(ncol,nlev,nelemd), &
         pg_data%omega_p(ncol,nlev,nelemd), pg_data%uv(ncol,2,nlev,nelemd), &
         pg_data%q(ncol,nlev,qsize,nelemd))
  end subroutine init

  subroutine finish()
    deallocate(pg_data%ps, pg_data%zs, pg_data%T, pg_data%uv, pg_data%omega_p, pg_data%q)
  end subroutine finish

  subroutine set_state(s, hvcoord, nt1, nt2, ntq, elem)
    use physical_constants, only: g
    use element_ops, only: set_elem_state
    use hybvcoord_mod, only: hvcoord_t

    type (State_t), intent(in) :: s
    type (hvcoord_t), intent(in) :: hvcoord
    integer, intent(in) :: nt1, nt2, ntq
    type (element_t), intent(inout) :: elem

    call set_elem_state(s%u, s%v, s%w, s%wi, s%T, s%ps, s%phis, s%p, s%dp, s%z, s%zi, &
         g, elem, nt1, nt2, ntq)
  end subroutine set_state

  subroutine get_state(elem, hvcoord, nt, ntq, s)
    use physical_constants, only: g
    use element_ops, only: get_elem_state => get_state
    use hybvcoord_mod, only: hvcoord_t

    type (element_t), intent(inout) :: elem
    type (hvcoord_t), intent(in) :: hvcoord
    integer, intent(in) :: nt, ntq
    type (State_t), intent(out) :: s

    call get_elem_state(s%u, s%v, s%w, s%T, s%p, s%dp, s%ps, s%rho, s%z, s%zi, &
         g, elem, hvcoord, nt, ntq)
  end subroutine get_state

  subroutine set_gll_state(hvcoord, elem, nt1, nt2)
    use dimensions_mod, only: nlev, qsize
    use physical_constants, only: g
    use coordinate_systems_mod, only: cartesian3D_t, change_coordinates
    use hybvcoord_mod, only: hvcoord_t
    use element_ops, only: get_field

    type (hvcoord_t) , intent(in) :: hvcoord
    type (element_t), intent(inout) :: elem
    integer, intent(in) :: nt1, nt2

    type (State_t) :: s1
    type (cartesian3D_t) :: p
    real(kind=real_kind) :: wr(np,np,nlev,2)
    integer :: i, j, k, q, d, tl

    do j = 1,np
       do i = 1,np
          p = change_coordinates(elem%spherep(i,j))
          do k = 1,nlev
             do q = 1,qsize
                elem%state%Q(i,j,k,q) = 1 + &
                     0.5*sin((0.5 + modulo(q,2))*p%x)* &
                     sin((0.5 + modulo(q,3))*1.5*p%y)* &
                     sin((-2.3 + modulo(q,5))*2.5*p%z)
             end do
          end do
          s1%ps(i,j) = 1.0d3*(1 + 0.05*sin(2*p%x+0.5)*sin(p%y+1.5)*sin(3*p%z+2.5))
          s1%phis(i,j) = 1 + 0.5*sin(p%x-0.5)*sin(0.5*p%y+2.5)*sin(2*p%z-2.5)
          do k = 1,nlev
             do d = 1,2
                wr(i,j,k,d) = sin(0.5*d*p%x+d-0.5)*sin(1.5*p%y-d++2.5)*sin(d*p%z+d-2.5)
             end do
             elem%derived%omega_p(i,j,k) = wr(i,j,k,1)
          end do
          do k = 1,nlev
             s1%T(i,j,k) = 1 + 0.5*sin(p%x+1.5)*sin(1.5*p%y+0.5)*sin(2*p%z-0.5)
          end do
       end do
    end do
    s1%u = wr(:,:,:,1)
    s1%v = wr(:,:,:,2)
    do k = 1,nlev
       s1%p(:,:,k) = hvcoord%hyam(k)*hvcoord%ps0 + hvcoord%hybm(k)*s1%ps
       s1%dp(:,:,k) = (hvcoord%hyai(k+1) - hvcoord%hyai(k))*hvcoord%ps0 + &
            (hvcoord%hybi(k+1) - hvcoord%hybi(k))*s1%ps
    end do
    s1%z = zero
    s1%zi = zero
    ! a bit of a kludge
    call set_state(s1, hvcoord, nt1, nt2, nt1, elem)
    call get_field(elem, 'rho', wr(:,:,:,1), hvcoord, nt1, nt1)
    s1%w = -elem%derived%omega_p/(wr(:,:,:,1)*g)
    s1%wi(:,:,:nlev) = s1%w
    s1%wi(:,:,nlevp) = s1%w(:,:,nlev)
    call set_state(s1, hvcoord, nt1, nt2, nt1, elem)
    do q = 1,qsize
       do tl = nt1,nt2
          elem%state%Qdp(:,:,:,q,tl) = &
               elem%state%Q(:,:,:,q)*elem%state%dp3d(:,:,:,nt1)
       end do
    end do
  end subroutine set_gll_state

  subroutine run(hybrid, hvcoord, elem, nets, nete, nphys, tendency)
    use hybvcoord_mod, only: hvcoord_t
    use dimensions_mod, only: nlev, qsize
    use coordinate_systems_mod, only: cartesian3D_t, change_coordinates
    use element_ops, only: get_temperature
    use prim_driver_base, only: applyCAMforcing_tracers
    use prim_advance_mod, only: applyCAMforcing_dynamics
    use parallel_mod, only: global_shared_buf, global_shared_sum
    use global_norms_mod, only: wrap_repro_sum
    use reduction_mod, only: ParallelMin, ParallelMax
    use physical_constants, only: g
    use gllfvremap_mod

    type (hybrid_t), intent(in) :: hybrid
    type (hvcoord_t) , intent(in) :: hvcoord
    type (element_t), intent(inout) :: elem(:)
    integer, intent(in) :: nets, nete, nphys
    logical, intent(in) :: tendency
    character(32) :: msg

    type (cartesian3D_t) :: p
    real(kind=real_kind) :: wr(np,np,nlev), tend(np,np,nlev), f, a, b, rd, qmin1, qmax1, &
         qmin2, qmax2, mass1, mass2, wr1(np,np,nlev), wr2(np,np,nlev)
    integer :: nf, ncol, nt1, nt2, ie, i, j, k, d, q, qi, tl, col

    nf = nphys
    ncol = nf*nf
    nt1 = 1
    nt2 = 2
    
    ! Set analytical GLL values.
    do ie = nets,nete
       call set_gll_state(hvcoord, elem(ie), nt1, nt2)
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

    ! Apply the tendencies.
    do ie = nets,nete
       call applyCAMforcing_tracers(elem(ie), hvcoord, nt2, nt2, one, .true.)
    end do
    call applyCAMforcing_dynamics(elem, hvcoord, nt2, one, nets, nete)

    ! Test GLL state nt2 vs the original state nt1.
    if (hybrid%masterthread) print '(a,l2)', 'gfrt> tendency', tendency
    tend = zero
    mass1 = zero; mass2 = zero
    qmin1 = one; qmax1 = -one
    qmin2 = one; qmax2 = -one
    do q = 1, qsize+3
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
          if (q > qsize) then
             qi = q - qsize
             if (qi < 3) then
                global_shared_buf(ie,1) = &
                     sum(wr*( &
                     elem(ie)%state%v(:,:,qi,:,nt2) - &
                     (elem(ie)%state%v(:,:,qi,:,nt1) + tend))**2)
                global_shared_buf(ie,2) = &
                     sum(wr*(elem(ie)%state%v(:,:,qi,:,nt1) + tend)**2)
             else
                call get_temperature(elem(ie), wr1, hvcoord, nt1)
                call get_temperature(elem(ie), wr2, hvcoord, nt2)
                global_shared_buf(ie,1) = sum(wr*(wr2 - (wr1 + tend))**2)
                global_shared_buf(ie,2) = sum(wr*(wr1 + tend)**2)                
             end if
          else
             global_shared_buf(ie,1) = &
                  sum(wr*( &
                  elem(ie)%state%Q(:,:,:,q) - &
                  (elem(ie)%state%Qdp(:,:,:,q,nt1)/elem(ie)%state%dp3d(:,:,:,nt1) + tend))**2)
             global_shared_buf(ie,2) = &
                  sum(wr*( &
                  elem(ie)%state%Qdp(:,:,:,q,nt1)/elem(ie)%state%dp3d(:,:,:,nt1) + tend)**2)
          end if
       end do
       call wrap_repro_sum(nvars=2, comm=hybrid%par%comm)
       if (hybrid%masterthread) then
          rd = sqrt(global_shared_sum(1)/global_shared_sum(2))
          msg = ''
          if (.not. tendency .and. rd > 5*eps) msg = ' ERROR'
          print '(a,i3,a,i3,es12.4,a8)', 'gfrt> q l2', q, ' of', qsize, rd, msg
       end if
    end do

    ! Test topo routines.
    ! Stash initial phis for later comparison.
    do ie = nets,nete
       elem(ie)%derived%vstar(:,:,1,1) = elem(ie)%state%phis
    end do
    call gfr_dyn_to_fv_phys_topo(hybrid, elem, pg_data%zs, nets, nete)
    call gfr_fv_phys_to_dyn_topo(hybrid, elem, pg_data%zs, nets, nete)
    ! Compare GLL phis1 against GLL phis0.
    do ie = nets,nete
       global_shared_buf(ie,1) = sum(elem(ie)%spheremp*(elem(ie)%state%phis - &
            elem(ie)%derived%vstar(:,:,1,1))**2)
       global_shared_buf(ie,2) = sum(elem(ie)%spheremp*elem(ie)%derived%vstar(:,:,1,1)**2)
    end do
    call wrap_repro_sum(nvars=2, comm=hybrid%par%comm)
    if (hybrid%masterthread) then
       rd = sqrt(global_shared_sum(1)/global_shared_sum(2))
       print '(a,es12.4)', 'gfrt> topo l2', rd
    end if
  end subroutine run

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
          call gfr_init(hybrid, elem, nphys, check=.true.)
          call init(nphys)
       end if
       !$omp barrier

       call run(hybrid, hvcoord, elem, nets, nete, nphys, .false.)
       call run(hybrid, hvcoord, elem, nets, nete, nphys, .true.)

       ! This is meant to be called after threading ends.
       !$omp barrier
       if (hybrid%ithr == 0) then
          call gfr_finish()
          call finish()
       end if
       !$omp barrier
    end do
  end subroutine gfr_check_api
end module gllfvremap_test_mod
