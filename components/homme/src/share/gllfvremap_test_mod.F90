#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!todo
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
       zero = 0.0_real_kind, half = 0.5_real_kind, &
       one = 1.0_real_kind, two = 2.0_real_kind, &
       eps = epsilon(1.0_real_kind)

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
                elem%state%Q(i,j,k,q) = one + &
                     half*sin((half + modulo(q,2))*p%x)* &
                     sin((half + modulo(q,3))*1.5d0*p%y)* &
                     sin((-2.3d0 + modulo(q,5))*p%z)
             end do
          end do
          s1%ps(i,j) = 1.0d3*(one + 0.05d0*sin(two*p%x+half)*sin(p%y+1.5d0)*sin(3*p%z+2.5d0))
          s1%phis(i,j) = one + half*sin(p%x-half)*sin(half*p%y+2.5d0)*sin(2*p%z-2.5d0)
          do k = 1,nlev
             do d = 1,2
                wr(i,j,k,d) = sin(half*d*p%x+d-half)*sin(1.5*p%y-d+2.5d0)*sin(d*p%z+d-2.5d0)
             end do
             elem%derived%omega_p(i,j,k) = wr(i,j,k,1)
          end do
          do k = 1,nlev
             s1%T(i,j,k) = one + half*sin(p%x+1.5d0)*sin(1.5d0*p%y+half)*sin(two*p%z-half)
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
    real(kind=real_kind) :: wr(np,np,nlev), tend(np,np,nlev), f, a, b, rd, &
         qmin1(qsize), qmax1(qsize), qmin2, qmax2, mass1, mass2, &
         wr1(np,np,nlev), wr2(np,np,nlev)
    integer :: nf, ncol, nt1, nt2, ie, i, j, k, d, q, qi, tl, col

    nf = nphys
    ncol = nf*nf
    nt1 = 1
    nt2 = 2

    !! Test 1.
    ! Test physgrid API.
    !   Test that if tendency is 0, then the original field is
    ! recovered with error eps ~ machine precision.
    !   OOA for pgN should min(N, 2). The OOA is limited to 2 because
    ! of the way the tendencies are created. Instead of setting the FV
    ! value to the average over the subcell, the sampled value at the
    ! cell midpoint (ref midpoint mapped to sphere) is used. This
    ! creates a 2nd-order error. Tests 2 and 3 test whether the remaps
    ! achieve design OOA, in particular OOA 3 for pg3 u, v, T, phis
    ! fields.

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
          print '(a,i3,a,i3,es12.4,a8)', 'gfrt> test1 q l2', q, ' of', qsize, rd, msg
       end if
    end do

    !! Test 2.
    ! Test topo routines. OOA should be N for pgN, N=1,3,4. Error
    ! should be eps for pg4; see Test 3 text for explanation.
  
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
       print '(a,es12.4)', 'gfrt> test2 topo l2', rd
    end if

    if (.not. tendency) return

    !! Test 3.
    ! Test FV fields that were not covered in the previous tests. This
    ! is done by copying them to look like tendencies.
    !   For pg4, u,v,T should have l2 errors that are eps because pg4
    ! reconstructions the fields exactly, even with DSS.
    !   For pg4, q should have l2 errors that converge at OOA >= 2. 2
    ! is the formal OOA b/c of the limiter. The limiter acts on the
    ! tendency, which in this case is exactly what is being examined.
    !   For pgN, N=1,2,3, u,v,T should have OOA N.
    !   For pgN, q should have OOA min(N,2).

    do ie = nets,nete
       call set_gll_state(hvcoord, elem(ie), nt1, nt2)
    end do
    call gfr_dyn_to_fv_phys(hybrid, nt2, hvcoord, elem, pg_data%ps, pg_data%zs, pg_data%T, &
         pg_data%uv, pg_data%omega_p, pg_data%q, nets, nete)
    ! Leave T, uv as they are. They will be mapped back as
    ! tendencies. Double Q so that this new value minus the
    ! original is Q.
    qmin1 = one; qmax1 = -one
    do ie = nets,nete
       pg_data%q(:ncol,:,:,ie) = two*pg_data%q(:ncol,:,:,ie)
       do q = 2,qsize
          qmin1(q) = min(qmin1(q), minval(elem(ie)%state%Q(:,:,1,q)))
          qmax1(q) = max(qmax1(q), maxval(elem(ie)%state%Q(:,:,1,q)))
          qmin1(q) = min(qmin1(q), minval(pg_data%q(:ncol,1,q,ie)))
          qmax1(q) = max(qmax1(q), maxval(pg_data%q(:ncol,1,q,ie)))
       end do
    end do
    call gfr_fv_phys_to_dyn(hybrid, nt2, hvcoord, elem, pg_data%T, pg_data%uv, pg_data%q, &
         nets, nete)
    call gfr_f2g_dss(hybrid, elem, nets, nete)
    ! Don't apply forcings; rather, the forcing fields now have the
    ! remapped quantities we want to compare against the original.
    do q = 2, qsize+3
       mass1 = zero; mass2 = zero
       qmin2 = one; qmax2 = -one
       do ie = nets,nete
          do k = 1,nlev
             wr(:,:,k) = elem(ie)%spheremp
          end do
          if (q > qsize) then
             qi = q - qsize
             if (qi < 3) then
                global_shared_buf(ie,3) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)* &
                     elem(ie)%derived%FM(:,:,qi,1))
                global_shared_buf(ie,4) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)* &
                     elem(ie)%state%v(:,:,qi,1,nt1))
                global_shared_buf(ie,1) = &
                     sum(wr*( &
                     elem(ie)%derived%FM(:,:,qi,:) - &
                     elem(ie)%state%v(:,:,qi,:,nt1))**2)
                global_shared_buf(ie,2) = &
                     sum(wr*elem(ie)%state%v(:,:,qi,:,nt1)**2)
             else
                call get_temperature(elem(ie), wr1, hvcoord, nt1)
                global_shared_buf(ie,3) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)* &
                     elem(ie)%derived%FT(:,:,1))
                global_shared_buf(ie,4) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)*wr1(:,:,1))
                global_shared_buf(ie,1) = sum(wr*(elem(ie)%derived%FT - wr1)**2)
                global_shared_buf(ie,2) = sum(wr*wr1**2)                
             end if
          else
             ! Extrema in level 1.
             qmin2 = min(qmin2, minval(elem(ie)%derived%FQ(:,:,1,q)))
             qmax2 = max(qmax2, maxval(elem(ie)%derived%FQ(:,:,1,q)))
             ! Mass in level 1.
             global_shared_buf(ie,3) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)* &
                  elem(ie)%derived%FQ(:,:,1,q))
             global_shared_buf(ie,4) = sum(wr(:,:,1)*elem(ie)%state%dp3d(:,:,1,nt1)* &
                  two*elem(ie)%state%Q(:,:,1,q))
             ! l2 error in volume.
             global_shared_buf(ie,1) = &
                  sum(wr*( &
                  elem(ie)%derived%FQ(:,:,:,q) - &
                  two*elem(ie)%state%Q(:,:,:,q))**2)
             global_shared_buf(ie,2) = &
                  sum(wr*elem(ie)%state%Q(:,:,:,q)**2)
          end if
       end do
       call wrap_repro_sum(nvars=4, comm=hybrid%par%comm)
       qmin1(q) = ParallelMin(qmin1(q), hybrid)
       qmax1(q) = ParallelMax(qmax1(q), hybrid)
       qmin2 = ParallelMin(qmin2, hybrid)
       qmax2 = ParallelMax(qmax2, hybrid)
       if (hybrid%masterthread) then
          rd = sqrt(global_shared_sum(1)/global_shared_sum(2))
          print '(a,i3,a,i3,es12.4)', 'gfrt> test3 q l2', q, ' of', qsize, rd
          b = max(abs(qmin1(q)), abs(qmax1(q)))
          if (q <= qsize .and. qmin2 < qmin1(q) - 5*eps*b .or. &
               qmax2 > qmax1(q) + 5*eps*b) then
             print '(a,i3,es12.4,es12.4,es12.4,es12.4)', 'gfrt> test3 q extrema', &
                  q, qmin1(q), qmin2-qmin1(q), qmax2-qmax1(q), qmax1(q)
          end if
          a = global_shared_sum(3)
          b = global_shared_sum(4)
          if (abs(b - a) > 5*eps*abs(a)) then
             print '(a,i3,es12.4,es12.4,es12.4)', 'gfrt> test3 q mass', &
                  q, a, b, abs(b - a)/abs(a)
          end if
       end if
    end do
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
          call gfr_init(hybrid%par, elem, nphys, check=.true.)
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
