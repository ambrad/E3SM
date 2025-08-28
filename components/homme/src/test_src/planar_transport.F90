#ifndef CAM
#include "config.h"

module planar_transport_tests
  use kinds, only: rl=>real_kind, iulog
  use element_mod, only: element_t
  use parallel_mod, only: parallel_t, abortmp
  use hybrid_mod, only: hybrid_t
  use hybvcoord_mod, only: hvcoord_t, set_layer_locations
  use derivative_mod, only: derivative_t
  use element_ops, only: set_state, set_state_i
  ! Planar geometry parameters.
  use physical_constants, only: Lx, Ly, dd_pi, &
       &                        Rgas, g, cp, dd_pi, p0
  use dimensions_mod, only: ne_x, ne_y, qsize, qsize_d, nlev, nlevp, np
  ! Test problem tools.
  use dcmip12_wrapper, only: get_evenly_spaced_z, set_hybrid_coefficients, &
       &                     pressure_thickness

  implicit none
  private

  public :: test_conv_planar_advection, print_conv_planar_advection_results

  real(rl), parameter :: &
       tau     = 12.d0 * 86400.d0, & ! period of motion 12 days
       T0      = 300.d0,           & ! temperature (K)
       ztop    = 12000.d0,         & ! model top (m)
       H       = Rgas * T0 / g       ! scale height

  character :: tc_major, tc_minor
  real(rl) :: zi(nlevp), zm(nlev)

contains  

  subroutine test_conv_planar_advection( &
       test_case, elem, hybrid, hvcoord, deriv, nets, nete, time, n0, n1)
    character(len=*), intent(in):: test_case
    type (element_t), intent(inout), target :: elem(:)
    type (hybrid_t), intent(in):: hybrid
    type (hvcoord_t), intent(inout) :: hvcoord
    type (derivative_t), intent(in):: deriv
    integer, intent(in):: nets, nete, n0, n1
    real(rl), intent(in):: time

    integer :: ie, k, j, i, qi
    real(rl) :: x, y, u, v, w, T, ps, phis, p, dp, z, q(qsize)

    if (time <= 0.d0) then
       call init(test_case, hybrid, hvcoord)
    end if

    do ie = nets,nete
       do k = 1,nlevp
          do j = 1,np
             do i = 1,np
                x = elem(ie)%spherep(i,j)%lon
                y = elem(ie)%spherep(i,j)%lat
                if (k < nlevp) then
                   call get_values(hvcoord, k, .true., time, x, y, &
                        &          ps, phis, p, z, T, u, v, w, q)
                   dp = pressure_thickness(ps,k,hvcoord)
                   if (time <= 0.d0) then
                      do qi = 1, qsize
                         elem(ie)%state%Q(i,j,k,qi) = q(qi)
                         elem(ie)%state%Qdp(i,j,k,qi,:) = elem(ie)%state%Q(i,j,k,qi) * dp
                      end do
                   end if
                   call set_state(u, v, w, T, ps, phis, p, dp, z, g, i, j, k, elem(ie), n0, n1)
                end if
                call get_values(hvcoord, k, .false., time, x, y, &
                     &          ps, phis, p, z, T, u, v, w, q)
                call set_state_i(u, v, w, T, ps, phis, p, z, g, i, j, k, elem(ie), n0, n1)
             end do
          end do
       end do
    end do
  end subroutine test_conv_planar_advection

  subroutine get_values(hvcoord, lev, mid, time, x, y, &
       &                ps, phis, p, z, T, u, v, w, q)
    type (hvcoord_t), intent(inout) :: hvcoord
    integer, intent(in) :: lev
    logical, intent(in) :: mid
    real(rl), intent(in) :: time, x, y
    real(rl), intent(out) :: ps, phis, p, z, T, u, v, w, q(qsize)

    real(rl), parameter :: ztop_t = 2000.d0

    real(rl) :: zs, ztaper

    zs = 2000.d0*(1 + sin(4*dd_pi*x/Lx))
    zs = -zs
    phis = g*zs
    ps = p0 * exp(-zs/H)

    if (mid) then
       p = hvcoord%hyam(lev)*p0 + hvcoord%hybm(lev)*ps
    else
       p = hvcoord%hyai(lev)*p0 + hvcoord%hybi(lev)*ps
    end if
    z = H * log(p0/p)

    if (z <= 0) then
       ztaper = 0
    elseif (z >= ztop_t) then
       ztaper = 1
    else
       ztaper = (1 + cos(dd_pi*(1 + z/ztop_t)))/2
    end if

    u = (Lx/tau)*ztaper
    v = 0; w = 0; T = T0

    q = 0
    if (z >= 6000.d0 .and. z <= 7000.d0) q = 1
  end subroutine get_values
  
  subroutine print_conv_planar_advection_results(test_case, elem, tl, hvcoord, par)
    use time_mod, only: timelevel_t
    use parallel_mod, only: parallel_t

    character(len=*), intent(in) :: test_case
    type(element_t), intent(in) :: elem(:)
    type(timelevel_t), intent(in) :: tl
    type(hvcoord_t), intent(in) :: hvcoord
    type(parallel_t), intent(in) :: par

  end subroutine print_conv_planar_advection_results

  subroutine init(test_case, hybrid, hvcoord)
    character(len=*), intent(in):: test_case
    type (hybrid_t), intent(in):: hybrid
    type (hvcoord_t), intent(inout) :: hvcoord

    !$omp barrier
    !$omp master
    ! Major and minor test case codes.
    tc_major = test_case(17:17)
    tc_minor = test_case(18:18) ! currently unused
    ! Vertical dimension.
    call get_evenly_spaced_z(zi, zm, 0.d0, ztop)
    hvcoord%etai = exp(-zi/H)
    call set_hybrid_coefficients(hvcoord, hybrid, hvcoord%etai(1), 1.d0)
    call set_layer_locations(hvcoord, .true., hybrid%masterthread)
    !$omp end master
    !$omp barrier
  end subroutine init
  
end module planar_transport_tests

#endif
