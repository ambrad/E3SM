#ifndef CAM
#include "config.h"

module planar_transport_tests
  use kinds, only: rl=>real_kind, iulog
  use element_mod, only: element_t
  use parallel_mod, only: parallel_t, abortmp
  use hybrid_mod, only: hybrid_t
  use hybvcoord_mod, only: hvcoord_t, set_layer_locations
  use derivative_mod, only: derivative_t
  use physical_constants, only: Lx, Ly, Sx, Sy, dx, dy, dx_ref, dy_ref, &
       &                        rearth0, Rgas, g, cp, dd_pi, p0
  use dimensions_mod, only: ne_x, ne_y, qsize, qsize_d

  implicit none
  private

  public :: test_conv_planar_advection, print_conv_planar_advection_results

  character :: tc_major, tc_minor

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

    if (time <= 0.d0) then
       call init(test_case, elem, hybrid, hvcoord, deriv, nets, nete)
    end if
    
  end subroutine test_conv_planar_advection
  
  subroutine print_conv_planar_advection_results(test_case, elem, tl, hvcoord, par)
    use time_mod, only: timelevel_t
    use parallel_mod, only: parallel_t

    character(len=*), intent(in) :: test_case
    type(element_t), intent(in) :: elem(:)
    type(timelevel_t), intent(in) :: tl
    type(hvcoord_t), intent(in) :: hvcoord
    type(parallel_t), intent(in) :: par

  end subroutine print_conv_planar_advection_results

  subroutine init(test_case, elem, hybrid, hvcoord, deriv, nets, nete)
    character(len=*), intent(in):: test_case
    type (element_t), intent(inout), target :: elem(:)
    type (hybrid_t), intent(in):: hybrid
    type (hvcoord_t), intent(inout) :: hvcoord
    type (derivative_t), intent(in):: deriv
    integer, intent(in):: nets, nete

    !$omp barrier
    !$omp master
    if (hybrid%masterthread) then
       tc_major = test_case(17:17)
       tc_minor = test_case(18:18)
       Lx = 2*dd_pi*rearth0
       dx = Lx / ne_x
       dy = dx
       Ly = ne_y*dy
       Sx = -Lx/2
       Sy = -Ly/2
    end if
    !$omp end master
    !$omp barrier

    
  end subroutine init
  
end module planar_transport_tests

#endif
