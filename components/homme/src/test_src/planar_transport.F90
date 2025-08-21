#ifndef CAM
#include "config.h"

module planar_transport_tests
  use kinds, only: rl=>real_kind, iulog
  use element_mod, only: element_t
  use parallel_mod, only: parallel_t, abortmp
  use hybrid_mod, only: hybrid_t
  use hybvcoord_mod, only: hvcoord_t, set_layer_locations
  use derivative_mod, only: derivative_t
  use physical_constants, only: Lx, Ly, Sx, Sy

  implicit none
  private

  public :: test_conv_planar_advection, print_conv_planar_advection_results

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
  
end module planar_transport_tests

#endif
