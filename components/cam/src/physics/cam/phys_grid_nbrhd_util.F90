module phys_grid_nbrhd_util
  ! These routines are in a separate module to resolve dependencies properly.

  use ppgrid, only: begchunk, endchunk, nbrhdchunk
  use kinds, only: real_kind, int_kind
  use phys_grid, only:
  use dyn_grid, only:
  use physics_types, only: physics_state
  use phys_grid_nbrhd ! all public routines

  implicit none
  private

  public :: &
       nbrhd_copy_states, &
       nbrhd_test_api

contains

  subroutine nbrhd_copy_states(phys_state)
    ! Copy state from normal chunks to the extra neighborhood chunk. These are
    ! neighborhood columns whose data already exist on this pe.

    type (physics_state), intent(inout) :: phys_state(begchunk:endchunk+nbrhdchunk)

    integer (int_kind) :: n, i, lchnk, lchnke, icol, icole, pcnst

    lchnke = endchunk+nbrhdchunk
    pcnst = nbrhd_get_option_pcnst()
    n = nbrhd_get_num_copies()
    do i = 1, n
       call nbrhd_get_copy_idxs(i, lchnk, icol, icole)
       phys_state(lchnke)%ps   (icole  ) = phys_state(lchnk)%ps   (icol  )
       phys_state(lchnke)%phis (icole  ) = phys_state(lchnk)%phis (icol  )
       phys_state(lchnke)%T    (icole,:) = phys_state(lchnk)%T    (icol,:)
       phys_state(lchnke)%u    (icole,:) = phys_state(lchnk)%u    (icol,:)
       phys_state(lchnke)%v    (icole,:) = phys_state(lchnk)%v    (icol,:)
       phys_state(lchnke)%omega(icole,:) = phys_state(lchnk)%omega(icol,:)
       phys_state(lchnke)%q(icole,:,1:pcnst) = phys_state(lchnk)%q(icol,:,1:pcnst)
    end do
  end subroutine nbrhd_copy_states

  subroutine nbrhd_test_api()
  end subroutine nbrhd_test_api

end module phys_grid_nbrhd_util
