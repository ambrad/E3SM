module phys_grid_nbrhd_util
  ! These routines are in a separate module to resolve dependencies properly.

  use parallel_mod, only: par
  use spmd_utils, only: iam, masterproc, npes
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile, only: iulog
  use ppgrid, only: begchunk, endchunk, nbrhdchunk
  use kinds, only: real_kind, int_kind
  use phys_grid, only:
  use dyn_grid, only: get_horiz_grid_dim_d, get_horiz_grid_d
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
    real(r8), allocatable, dimension(:) :: lats_d, lons_d
    integer :: d1, d2, ngcols

    call get_horiz_grid_dim_d(d1, d2)
    ngcols = d1*d2
    allocate(lats_d(ngcols), lons_d(ngcols))
    call get_horiz_grid_d(ngcols, clat_d_out=lats_d, clon_d_out=lons_d)

    if (nbrhd_get_option_block_to_chunk_on()) call test_api(lats_d, lons_d, .true. )
    if (nbrhd_get_option_chunk_to_chunk_on()) call test_api(lats_d, lons_d, .false.)

    deallocate(lats_d, lons_d)
  end subroutine nbrhd_test_api

  subroutine test_api(lats_d, lons_d, owning_blocks)
    real(r8), intent(in) :: lats_d(:), lons_d(:)
    logical, intent(in) :: owning_blocks
  end subroutine test_api

  function test(nerr, cond, message) result(out)
    integer, intent(inout) :: nerr
    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    if (.not. cond) then
       write(iulog,*) 'nbr> test ', trim(message)
       nerr = nerr + 1
    end if
    out = cond
  end function test

end module phys_grid_nbrhd_util
