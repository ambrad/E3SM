module gllfvremap_mod
  ! API for high-order, shape-preseving FV <-> GLL remap.
  ! AMB 2019/07 Initial

  use hybrid_mod, only: hybrid_t
  use kinds, only: real_kind
  use dimensions_mod, only: nlev, np, npsq, qsize, nelemd

  implicit none

  private

  integer, parameter :: nphys_max = np

  ! Data type and functions for high-order, shape-preserving FV <-> GLL remap.
  type, public :: GllFvRemap_t
     integer :: nphys
     real(kind=real_kind) :: &
          w_gg(np,np), &
          w_ff(nphys_max,nphys_max), &
          M_gf(np,np,nphys_max,nphys_max), &
          w_sgsg(np,np), &
          M_sgf(np,np,nphys_max,nphys_max), &
          interp(np,np,nphys_max,nphys_max), &
          R(npsq,nphys_max*nphys_max)
     real(kind=real_kind), allocatable :: &
          fv_metdet(:,:,:) ! (nphys,nphys,nelemd)
  end type GllFvRemap_t

  type (GllFvRemap_t), private :: gfr

  public :: &
       gfr_init, &
       gfr_finish, &
       gfr_fv_phys_to_dyn, &
       gfr_fv_phys_to_dyn_topo, &
       gfr_dyn_to_fv_phys, &
       gfr_test

contains

  subroutine gfr_init(nphys)
    use parallel_mod, only: abortmp

    integer, intent(in) :: nphys

    if (nphys > np) then
       call abortmp('nphys must be <= np')
    end if

    gfr%nphys = nphys
    allocate(gfr%fv_metdet(nphys,nphys,nelemd))
  end subroutine gfr_init

  subroutine gfr_finish()
    if (allocated(gfr%fv_metdet)) deallocate(gfr%fv_metdet)
  end subroutine gfr_finish

  subroutine gfr_fv_phys_to_dyn()
  end subroutine gfr_fv_phys_to_dyn

  subroutine gfr_fv_phys_to_dyn_topo()
  end subroutine gfr_fv_phys_to_dyn_topo

  subroutine gfr_dyn_to_fv_phys()
  end subroutine gfr_dyn_to_fv_phys

  subroutine gfr_test(hybrid, nets, nete, hvcoord, deriv, elem)
    use derivative_mod, only: derivative_t
    use element_mod, only: element_t
    use hybvcoord_mod, only: hvcoord_t

    type (hybrid_t), intent(in) :: hybrid
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(inout) :: elem(:)
    type (hvcoord_t) , intent(in) :: hvcoord
    integer, intent(in) :: nets, nete

    integer :: nphys

    print *, 'gfr_test'

    do nphys = 1, np
       call gfr_init(nphys)
       call gfr_finish()
    end do
  end subroutine gfr_test

end module gllfvremap_mod
