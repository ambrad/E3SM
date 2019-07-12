! AMB 2019/07 Add GllFvRemap_t API for high-order FV -> GLL remap and
!             shape-preserving FV <-> GLL remap. GLL -> FV remap was already
!             high order.

module gllfvremap_mod
  implicit none

  private

  ! Data type and functions for high-order, shape-preserving FV <-> GLL remap.
  type, public :: GllFvRemap_t
     real(kind=real_kind) :: &
          w_gg(np,np), &
          w_ff(fv_nphys,fv_nphys), &
          M_gf(np,np,fv_nphys,fv_nphys), &
          w_sgsg(np,np), &
          M_sgf(np,np,fv_nphys,fv_nphys), &
          interp(np,np,fv_nphys,fv_nphys), &
          R(npsq,fv_nphys*fv_nphys)
     real(kind=real_kind), allocatable :: &
          fv_metdet(:,:,:) ! (fv_nphys,fv_nphys,nelem)
  end type GllFvRemap_t

  type (GllFvRemap_t), private :: gfr

  public :: &
       gfr_init, &
       gfr_finish, &
       gfr_fv_phys_to_dyn, &
       fv_phys_to_dyn_topo, &
       gfr_dyn_to_fv_phys

contains


  subroutine gfr_init()
  end subroutine gfr_init

  subroutine gfr_finish()
  end subroutine gfr_finish

  subroutine gfr_fv_phys_to_dyn()
  end subroutine gfr_fv_phys_to_dyn

  subroutine fv_phys_to_dyn_topo()
  end subroutine fv_phys_to_dyn_topo

  subroutine gfr_dyn_to_fv_phys()
  end subroutine gfr_dyn_to_fv_phys

end module gllfvremap_mod
