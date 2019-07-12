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
     integer :: nphys, npi
     real(kind=real_kind) :: &
          w_gg(np,np), &
          w_ff(nphys_max,nphys_max), &
          M_gf(np,np,nphys_max,nphys_max), &
          w_sgsg(np,np), &
          M_sgf(np,np,nphys_max,nphys_max), &
          R(npsq,nphys_max*nphys_max), &
          interp(np,np,nphys_max,nphys_max)
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
    if (gfr%nphys == 1) then
       gfr%npi = 2
    else
       gfr%npi = max(3, nphys)
    end if

    call gfr_init_w_gg(np, gfr%w_gg)
    call gfr_init_w_gg(gfr%npi, gfr%w_sgsg)
    call gfr_init_w_ff(nphys, gfr%w_ff)
    call gfr_init_M_gf(np, nphys, gfr%M_gf)
    call gfr_init_M_gf(gfr%npi, nphys, gfr%M_sgf)
    call gfr_init_R(gfr%npi, nphys, gfr%w_sgsg, gfr%M_sgf, gfr%R)
    call gfr_init_interp_matrix(gfr%npi, gfr%interp)

    allocate(gfr%fv_metdet(nphys,nphys,nelemd))
    call gfr_init_fv_metdet(gfr)
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

  subroutine gfr_init_w_gg(np, w_gg)
    integer, intent(in) :: np
    real(kind=real_kind), intent(out) :: w_gg(:,:)
  end subroutine gfr_init_w_gg

  subroutine gfr_init_w_ff(nphys, w_ff)
    integer, intent(in) :: nphys
    real(kind=real_kind), intent(out) :: w_ff(:,:)
  end subroutine gfr_init_w_ff

  subroutine gfr_init_M_gf(np, nphys, M_gf)
    integer, intent(in) :: np, nphys
    real(kind=real_kind), intent(out) :: M_gf(:,:,:,:)
  end subroutine gfr_init_M_gf

  subroutine gfr_init_R(np, nphys, w_gg, M_gf, R)
    integer, intent(in) :: np, nphys
    real(kind=real_kind), intent(in) :: w_gg(:,:), M_gf(:,:,:,:)
    real(kind=real_kind), intent(out) :: R(:,:)
  end subroutine gfr_init_R

  subroutine gfr_init_fv_metdet(gfr)
    type (GllFvRemap_t), intent(inout) :: gfr
  end subroutine gfr_init_fv_metdet

  subroutine gfr_init_interp_matrix(npsrc, interp)
    integer, intent(in) :: npsrc
    real(kind=real_kind), intent(out) :: interp(:,:,:,:)
  end subroutine gfr_init_interp_matrix

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
       ! This is meant to be called before threading starts.
       if (hybrid%masterthread) call gfr_init(nphys)
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif

       ! This is meant to be called after threading ends.
       if (hybrid%masterthread) call gfr_finish()
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif
    end do
  end subroutine gfr_test

end module gllfvremap_mod
