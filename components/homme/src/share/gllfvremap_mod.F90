#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!todo
! - omega_p
! - mod vector_dp routine to do in/out arrays efficiently in dp_coupling
! - checker routine callable from dcmip1
! - test vector_dp routines: conservation
! - area correction: alpha
! - topo roughness
! - np4-np2 instead of np4-pg1
! - halo exchange buffers

module gllfvremap_mod
  ! High-order, mass-conserving, optionally shape-preserving
  !     FV physics <-> GLL dynamics
  ! remap.
  !
  ! AMB 2019/07 Initial

  use hybrid_mod, only: hybrid_t
  use kinds, only: real_kind
  use dimensions_mod, only: np, npsq, qsize, nelemd
  use element_mod, only: element_t
  use coordinate_systems_mod, only: spherical_polar_t

  implicit none

  private

  integer, parameter :: nphys_max = np

  real(kind=real_kind), parameter :: &
       zero = 0.0_real_kind, half = 0.5_real_kind, &
       one = 1.0_real_kind, two = 2.0_real_kind

  ! Data type and functions for high-order, shape-preserving FV <-> GLL remap.
  type, public :: GllFvRemap_t
     integer :: nphys, npi
     real(kind=real_kind) :: &
          ! Node or cell weights
          w_gg(np,np), &   ! GLL np
          w_ff(nphys_max,nphys_max), & ! FV nphys
          w_sgsg(np,np), & ! GLL npi
          ! Mixed mass matrices
          M_sgf(np,np,nphys_max,nphys_max), & ! GLL npi, FV nphys
          M_gf(np,np,nphys_max,nphys_max), &  ! GLL np,  FV nphys
          ! Interpolate from GLL npi to GLL np
          interp(np,np,np,np), &
          ! Remap FV nphys -> GLL np
          f2g_remapd(nphys_max,nphys_max,np,np)
     ! FV subcell areas; FV analogue of GLL elem(ie)%metdet arrays
     real(kind=real_kind), allocatable :: &
          fv_metdet(:,:,:), & ! (nphys,nphys,nelemd)
          ! Vector on ref elem -> vector on sphere
          D_f(:,:,:,:,:), &   ! (nphys,nphys,2,2,nelemd)
          ! Inverse of D_f
          Dinv_f(:,:,:,:,:), &
          qmin(:,:,:), qmax(:,:,:)
     type (spherical_polar_t), allocatable :: &
          spherep_f(:,:,:) ! (nphys,nphys,nelemd)
  end type GllFvRemap_t

  type (GllFvRemap_t), private :: gfr

  ! Main API.
  public :: &
       gfr_init, &
       gfr_finish, &
       gfr_fv_phys_to_dyn, &
       gfr_fv_phys_to_dyn_topo, &
       gfr_dyn_to_fv_phys

  ! Testing API.
  public :: &
       gfr_test, &
       gfr_g2f_scalar, gfr_g2f_scalar_dp, gfr_g2f_vector, gfr_g2f_vector_dp, &
       gfr_g2f_mixing_ratios, gfr_f2g_scalar, gfr_f2g_scalar_dp, gfr_f2g_vector, &
       gfr_f2g_vector_dp, gfr_f2g_mixing_ratios_a, gfr_f2g_mixing_ratios_b, &
       gfr_f2g_mixing_ratios_c, gfr_f2g_dss, gfr_get_latlon, gfr_g_make_nonnegative

contains

  subroutine gfr_init(hybrid, elem, nphys)
    use dimensions_mod, only: nlev
    use parallel_mod, only: abortmp

    type (hybrid_t), intent(in) :: hybrid
    type (element_t), intent(in) :: elem(:)
    integer, intent(in) :: nphys

    real(real_kind) :: R(npsq,nphys_max*nphys_max)

    if (hybrid%masterthread) print *, 'gfr> init nphys', nphys

    if (nphys > np) then
       ! The FV -> GLL map is defined only if nphys <= np. If we ever are
       ! interested in the case of nphys > np, we will need to write a different
       ! map. See "!assume" annotations for mathematical assumptions in
       ! particular routines.
       call abortmp('nphys must be <= np')
    end if

    gfr%nphys = nphys
    if (gfr%nphys == 1) then
       ! If 3 is used, then the gfr_reconstructd_* routines must be mod'ed. In
       ! particular, a limiter is needed, complicating the algorithm.
       gfr%npi = 2
    else
       gfr%npi = max(3, nphys)
    end if

    call gfr_init_w_gg(np, gfr%w_gg)
    call gfr_init_w_gg(gfr%npi, gfr%w_sgsg)
    call gfr_init_w_ff(nphys, gfr%w_ff)
    call gfr_init_M_gf(np, nphys, gfr%M_gf)
    call gfr_init_M_gf(gfr%npi, nphys, gfr%M_sgf)
    call gfr_init_R(gfr%npi, nphys, gfr%w_sgsg, gfr%M_sgf, R)
    call gfr_init_interp_matrix(gfr%npi, gfr%interp)
    call gfr_init_f2g_remapd(gfr, R)

    allocate(gfr%fv_metdet(nphys,nphys,nelemd), &
         gfr%D_f(nphys,nphys,2,2,nelemd), gfr%Dinv_f(nphys,nphys,2,2,nelemd), &
         gfr%qmin(nlev,qsize,nelemd), gfr%qmax(nlev,qsize,nelemd), &
         gfr%spherep_f(nphys,nphys,nelemd))
    call gfr_init_fv_metdet(elem, gfr)
    call gfr_init_geometry(elem, gfr)
  end subroutine gfr_init

  subroutine gfr_finish()
    if (allocated(gfr%fv_metdet)) then
       deallocate(gfr%fv_metdet, gfr%D_f, gfr%Dinv_f, gfr%qmin, gfr%qmax, gfr%spherep_f)
    end if
  end subroutine gfr_finish

  subroutine gfr_dyn_to_fv_phys(nt, hvcoord, elem, ps, phis, T, uv, omega_p, q, &
       nets_in, nete_in)
    use dimensions_mod, only: nlev
    use hybvcoord_mod, only: hvcoord_t
    use element_ops, only: get_temperature

    integer, intent(in) :: nt
    type (hvcoord_t), intent(in) :: hvcoord
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(inout) :: ps(:,:), phis(:,:), T(:,:,:), &
         uv(:,:,:,:), omega_p(:,:,:), q(:,:,:,:)
    integer, intent(in), optional :: nets_in, nete_in

    real(kind=real_kind) :: dp(np,np,nlev), dp_fv(np,np,nlev), wr1(np,np,nlev), wr2(np,np,nlev)
    integer :: nets, nete, ie, nf, ncol, qi, qsize

    if (present(nets_in)) then
       nets = nets_in
       nete = nete_in
    else
       nets = 1
       nete = size(elem)
    end if
    nf = gfr%nphys
    ncol = nf*nf

    qsize = size(q,3)
    
    do ie = nets,nete
       call gfr_g2f_scalar(ie, elem(ie)%metdet, elem(ie)%state%ps_v(:,:,nt:nt), &
            wr1(:,:,:1))
       ps(:ncol,ie) = reshape(wr1(:nf,:nf,1), (/ncol/))

       wr2(:,:,1) = elem(ie)%state%phis(:,:)
       call gfr_g2f_scalar(ie, elem(ie)%metdet, wr2(:,:,:1), wr1(:,:,:1))
       phis(:ncol,ie) = reshape(wr1(:nf,:nf,1), (/ncol/))

       call calc_dp(hvcoord, elem(ie)%state%ps_v(:,:,nt), dp)
       call gfr_g2f_scalar(ie, elem(ie)%metdet, dp, dp_fv)

       call get_temperature(elem(ie), wr2, hvcoord, nt)
       call gfr_g2f_scalar_dp(ie, elem(ie)%metdet, dp, dp_fv, wr2, wr1)
       T(:ncol,:,ie) = reshape(wr1(:nf,:nf,:), (/ncol,nlev/))

       call gfr_g2f_vector_dp(ie, elem, dp, dp_fv, &
            elem(ie)%state%v(:,:,1,:,nt), elem(ie)%state%v(:,:,2,:,nt), &
            wr1, wr2)
       uv(:ncol,1,:,ie) = reshape(wr1(:nf,:nf,:), (/ncol,nlev/))
       uv(:ncol,2,:,ie) = reshape(wr2(:nf,:nf,:), (/ncol,nlev/))

       !TODO omega_p

       do qi = 1,qsize
          call gfr_g2f_mixing_ratio(ie, elem(ie)%metdet, dp, dp_fv, &
               dp*elem(ie)%state%Q(:,:,:,qi), wr1)
          q(:ncol,:,qi,ie) = reshape(wr1(:nf,:nf,:), (/ncol,nlev/))
       end do
    end do
  end subroutine gfr_dyn_to_fv_phys

  subroutine gfr_fv_phys_to_dyn(hybrid, nt, hvcoord, elem, T, uv, q, nets_in, nete_in)
    use dimensions_mod, only: nlev
    use hybvcoord_mod, only: hvcoord_t

    type (hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: nt
    type (hvcoord_t), intent(in) :: hvcoord
    type (element_t), intent(inout) :: elem(:)
    real(kind=real_kind), intent(in) :: T(:,:,:), uv(:,:,:,:), q(:,:,:,:)
    integer, intent(in), optional :: nets_in, nete_in

    real(kind=real_kind) :: dp(np,np,nlev), dp_fv(np,np,nlev), wr1(np,np,nlev), wr2(np,np,nlev)
    integer :: nets, nete, ie, nf, ncol, k, qsize, qi

    if (present(nets_in)) then
       nets = nets_in
       nete = nete_in
    else
       nets = 1
       nete = size(elem)
    end if
    nf = gfr%nphys
    ncol = nf*nf

    qsize = size(q,3)

    do ie = nets,nete
       call calc_dp(hvcoord, elem(ie)%state%ps_v(:,:,nt), dp)
       call gfr_g2f_scalar(ie, elem(ie)%metdet, dp, dp_fv)

       wr1(:nf,:nf,:) = reshape(uv(:ncol,1,:,ie), (/nf,nf,nlev/))
       wr2(:nf,:nf,:) = reshape(uv(:ncol,2,:,ie), (/nf,nf,nlev/))
       call gfr_f2g_vector_dp(ie, elem, dp_fv, dp, wr1, wr2, &
            elem(ie)%derived%FM(:,:,1,:), elem(ie)%derived%FM(:,:,2,:))

       wr1(:nf,:nf,:) = reshape(T(:ncol,:,ie), (/nf,nf,nlev/))
       call gfr_f2g_scalar_dp(ie, elem(ie)%metdet, dp_fv, dp, wr1, elem(ie)%derived%FT)

       do qi = 1,qsize
          ! FV Q_ten
          !   GLL Q0 -> FV Q0
          call gfr_g2f_mixing_ratio(ie, elem(ie)%metdet, dp, dp_fv, &
               dp*elem(ie)%state%Q(:,:,:,qi), wr1)
          !   FV Q_ten = FV Q1 - FV Q0
          wr1(:nf,:nf,:) = reshape(q(:ncol,:,qi,ie), (/nf,nf,nlev/)) - wr1(:nf,:nf,:)
          ! GLL Q_ten
          call gfr_f2g_scalar_dp(ie, elem(ie)%metdet, dp_fv, dp, wr1, wr2)
          ! GLL Q1
          elem(ie)%derived%FQ(:,:,:,qi) = elem(ie)%state%Q(:,:,:,qi) + wr2
          ! Get limiter bounds.
          do k = 1,nlev
             gfr%qmin(k,qi,ie) = minval(q(:ncol,k,qi,ie))
             gfr%qmax(k,qi,ie) = maxval(q(:ncol,k,qi,ie))
          end do
       end do
    end do

    ! Halo exchange limiter bounds.
    call gfr_f2g_mixing_ratios_b(hybrid, nets, nete, gfr%qmin, gfr%qmax)

    do ie = nets,nete
       call calc_dp(hvcoord, elem(ie)%state%ps_v(:,:,nt), dp)
       do qi = 1,qsize
          ! Limit GLL Q1.
          do k = 1,nlev
             call limiter_clip_and_sum(np, elem(ie)%spheremp, gfr%qmin(k,qi,ie), &
                  gfr%qmax(k,qi,ie), dp(:,:,k), elem(ie)%derived%FQ(:,:,k,qi))
          end do
          ! Final GLL Q1, except for DSS, which is not done in this routine.
          elem(ie)%derived%FQ(:,:,:,qi) = elem(ie)%derived%FQ(:,:,:,qi)/dp
       end do
    end do
  end subroutine gfr_fv_phys_to_dyn

  subroutine gfr_fv_phys_to_dyn_topo()
  end subroutine gfr_fv_phys_to_dyn_topo

  subroutine gfr_init_w_gg(np, w_gg)
    use quadrature_mod, only : gausslobatto, quadrature_t
    
    integer, intent(in) :: np
    real(kind=real_kind), intent(out) :: w_gg(:,:)

    type (quadrature_t) :: gll
    integer :: i,j

    gll = gausslobatto(np)

    do j = 1,np
       do i = 1,np
          w_gg(i,j) = gll%weights(i)*gll%weights(j)
       end do
    end do
    
    call gll_cleanup(gll)
  end subroutine gfr_init_w_gg

  subroutine gfr_init_w_ff(nphys, w_ff)
    integer, intent(in) :: nphys
    real(kind=real_kind), intent(out) :: w_ff(:,:)

    w_ff(:nphys,:nphys) = two/real(nphys, real_kind)
  end subroutine gfr_init_w_ff

  subroutine gll_cleanup(gll)
    use quadrature_mod, only : quadrature_t

    type (quadrature_t), intent(inout) :: gll

    deallocate(gll%points, gll%weights)
  end subroutine gll_cleanup

  subroutine calc_dp(hvcoord, ps, dp)
    use hybvcoord_mod, only: hvcoord_t
    use dimensions_mod, only: nlev

    type (hvcoord_t), intent(in) :: hvcoord
    real(kind=real_kind), intent(in) :: ps(:,:)
    real(kind=real_kind), intent(out) :: dp(:,:,:)

    integer :: k

    do k = 1,nlev
       dp(:,:,k) = (hvcoord%hyai(k+1) - hvcoord%hyai(k))*hvcoord%ps0 + &
            (hvcoord%hybi(k+1) - hvcoord%hybi(k))*ps
    end do
  end subroutine calc_dp

  subroutine eval_lagrange_bases(gll, np, x, y)
    ! Evaluate the GLL basis functions at x in [-1,1], writing the values to
    ! y(1:np).
    use quadrature_mod, only : quadrature_t
    
    type (quadrature_t), intent(in) :: gll
    integer, intent(in) :: np
    real(kind=real_kind), intent(in) :: x ! in [-1,1]
    real(kind=real_kind), intent(out) :: y(np)

    integer :: i, j
    real(kind=real_kind) :: f

    do i = 1,np
       f = one
       do j = 1,np
          if (j /= i) then
             f = f*((x - gll%points(j))/(gll%points(i) - gll%points(j)))
          end if
       end do
       y(i) = f
    end do
  end subroutine eval_lagrange_bases

  subroutine gfr_init_M_gf(np, nphys, M_gf)
    use quadrature_mod, only : gausslobatto, quadrature_t

    integer, intent(in) :: np, nphys
    real(kind=real_kind), intent(out) :: M_gf(:,:,:,:)

    type (quadrature_t) :: gll
    integer :: gi, gj, fi, fj, qi, qj
    real(kind=real_kind) :: xs, xe, ys, ye, ref, bi(np), bj(np)

    gll = gausslobatto(np)

    M_gf = zero

    do fj = 1,nphys
       ! The subcell is [xs,xe]x[ys,ye].
       xs = two*real(fj-1, real_kind)/real(nphys, real_kind) - one
       xe = two*real(fj, real_kind)/real(nphys, real_kind) - one
       do fi = 1,nphys
          ys = two*real(fi-1, real_kind)/real(nphys, real_kind) - one
          ye = two*real(fi, real_kind)/real(nphys, real_kind) - one
          ! Use GLL quadrature within this subcell.
          do qj = 1,np
             ! (xref,yref) are w.r.t. the [-1,1]^2 reference domain mapped to
             ! the subcell.
             ref = xs + half*(xe - xs)*(1 + gll%points(qj))
             call eval_lagrange_bases(gll, np, ref, bj)
             do qi = 1,np
                ref = ys + half*(ye - ys)*(1 + gll%points(qi))
                call eval_lagrange_bases(gll, np, ref, bi)
                do gj = 1,np
                   do gi = 1,np
                      ! Accumulate each GLL basis's contribution to this
                      ! subcell.
                      M_gf(gi,gj,fi,fj) = M_gf(gi,gj,fi,fj) + &
                           gll%weights(qi)*gll%weights(qj)*bi(gi)*bj(gj)
                   end do
                end do
             end do
          end do
       end do
    end do

    M_gf = M_gf/real(nphys*nphys, real_kind)

    call gll_cleanup(gll)
  end subroutine gfr_init_M_gf

  subroutine gfr_init_R(np, nphys, w_gg, M_gf, R)
    ! We want to solve
    !     min_g 1/2 g'M_gg g - g' M_gf f
    !      st   M_gf' g = M_ff f,
    ! which gives
    !     [M_gg -M_gf] [g] = [M_gf f]
    !     [M_gf'  0  ] [y]   [M_ff f].
    ! Recall M_gg, M_ff are diag. Let
    !     S = M_gf' inv(M_gg) M_gf.
    ! Then
    !     g = inv(M_gg) M_gf inv(S) M_ff f.
    ! Compute the QR factorization sqrt(inv(M_gg)) M_gf = Q R so that S =
    ! R'R. In this module, we can take M_gg = diag(w_gg) and M_ff = diag(w_ff)
    ! with no loss of accuracy.
    integer, intent(in) :: np, nphys
    real(kind=real_kind), intent(in) :: w_gg(:,:), M_gf(:,:,:,:)
    real(kind=real_kind), intent(out) :: R(:,:)

    real(kind=real_kind) :: wrk1(np*np*nphys*nphys), wrk2(np*np*nphys*nphys)
    integer :: gi, gj, fi, fj, npsq, info

    !assume nphys <= np

    do fj = 1,nphys
       do fi = 1,nphys
          do gi = 1,np
             do gj = 1,np
                R(np*(gi-1) + gj, nphys*(fi-1) + fj) = &
                     M_gf(gi,gj,fi,fj)/sqrt(w_gg(gi,gj))
             end do
          end do
       end do
    end do
    call dgeqrf(np*np, nphys*nphys, R, size(R,1), wrk1, wrk2, np*np*nphys*nphys, info)
  end subroutine gfr_init_R

  subroutine gfr_init_interp_matrix(npsrc, interp)
    use quadrature_mod, only : gausslobatto, quadrature_t

    integer, intent(in) :: npsrc
    real(kind=real_kind), intent(out) :: interp(:,:,:,:)

    type (quadrature_t) :: glls, gllt
    integer :: si, sj, ti, tj
    real(kind=real_kind) :: bi(npsrc), bj(npsrc)

    glls = gausslobatto(npsrc)
    gllt = gausslobatto(np)

    do tj = 1,np
       call eval_lagrange_bases(glls, npsrc, real(gllt%points(tj), real_kind), bj)
       do ti = 1,np
          call eval_lagrange_bases(glls, npsrc, real(gllt%points(ti), real_kind), bi)
          do sj = 1,npsrc
             do si = 1,npsrc
                interp(si,sj,ti,tj) = bi(si)*bj(sj)
             end do
          end do
       end do
    end do

    call gll_cleanup(glls)
    call gll_cleanup(gllt)
  end subroutine gfr_init_interp_matrix

  subroutine gfr_init_f2g_remapd(gfr, R)
    type (GllFvRemap_t), intent(inout) :: gfr
    real(kind=real_kind), intent(in) :: R(:,:)

    integer :: fi, fj
    real(kind=real_kind) :: f(np,np), g(np,np)

    !assume nphys <= np

    ! Apply gfr_init_f2g_remapd_op to the Id matrix to get the remap operator's
    ! matrix representation.
    f = zero
    do fi = 1,gfr%nphys
       do fj = 1,gfr%nphys
          f(fi,fj) = one
          call gfr_f2g_remapd_op(gfr, R, f, g)
          gfr%f2g_remapd(fi,fj,:,:) = g
          f(fi,fj) = zero
       end do
    end do
  end subroutine gfr_init_f2g_remapd

  subroutine gfr_f2g_remapd_op(gfr, R, f, g)
    type (GllFvRemap_t), intent(in) :: gfr
    real(kind=real_kind), intent(in) :: R(:,:)
    real(kind=real_kind), intent(in) :: f(:,:)
    real(kind=real_kind), intent(out) :: g(:,:)

    integer :: nf, nf2, npi, np2, gi, gj, fi, fj, info
    real(kind=real_kind) :: accum, wrk(gfr%nphys,gfr%nphys), x(np,np)

    !assume nphys <= np

    nf = gfr%nphys
    nf2 = nf*nf
    npi = gfr%npi
    np2 = np*np

    ! Solve the constrained projection described in gfr_init_R:
    !     g = inv(M_sgsg) M_sgf inv(S) M_ff f
    wrk(:nf,:nf) = gfr%w_ff(:nf,:nf)*f(:nf,:nf)
    call dtrtrs('u', 't', 'n', nf2, 1, R, size(R,1), wrk, np2, info)
    call dtrtrs('u', 'n', 'n', nf2, 1, R, size(R,1), wrk, np2, info)
    g(:npi,:npi) = zero
    do fj = 1,nf
       do fi = 1,nf
          do gj = 1,npi
             do gi = 1,npi
                g(gi,gj) = g(gi,gj) + gfr%M_sgf(gi,gj,fi,fj)*wrk(fi,fj)
             end do
          end do
       end do
    end do
    if (npi < np) then
       ! Finish the projection:
       !     wrk = inv(M_sgsg) g
       do gj = 1,npi
          do gi = 1,npi
             x(gi,gj) = g(gi,gj)/gfr%w_sgsg(gi,gj)
          end do
       end do 
       ! Interpolate from npi to np; if npi = np, this is just the Id matrix.
       do fj = 1,np
          do fi = 1,np
             accum = zero
             do gj = 1,npi
                do gi = 1,npi
                   accum = accum + gfr%interp(gi,gj,fi,fj)*x(gi,gj)
                end do
             end do
             g(fi,fj) = accum
          end do
       end do
    else
       ! Finish the projection.
       do gj = 1,np
          do gi = 1,np
             g(gi,gj) = g(gi,gj)/gfr%w_gg(gi,gj)
          end do
       end do
    end if
  end subroutine gfr_f2g_remapd_op

  subroutine gfr_init_fv_metdet(elem, gfr)
    type (element_t), intent(in) :: elem(:)
    type (GllFvRemap_t), intent(inout) :: gfr

    real(kind=real_kind) :: ones(np,np)
    integer :: ie

    ones = one
    do ie = 1,nelemd
       call gfr_g2f_remapd(gfr, elem(ie)%metdet, ones, ones, gfr%fv_metdet(:,:,ie))
    end do
  end subroutine gfr_init_fv_metdet

  subroutine gfr_f_ref_coord(nphys, i, a)
    ! FV subcell center in ref [-1,1]^2 coord.
    integer, intent(in) :: nphys, i
    real(kind=real_kind), intent(out) :: a

    a = two*((real(i-1, real_kind) + half)/real(nphys, real_kind)) - one
  end subroutine gfr_f_ref_coord

  subroutine gfr_init_geometry(elem, gfr)
    use cube_mod, only: Dmap, ref2sphere
    use control_mod, only: cubed_sphere_map

    type (element_t), intent(in) :: elem(:)
    type (GllFvRemap_t), intent(inout) :: gfr

    real(kind=real_kind) :: wrk(2,2), det, a, b
    integer :: ie, nf, i, j

    nf = gfr%nphys

    do ie = 1,nelemd
       do j = 1,nf
          call gfr_f_ref_coord(nf, j, b)
          do i = 1,nf
             call gfr_f_ref_coord(nf, i, a)

             call Dmap(wrk, a, b, elem(ie)%corners3D, cubed_sphere_map, elem(ie)%cartp, &
                  elem(ie)%facenum)
             gfr%D_f(i,j,:,:,ie) = wrk

             det = wrk(1,1)*wrk(2,2) - wrk(1,2)*wrk(2,1)
             gfr%Dinv_f(i,j,1,1,ie) =  wrk(2,2)/det
             gfr%Dinv_f(i,j,1,2,ie) = -wrk(1,2)/det
             gfr%Dinv_f(i,j,2,1,ie) = -wrk(2,1)/det
             gfr%Dinv_f(i,j,2,2,ie) =  wrk(1,1)/det

             gfr%spherep_f(i,j,ie) = ref2sphere(a, b, elem(ie)%corners3D, cubed_sphere_map, &
                  elem(ie)%corners, elem(ie)%facenum)
          end do
       end do
    end do
  end subroutine gfr_init_geometry

  subroutine gfr_g2f_scalar(ie, gll_metdet, g, f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), g(:,:,:)
    real(kind=real_kind), intent(out) :: f(:,:,:)

    integer :: k

    do k = 1, size(g,3)
       call gfr_g2f_remapd(gfr, gll_metdet, gfr%fv_metdet(:,:,ie), g(:,:,k), f(:,:,k))
    end do
  end subroutine gfr_g2f_scalar

  subroutine gfr_g2f_scalar_dp(ie, gll_metdet, dp_g, dp_f, g, f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_g(:,:,:), dp_f(:,:,:), g(:,:,:)
    real(kind=real_kind), intent(out) :: f(:,:,:)

    call gfr_g2f_scalar(ie, gll_metdet, dp_g*g, f)
    f = f/dp_f
  end subroutine gfr_g2f_scalar_dp

  subroutine gfr_g2f_vector(ie, elem, u_g, v_g, u_f, v_f)
    integer, intent(in) :: ie
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(in) :: u_g(:,:,:), v_g(:,:,:)
    real(kind=real_kind), intent(out) :: u_f(:,:,:), v_f(:,:,:)

    real(kind=real_kind) :: wg(np,np,2), wf(np,np,2), ones(np,np)
    integer :: k, d, nf

    nf = gfr%nphys
    ones = one

    do k = 1, size(u_g,3)
       ! sphere -> GLL ref
       do d = 1,2
          wg(:,:,d) = elem(ie)%D(:,:,d,1)*u_g(:,:,k) + elem(ie)%D(:,:,d,2)*v_g(:,:,k)
       end do
       ! Since we mapped to the ref element, we no longer should use the ref ->
       ! sphere Jacobians; use 1s instead.
       do d = 1,2
          call gfr_g2f_remapd(gfr, ones, ones, wg(:,:,d), wf(:,:,d))
       end do
       ! FV ref -> sphere
       do d = 1,2
          wg(:nf,:nf,d) = gfr%Dinv_f(:nf,:nf,d,1,ie)*wf(:nf,:nf,1) + &
               gfr%Dinv_f(:nf,:nf,d,2,ie)*wf(:nf,:nf,2)
       end do
       u_f(:nf,:nf,k) = wg(:nf,:nf,1)
       v_f(:nf,:nf,k) = wg(:nf,:nf,2)
    end do
  end subroutine gfr_g2f_vector

  subroutine gfr_g2f_vector_dp(ie, elem, dp_g, dp_f, u_g, v_g, u_f, v_f)
    integer, intent(in) :: ie
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(in) :: dp_g(:,:,:), dp_f(:,:,:), u_g(:,:,:), v_g(:,:,:)
    real(kind=real_kind), intent(out) :: u_f(:,:,:), v_f(:,:,:)

    call gfr_g2f_vector(ie, elem, dp_g*u_g, dp_g*v_g, u_f, v_f)
    u_f = u_f/dp_f
    v_f = v_f/dp_f
  end subroutine gfr_g2f_vector_dp

  subroutine gfr_g2f_mixing_ratio(ie, gll_metdet, dp_g, dp_f, qdp_g, q_f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_g(:,:,:), dp_f(:,:,:), &
         qdp_g(:,:,:)
    real(kind=real_kind), intent(out) :: q_f(:,:,:)

    real(kind=real_kind) :: qmin, qmax, wrk(np,np)
    integer :: q, k, nf

    nf = gfr%nphys
    do k = 1, size(qdp_g,3)
       call gfr_g2f_remapd(gfr, gll_metdet, gfr%fv_metdet(:,:,ie), &
            qdp_g(:,:,k), q_f(:,:,k))
       wrk = qdp_g(:,:,k)/dp_g(:,:,k)
       qmin = minval(wrk)
       qmax = maxval(wrk)
       call limiter_clip_and_sum(gfr%nphys, gfr%w_ff(:nf,:nf)*gfr%fv_metdet(:nf,:nf,ie), &
            qmin, qmax, dp_f(:,:,k), q_f(:,:,k))
       q_f(:,:,k) = q_f(:,:,k)/dp_f(:,:,k)
    end do
  end subroutine gfr_g2f_mixing_ratio

  subroutine gfr_g2f_mixing_ratios(ie, gll_metdet, dp_g, dp_f, qdp_g, q_f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_g(:,:,:), dp_f(:,:,:), &
         qdp_g(:,:,:,:)
    real(kind=real_kind), intent(out) :: q_f(:,:,:,:)

    real(kind=real_kind) :: qmin, qmax, wrk(np,np)
    integer :: q, k, nf

    nf = gfr%nphys
    do q = 1, size(qdp_g,4)
       call gfr_g2f_mixing_ratio(ie, gll_metdet, dp_g, dp_f, qdp_g(:,:,:,q), q_f(:,:,:,q))
    end do
  end subroutine gfr_g2f_mixing_ratios

  subroutine gfr_f2g_scalar(ie, gll_metdet, f, g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), f(:,:,:)
    real(kind=real_kind), intent(out) :: g(:,:,:)

    integer :: k

    do k = 1, size(g,3)
       call gfr_f2g_remapd(gfr, gll_metdet, gfr%fv_metdet(:,:,ie), f(:,:,k), g(:,:,k))
    end do
  end subroutine gfr_f2g_scalar

  subroutine gfr_f2g_scalar_dp(ie, gll_metdet, dp_f, dp_g, f, g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_f(:,:,:), dp_g(:,:,:), f(:,:,:)
    real(kind=real_kind), intent(out) :: g(:,:,:)

    call gfr_f2g_scalar(ie, gll_metdet, dp_f*f, g)
    g = g/dp_g
  end subroutine gfr_f2g_scalar_dp

  subroutine gfr_f2g_vector(ie, elem, u_f, v_f, u_g, v_g)
    integer, intent(in) :: ie
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(in) :: u_f(:,:,:), v_f(:,:,:)
    real(kind=real_kind), intent(out) :: u_g(:,:,:), v_g(:,:,:)

    real(kind=real_kind) :: wg(np,np,2), wf(np,np,2), ones(np,np)
    integer :: k, d, nf

    nf = gfr%nphys
    ones = one

    do k = 1, size(u_g,3)
       ! sphere -> FV ref
       do d = 1,2
          wf(:nf,:nf,d) = gfr%D_f(:nf,:nf,d,1,ie)*u_f(:nf,:nf,k) + &
               gfr%D_f(:nf,:nf,d,2,ie)*v_f(:nf,:nf,k)
       end do
       do d = 1,2
          call gfr_f2g_remapd(gfr, ones, ones, wf(:,:,d), wg(:,:,d))
       end do
       ! GLL ref -> sphere
       do d = 1,2
          wf(:,:,d) = elem(ie)%Dinv(:,:,d,1)*wg(:,:,1) + elem(ie)%Dinv(:,:,d,2)*wg(:,:,2)
       end do
       u_g(:,:,k) = wf(:,:,1)
       v_g(:,:,k) = wf(:,:,2)
    end do
  end subroutine gfr_f2g_vector

  subroutine gfr_f2g_vector_dp(ie, elem, dp_f, dp_g, u_f, v_f, u_g, v_g)
    integer, intent(in) :: ie
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(in) :: dp_f(:,:,:), dp_g(:,:,:), u_f(:,:,:), v_f(:,:,:)
    real(kind=real_kind), intent(out) :: u_g(:,:,:), v_g(:,:,:)

    call gfr_f2g_vector(ie, elem, dp_f*u_f, dp_f*v_f, u_g, v_g)
    u_g = u_g/dp_g
    v_g = v_g/dp_g
  end subroutine gfr_f2g_vector_dp

  subroutine gfr_f2g_mixing_ratios_a(ie, gll_metdet, dp_f, dp_g, q_f, q_g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_f(:,:,:), dp_g(:,:,:), &
         q_f(:,:,:,:)
    real(kind=real_kind), intent(out) :: q_g(:,:,:,:)

    integer :: q, k

    do q = 1, size(q_f,4)
       call gfr_f2g_scalar_dp(ie, gll_metdet, dp_f, dp_g, q_f(:,:,:,q), q_g(:,:,:,q))
    end do
  end subroutine gfr_f2g_mixing_ratios_a

  subroutine gfr_f2g_mixing_ratios_b(hybrid, nets, nete, qmin, qmax)
    use viscosity_mod, only: neighbor_minmax
    use prim_advection_base, only: edgeAdvQminmax !TODO rm kludge

    type (hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: nets, nete
    real(kind=real_kind), intent(inout) :: qmin(:,:,:), qmax(:,:,:)

    call neighbor_minmax(hybrid, edgeAdvQminmax, nets, nete, qmin, qmax)
  end subroutine gfr_f2g_mixing_ratios_b

  subroutine gfr_f2g_mixing_ratios_c(ie, elem, qmin, qmax, dp, q0, q_ten)
    ! Solve
    !     min norm(q_ten - q_ten*, 1)
    !      st dp'q_ten unchanged
    !         qmin <= q0 + q_ten <= qmax
    integer, intent(in) :: ie
    type (element_t), intent(in) :: elem(:)
    real(kind=real_kind), intent(in) :: dp(:,:,:), q0(:,:,:,:)
    real(kind=real_kind), intent(inout) :: qmin(:,:), qmax(:,:), q_ten(:,:,:,:)

    real(kind=real_kind) :: wrk(np,np)
    integer :: q, k

    do q = 1, size(q0,4)
       do k = 1, size(q0,3)
          wrk = dp(:,:,k)*(q0(:,:,k,q) + q_ten(:,:,k,q))
          call limiter_clip_and_sum(np, elem(ie)%spheremp, qmin(k,q), qmax(k,q), dp(:,:,k), wrk)
          q_ten(:,:,k,q) = wrk/dp(:,:,k) - q0(:,:,k,q)
       end do
    end do
  end subroutine gfr_f2g_mixing_ratios_c

  subroutine gfr_f2g_dss(hybrid, elem, nets, nete)
    use dimensions_mod, only: nlev, qsize
    use edge_mod, only: edgevpack_nlyr, edgevunpack_nlyr, edge_g
    use bndry_mod, only: bndry_exchangev

    type (hybrid_t), intent(in) :: hybrid
    type (element_t), intent(inout) :: elem(:)
    integer, intent(in) :: nets, nete

    integer :: ie, q, k

    !kludge 2 HEs until i get edge_g alloc'ed with the right size
    !TODO fix this kludge

    do ie = nets, nete
       do q = 1,qsize
          do k = 1,nlev
             elem(ie)%derived%FQ(:,:,k,q) = elem(ie)%derived%FQ(:,:,k,q)*elem(ie)%spheremp(:,:)
          end do
       end do
       do q = 1,2
          do k = 1,nlev
             elem(ie)%derived%FM(:,:,k,q) = elem(ie)%derived%FM(:,:,k,q)*elem(ie)%spheremp(:,:)
          end do
       end do
       elem(ie)%derived%FT(:,:,k) = elem(ie)%derived%FT(:,:,k)*elem(ie)%spheremp(:,:)
    end do
    do ie = nets, nete
       call edgeVpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FQ, qsize*nlev, 0, qsize*nlev)
    end do
    call bndry_exchangeV(hybrid, edge_g)
    do ie = nets, nete
       call edgeVunpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FQ, qsize*nlev, 0, qsize*nlev)
    end do
    do ie = nets, nete
       call edgeVpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FM, 2*nlev, 0, 3*nlev)
       call edgeVpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FT, nlev, 2*nlev, 3*nlev)
    end do
    call bndry_exchangeV(hybrid, edge_g)
    do ie = nets, nete
       call edgeVunpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FM, 2*nlev, 0, 3*nlev)
       call edgeVunpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%derived%FT, nlev, 2*nlev, 3*nlev)
    end do
    do ie = nets, nete
       do q = 1,qsize
          do k = 1,nlev
             elem(ie)%derived%FQ(:,:,k,q) = elem(ie)%derived%FQ(:,:,k,q)*elem(ie)%rspheremp(:,:)
          end do
       end do
       do q = 1,2
          do k = 1,nlev
             elem(ie)%derived%FM(:,:,k,q) = elem(ie)%derived%FM(:,:,k,q)*elem(ie)%rspheremp(:,:)
          end do
       end do
       elem(ie)%derived%FT(:,:,k) = elem(ie)%derived%FT(:,:,k)*elem(ie)%rspheremp(:,:)
    end do
  end subroutine gfr_f2g_dss

  subroutine gfr_g_make_nonnegative(gll_metdet, g)
    real(kind=real_kind), intent(in) :: gll_metdet(:,:)
    real(kind=real_kind), intent(inout) :: g(:,:,:)

    integer :: k, i, j
    real(kind=real_kind) :: nmass, spheremp(np,np), w(np,np)

    spheremp = gfr%w_gg*gll_metdet
    do k = 1, size(g,3)
       nmass = zero
       do j = 1,np
          do i = 1,np
             if (g(i,j,k) < zero) then
                nmass = nmass + spheremp(i,j)*g(i,j,k)
                g(i,j,k) = zero
                w(i,j) = zero
             else
                w(i,j) = spheremp(i,j)
             end if
          end do
       end do
       if (nmass == zero) cycle
       w = (w/sum(w))/spheremp
       g(:,:,k) = g(:,:,k) + w*nmass
    end do
  end subroutine gfr_g_make_nonnegative

  subroutine gfr_get_latlon(ie, i, j, lat, lon)
    integer, intent(in) :: ie, i, j
    real(kind=real_kind), intent(out) :: lat, lon

    lat = gfr%spherep_f(i,j,ie)%lat
    lon = gfr%spherep_f(i,j,ie)%lon
  end subroutine gfr_get_latlon

  ! d suffix means the inputs, outputs are densities.
  subroutine gfr_g2f_remapd(gfr, gll_metdet, fv_metdet, g, f)
    type (GllFvRemap_t), intent(in) :: gfr
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), fv_metdet(:,:), g(:,:)
    real(kind=real_kind), intent(out) :: f(:,:)

    integer :: gi, gj, fi, fj
    real(kind=real_kind) :: accum

    do fj = 1,gfr%nphys
       do fi = 1,gfr%nphys
          accum = zero
          do gj = 1,np
             do gi = 1,np
                accum = accum + gfr%M_gf(gi,gj,fi,fj)*g(gi,gj)*gll_metdet(gi,gj)
             end do
          end do
          f(fi,fj) = accum/(gfr%w_ff(fi,fj)*fv_metdet(fi,fj))
       end do
    end do
  end subroutine gfr_g2f_remapd

  subroutine gfr_f2g_remapd(gfr, gll_metdet, fv_metdet, f, g)
    type (GllFvRemap_t), intent(in) :: gfr
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), fv_metdet(:,:), f(:,:)
    real(kind=real_kind), intent(out) :: g(:,:)

    integer :: gi, gj, fi, fj
    real(kind=real_kind) :: accum

    do gj = 1,np
       do gi = 1,np
          accum = zero
          do fj = 1,gfr%nphys
             do fi = 1,gfr%nphys
                accum = accum + gfr%f2g_remapd(fi,fj,gi,gj)*f(fi,fj)*fv_metdet(fi,fj)
             end do
          end do
          g(gi,gj) = accum/gll_metdet(gi,gj)
       end do
    end do
  end subroutine gfr_f2g_remapd

  subroutine gfr_reconstructd_nphys1(gfr, gll_metdet, g)
    type (GllFvRemap_t), intent(in) :: gfr
    real(kind=real_kind), intent(in) :: gll_metdet(:,:)
    real(kind=real_kind), intent(inout) :: g(:,:)

    real(kind=real_kind) :: accum, wrk(2,2)
    integer :: fi, fj, gi, gj

    !assume np = 2

    ! Use just the corner points. The idea is that the non-corner points have
    ! not interacted with the corner points. Thus, the corners behave as though
    ! everything has been done on a GLL np=2 grid. In particular, this means
    ! that using just the corner values now is still mass conserving.
    wrk(1,1) = gll_metdet(1 ,1 )*g(1 ,1 )
    wrk(2,1) = gll_metdet(np,1 )*g(np,1 )
    wrk(1,2) = gll_metdet(1 ,np)*g(1 ,np)
    wrk(2,2) = gll_metdet(np,np)*g(np,np)

    do fj = 1,np
       do fi = 1,np
          accum = zero
          do gj = 1,gfr%npi
             do gi = 1,gfr%npi
                accum = accum + gfr%interp(gi,gj,fi,fj)*wrk(gi,gj)
             end do
          end do
          g(fi,fj) = accum/gll_metdet(fi,fj)
       end do
    end do    
  end subroutine gfr_reconstructd_nphys1

  subroutine limiter_clip_and_sum(n, spheremp, qmin, qmax, dp, Qdp)
    use kinds, only: real_kind

    integer, intent(in) :: n
    real (kind=real_kind), intent(inout) :: qmin, qmax, Qdp(:,:)
    real (kind=real_kind), intent(in) :: spheremp(:,:), dp(:,:)

    integer :: k1, i, j
    logical :: modified
    real(kind=real_kind) :: addmass, mass, sumc, den
    real(kind=real_kind) :: x(n*n), c(n*n), v(n*n)

    k1 = 1
    do j = 1, n
       do i = 1, n
          c(k1) = spheremp(i,j)*dp(i,j)
          x(k1) = Qdp(i,j)/dp(i,j)
          k1 = k1+1
       enddo
    enddo

    sumc = sum(c)
    mass = sum(c*x)
    ! This should never happen, but if it does, don't limit.
    if (sumc <= 0) return
    ! Relax constraints to ensure limiter has a solution; this is only needed
    ! if running with the SSP CFL>1 or due to roundoff errors.
    if (mass < qmin*sumc) then
       qmin = mass / sumc
    endif
    if (mass > qmax*sumc) then
       qmax = mass / sumc
    endif

    addmass = zero

    ! Clip.
    modified = .false.
    do k1 = 1, n*n
       if (x(k1) > qmax) then
          modified = .true.
          addmass = addmass + (x(k1) - qmax)*c(k1)
          x(k1) = qmax
       elseif (x(k1) < qmin) then
          modified = .true.
          addmass = addmass + (x(k1) - qmin)*c(k1)
          x(k1) = qmin
       end if
    end do
    if (.not. modified) return

    if (addmass /= zero) then
       ! Determine weights.
       if (addmass > zero) then
          v(:) = qmax - x(:)
       else
          v(:) = x(:) - qmin
       end if
       den = sum(v*c)
       if (den > zero) then
          ! Update.
          x(:) = x(:) + (addmass/den)*v(:)
       end if
    end if

    k1 = 1
    do j = 1,n
       do i = 1,n
          Qdp(i,j) = x(k1)*dp(i,j)
          k1 = k1+1
       end do
    end do
  end subroutine limiter_clip_and_sum

  ! --- testing ---
  ! Everything below is for testing.

  subroutine set_ps_Q(elem, nets, nete, timeidx, qidx, nlev)
    use coordinate_systems_mod, only: cartesian3D_t, change_coordinates

    type (element_t), intent(inout) :: elem(:)
    integer, intent(in) :: nets, nete, timeidx, qidx, nlev

    integer :: ie, i, j, k
    type (cartesian3D_t) :: p
    real(kind=real_kind) :: q

    do ie = nets, nete
       do j = 1,np
          do i = 1,np
             p = change_coordinates(elem(ie)%spherep(i,j))
             elem(ie)%state%ps_v(i,j,timeidx) = &
                  1.0d3*(1 + 0.05*sin(2*p%x+0.5)*sin(p%y+1.5)*sin(3*p%z+2.5))
             q = 0.5*(1 + sin(3*p%x)*sin(3*p%y)*sin(4*p%z))
             do k = 1,nlev
                elem(ie)%state%Q(i,j,k,qidx) = q
             end do
          end do
       end do
    end do
  end subroutine set_ps_Q

  subroutine check(gfr, hybrid, elem, nets, nete, verbose)
    use dimensions_mod, only: nlev, qsize
    use edge_mod, only: edge_g, edgevpack_nlyr, edgevunpack_nlyr
    use bndry_mod, only: bndry_exchangev
    use viscosity_mod, only: neighbor_minmax
    use parallel_mod, only: global_shared_buf, global_shared_sum
    use global_norms_mod, only: wrap_repro_sum
    use reduction_mod, only: ParallelMin, ParallelMax
    use prim_advection_base, only: edgeAdvQminmax

    type (GllFvRemap_t), intent(in) :: gfr
    type (hybrid_t), intent(in) :: hybrid
    type (element_t), intent(inout) :: elem(:)
    integer, intent(in) :: nets, nete
    logical, intent(in) :: verbose

    real(kind=real_kind) :: a, b, rd, x, y, f0(np,np), f1(np,np), g(np,np), &
         wrk(np,np), wrk3(np,np,1), qmin, qmax, qmin1, qmax1, mass0, mass1
    integer :: nf, ie, i, j, iremap, info, ilimit, sign
    real(kind=real_kind), allocatable :: Qdp_fv(:,:,:), ps_v_fv(:,:,:), &
         qmins(:,:,:), qmaxs(:,:,:)
    logical :: limit

    nf = gfr%nphys

    if (hybrid%masterthread) then
       print *, 'gfr> npi', gfr%npi, 'nphys', nf
       if (verbose) then
          print *, 'gfr> w_ff', nf, gfr%w_ff(:nf, :nf)
          print *, 'gfr> w_gg', np, gfr%w_gg(:np, :np)
          print *, 'gfr> w_sgsg', gfr%npi, gfr%w_sgsg(:gfr%npi, :gfr%npi)
          print *, 'gfr> M_gf', np, nf, gfr%M_gf(:np, :np, :nf, :nf)
          print *, 'gfr> M_sgf', gfr%npi, nf, gfr%M_sgf(:gfr%npi, :gfr%npi, :nf, :nf)
          print *, 'gfr> interp', gfr%npi, np, gfr%interp(:gfr%npi, :gfr%npi, :np, :np)
          print *, 'gfr> f2g_remapd', np, nf, gfr%f2g_remapd(:nf,:nf,:,:)
       end if
    end if

    ! Cell-local correctness checks
    do ie = nets, nete
       ! Check that areas match.
       a = sum(elem(ie)%metdet * gfr%w_gg)
       b = sum(gfr%fv_metdet(:,:,ie) * gfr%w_ff(:nf, :nf))
       rd = abs(b - a)/abs(a)
       if (rd /= rd .or. rd > 1e-15) print *, 'gfr> area', ie, rd

       ! Check that FV -> GLL -> FV recovers the original FV values exactly
       ! (with no DSS and no limiter).
       do j = 1,nf
          x = real(j-1, real_kind)/real(nf, real_kind)
          do i = 1,nf
             y = real(i-1, real_kind)/real(nf, real_kind)
             f0(i,j) = real(ie)/nelemd + x*x + ie*x + cos(ie + 4.2*y)
          end do
       end do
       call gfr_f2g_remapd(gfr, elem(ie)%metdet, gfr%fv_metdet(:,:,ie), f0, g)
       call gfr_g2f_remapd(gfr, elem(ie)%metdet, gfr%fv_metdet(:,:,ie), g, f1)
       wrk(:nf,:nf) = gfr%w_ff(:nf,:nf)*gfr%fv_metdet(:nf,:nf,ie)
       a = sum(wrk(:nf,:nf)*abs(f1(:nf,:nf) - f0(:nf,:nf)))
       b = sum(wrk(:nf,:nf)*abs(f0(:nf,:nf)))
       rd = a/b
       if (rd /= rd .or. rd > 1e-15) print *, 'gfr> recover', ie, a, b, rd

       ! Check gfr_g_make_nonnegative.
       sign = 1
       do j = 1,np
          do i = 1,np
             wrk3(i,j,1) = one + sign*i*j
             sign = -sign
          end do
       end do
       mass0 = sum(elem(ie)%spheremp*wrk3(:,:,1))
       call gfr_g_make_nonnegative(elem(ie)%metdet, wrk3)
       mass1 = sum(elem(ie)%spheremp*wrk3(:,:,1))
       rd = (mass1 - mass0)/mass0
       if (rd /= rd .or. rd > 1e-15) print *, 'gfr> nonnegative', ie, mass0, mass1
    end do

    ! For convergence testing. Run this testing routine with a sequence of ne
    ! values and plot log l2 error vs log ne.
    allocate(Qdp_fv(gfr%nphys, gfr%nphys, nets:nete), ps_v_fv(gfr%nphys, gfr%nphys, nets:nete))
    allocate(qmins(nlev,qsize,nets:nete), qmaxs(nlev,qsize,nets:nete))
    do ilimit = 0,1
       limit = ilimit > 0
       ! 0. Create synthetic q and ps_v.
       call set_ps_Q(elem, nets, nete, 1, 1, nlev)
       call set_ps_Q(elem, nets, nete, 2, 2, nlev)
       do iremap = 1,1
          ! 1. GLL -> FV
          do ie = nets, nete !TODO move to own routine
             call gfr_g2f_remapd(gfr, elem(ie)%metdet, gfr%fv_metdet(:,:,ie), &
                  elem(ie)%state%ps_v(:,:,1)*elem(ie)%state%Q(:,:,1,1), Qdp_fv(:,:,ie))
             if (limit) then
                call gfr_g2f_remapd(gfr, elem(ie)%metdet, gfr%fv_metdet(:,:,ie), &
                     elem(ie)%state%ps_v(:,:,1), ps_v_fv(:,:,ie))
                qmin = minval(elem(ie)%state%Q(:,:,1,1))
                qmax = maxval(elem(ie)%state%Q(:,:,1,1))
                call limiter_clip_and_sum(nf, gfr%w_ff(:nf,:nf)*gfr%fv_metdet(:nf,:nf,ie), &
                     qmin, qmax, ps_v_fv(:,:,ie), Qdp_fv(:,:,ie))
             end if
          end do
          ! 2. FV -> GLL
          if (limit) then
             ! 2a. Get q bounds
             do ie = nets, nete
                wrk(:nf,:nf) = Qdp_fv(:nf,:nf,ie)/ps_v_fv(:nf,:nf,ie)
                qmins(1,1,ie) = minval(wrk(:nf,:nf))
                qmaxs(1,1,ie) = maxval(wrk(:nf,:nf))
             end do
             ! 2b. Halo exchange q bounds.
             call neighbor_minmax(hybrid, edgeAdvQminmax, nets, nete, qmins, qmaxs)
          endif
          ! 2c. Remap
          do ie = nets, nete !TODO move to own routine
             call gfr_f2g_remapd(gfr, elem(ie)%metdet, gfr%fv_metdet(:,:,ie), &
                  Qdp_fv(:,:,ie), elem(ie)%state%Q(:,:,1,1))
             if (limit) then
                qmin = qmins(1,1,ie)
                qmax = qmaxs(1,1,ie)
                call limiter_clip_and_sum(np, elem(ie)%spheremp, & ! same as w_gg*gll_metdet
                     qmin, qmax, elem(ie)%state%ps_v(:,:,1), elem(ie)%state%Q(:,:,1,1))
             end if
             elem(ie)%state%Q(:,:,1,1) = elem(ie)%state%Q(:,:,1,1)/elem(ie)%state%ps_v(:,:,1)
          end do
          ! 3. DSS
          do ie = nets, nete
             elem(ie)%state%Q(:,:,1,1) = &
                  elem(ie)%state%ps_v(:,:,1)*elem(ie)%state%Q(:,:,1,1)*elem(ie)%spheremp(:,:)
             call edgeVpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%state%Q(:,:,1,1), 1, 0, 1)
          end do
          call bndry_exchangeV(hybrid, edge_g)
          do ie = nets, nete
             call edgeVunpack_nlyr(edge_g, elem(ie)%desc, elem(ie)%state%Q(:,:,1,1), 1, 0, 1)
             elem(ie)%state%Q(:,:,1,1) = &
                  (elem(ie)%state%Q(:,:,1,1)*elem(ie)%rspheremp(:,:))/elem(ie)%state%ps_v(:,:,1)
          end do
          ! 4. pg1 special case
          if (gfr%nphys == 1 .and. .not. limit) then
             do ie = nets, nete !TODO move to own routine
                elem(ie)%state%Q(:,:,1,1) = elem(ie)%state%Q(:,:,1,1)*elem(ie)%state%ps_v(:,:,1)
                call gfr_reconstructd_nphys1(gfr, elem(ie)%metdet, elem(ie)%state%Q(:,:,1,1))
                elem(ie)%state%Q(:,:,1,1) = elem(ie)%state%Q(:,:,1,1)/elem(ie)%state%ps_v(:,:,1)
             end do
          end if
       end do
       ! 5. Compute error.
       qmin = two
       qmax = -two
       qmin1 = two
       qmax1 = -two
       do ie = nets, nete
          wrk = gfr%w_gg(:,:)*elem(ie)%metdet(:,:)
          ! L2 on q. Might switch to q*ps_v.
          global_shared_buf(ie,1) = &
               sum(wrk*(elem(ie)%state%Q(:,:,1,1) - elem(ie)%state%Q(:,:,1,2))**2)
          global_shared_buf(ie,2) = &
               sum(wrk*elem(ie)%state%Q(:,:,1,2)**2)
          ! Mass conservation.
          wrk = wrk*elem(ie)%state%ps_v(:,:,1)
          global_shared_buf(ie,3) = sum(wrk*elem(ie)%state%Q(:,:,1,2))
          global_shared_buf(ie,4) = sum(wrk*elem(ie)%state%Q(:,:,1,1))
          qmin = min(qmin, minval(elem(ie)%state%Q(:,:,1,1)))
          qmin1 = min(qmin1, minval(elem(ie)%state%Q(:,:,1,2)))
          qmax = max(qmax, maxval(elem(ie)%state%Q(:,:,1,1)))
          qmax1 = max(qmax1, maxval(elem(ie)%state%Q(:,:,1,2)))
       end do
       call wrap_repro_sum(nvars=4, comm=hybrid%par%comm)
       qmin = ParallelMin(qmin, hybrid)
       qmax = ParallelMax(qmax, hybrid)
       qmin1 = ParallelMin(qmin1, hybrid)
       qmax1 = ParallelMax(qmax1, hybrid)
       if (hybrid%masterthread) then
          print *, 'gfr> limit', ilimit
          rd = sqrt(global_shared_sum(1)/global_shared_sum(2))
          print *, 'gfr> l2  ', rd
          rd = (global_shared_sum(4) - global_shared_sum(3))/global_shared_sum(3)
          print *, 'gfr> mass', rd
          print *, 'gfr> limit', min(zero, qmin - qmin1), max(zero, qmax - qmax1)
       end if
    end do
    deallocate(Qdp_fv, ps_v_fv, qmins, qmaxs)
  end subroutine check

  subroutine gfr_test(hybrid, nets, nete, hvcoord, deriv, elem)
    use derivative_mod, only: derivative_t
    use hybvcoord_mod, only: hvcoord_t

    type (hybrid_t), intent(in) :: hybrid
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(inout) :: elem(:)
    type (hvcoord_t) , intent(in) :: hvcoord
    integer, intent(in) :: nets, nete

    integer :: nphys

    do nphys = 1, np
       ! This is meant to be called before threading starts.
       if (hybrid%ithr == 0) call gfr_init(hybrid, elem, nphys)
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif

       call check(gfr, hybrid, elem, nets, nete, .false.)

       ! This is meant to be called after threading ends.
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif
       if (hybrid%ithr == 0) call gfr_finish()
    end do
#ifdef HORIZ_OPENMP
    !$omp barrier
#endif
  end subroutine gfr_test
end module gllfvremap_mod
