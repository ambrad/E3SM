#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

!todo
! - contravariant u,v
! - rho in u,v?
! - online coords
! - topo roughness
! - np4-np2 instead of np4-pg2

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
          fv_metdet(:,:,:) ! (nphys,nphys,nelemd)
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
       gfr_g2f_pressure, gfr_g2f_scalar, gfr_g2f_scalar_dp, gfr_g2f_mixing_ratio, &
       gfr_f2g_scalar, gfr_f2g_scalar_dp, gfr_f2g_mixing_ratio_a, &
       gfr_f2g_mixing_ratio_b, gfr_f2g_mixing_ratio_c, gfr_f2g_dss

contains

  subroutine gfr_init(nphys, elem)
    use parallel_mod, only: abortmp

    integer, intent(in) :: nphys
    type (element_t), intent(in) :: elem(nelemd)

    real(real_kind) :: R(npsq,nphys_max*nphys_max)

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

    allocate(gfr%fv_metdet(nphys,nphys,nelemd))
    call gfr_init_fv_metdet(elem, gfr)
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

    integer :: i,j

    do j = 1,nphys
       do i = 1,nphys
          w_ff(i,j) = two/real(nphys, real_kind)
       end do
    end do
  end subroutine gfr_init_w_ff

  subroutine gll_cleanup(gll)
    use quadrature_mod, only : quadrature_t

    type (quadrature_t), intent(inout) :: gll

    deallocate(gll%points, gll%weights)
  end subroutine gll_cleanup

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
    type (element_t), intent(in) :: elem(nelemd)
    type (GllFvRemap_t), intent(inout) :: gfr

    real (kind=real_kind) :: ones_f(gfr%nphys, gfr%nphys), ones_g(np,np)
    integer :: ie

    ones_f = one
    ones_g = one
    do ie = 1,nelemd
       call gfr_g2f_remapd(gfr, elem(ie)%metdet, ones_f, ones_g, gfr%fv_metdet(:,:,ie))
    end do
  end subroutine gfr_init_fv_metdet

  subroutine gfr_g2f_pressure(ie, gll_metdet, p_g, p_f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), p_g(:,:,:)
    real(kind=real_kind), intent(out) :: p_f(:,:,:)

    p_f = p_g
  end subroutine gfr_g2f_pressure

  subroutine gfr_g2f_scalar(ie, gll_metdet, g, f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), g(:,:,:)
    real(kind=real_kind), intent(out) :: f(:,:,:)

    f = g
  end subroutine gfr_g2f_scalar

  subroutine gfr_g2f_scalar_dp(ie, gll_metdet, dp_g, dp_f, g, f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_g(:,:,:), dp_f(:,:,:), g(:,:,:)
    real(kind=real_kind), intent(out) :: f(:,:,:)

    f = g
  end subroutine gfr_g2f_scalar_dp

  subroutine gfr_g2f_mixing_ratio(ie, gll_metdet, dp_g, dp_f, g, f)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_g(:,:,:), dp_f(:,:,:), g(:,:,:,:)
    real(kind=real_kind), intent(out) :: f(:,:,:,:)

    integer :: q, k

    do q = 1, size(g,4)
       f(:,:,:,q) = g(:,:,:,q)/dp_g
    end do
  end subroutine gfr_g2f_mixing_ratio

  subroutine gfr_f2g_scalar(ie, gll_metdet, f, g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), f(:,:,:)
    real(kind=real_kind), intent(out) :: g(:,:,:)

    g = f
  end subroutine gfr_f2g_scalar

  subroutine gfr_f2g_scalar_dp(ie, gll_metdet, dp_f, dp_g, f, g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_f(:,:,:), dp_g(:,:,:), f(:,:,:)
    real(kind=real_kind), intent(out) :: g(:,:,:)

    g = f
  end subroutine gfr_f2g_scalar_dp

  subroutine gfr_f2g_mixing_ratio_a(ie, gll_metdet, dp_f, dp_g, f, g)
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), dp_f(:,:,:), dp_g(:,:,:), f(:,:,:,:)
    real(kind=real_kind), intent(out) :: g(:,:,:,:)

    g = f
  end subroutine gfr_f2g_mixing_ratio_a

  subroutine gfr_f2g_mixing_ratio_b(hybrid, nets, nete, qmin, qmax)
    type (hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: nets, nete
    real(kind=real_kind), intent(inout) :: qmin(:,:,:), qmax(:,:,:)
  end subroutine gfr_f2g_mixing_ratio_b

  subroutine gfr_f2g_mixing_ratio_c(ie, gll_metdet, qmin, qmax, dp, q0, q_ten)
    ! Solve
    !     min norm(q_ten - q_ten*, 1)
    !      st dp'q_ten unchanged
    !         qmin <= q0 + q_ten <= qmax
    !TODO need to think about feasibility and safety problem
    integer, intent(in) :: ie
    real(kind=real_kind), intent(in) :: gll_metdet(:,:), qmin(:,:), qmax(:,:), &
         dp(:,:,:), q0(:,:,:,:)
    real(kind=real_kind), intent(inout) :: q_ten(:,:,:,:)
  end subroutine gfr_f2g_mixing_ratio_c

  subroutine gfr_f2g_dss(elem)
    type (element_t), intent(inout) :: elem(:)
  end subroutine gfr_f2g_dss

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
         wrk(np,np), qmin, qmax, qmin1, qmax1
    integer :: nf, ie, i, j, iremap, info, ilimit
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
    end do

    ! For convergence testing.
    allocate(Qdp_fv(gfr%nphys, gfr%nphys, nets:nete), ps_v_fv(gfr%nphys, gfr%nphys, nets:nete))
    allocate(qmins(nlev,qsize,nets:nete), qmaxs(nlev,qsize,nets:nete))
    do ilimit = 0,1
       limit = ilimit > 0
       !if (limit .and. nf == 1) cycle
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
       if (hybrid%ithr == 0) call gfr_init(nphys, elem)
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
