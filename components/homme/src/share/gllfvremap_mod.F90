module gllfvremap_mod
  ! API for high-order, shape-preseving FV <-> GLL remap.
  !
  ! AMB 2019/07 Initial

  use hybrid_mod, only: hybrid_t
  use kinds, only: real_kind
  use dimensions_mod, only: np, npsq, qsize, nelemd

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
          w_gg(np,np), &
          w_ff(nphys_max,nphys_max), &
          M_gf(np,np,nphys_max,nphys_max), &
          w_sgsg(np,np), &
          M_sgf(np,np,nphys_max,nphys_max), &
          R(npsq,nphys_max*nphys_max), &
          interp(nphys_max,nphys_max,np,np)
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

  subroutine gfr_init(nphys, elem)
    use parallel_mod, only: abortmp
    use element_mod, only: element_t

    integer, intent(in) :: nphys
    type (element_t), intent(in) :: elem(nelemd)

    if (nphys > np) then
       ! The FV -> GLL map is defined only if nphys <= np. If we ever are
       ! interested in the case of nphys > np, we will need to write a different
       ! map.
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

  subroutine gfr_init_fv_metdet(elem, gfr)
    use element_mod, only: element_t

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

    integer :: nf, nf2, npi, np2, gi, gj, fi, fj
    real(kind=real_kind) :: accum, wrk(np,np)

    nf = gfr%nphys
    nf2 = nf*nf
    npi = gfr%npi
    np2 = np*np

    ! Solve the constrained projection described in gfr_init_R:
    !     g = inv(M_sgsg) M_sgf inv(S) M_ff fv_metdet f
    wrk(:nf,:nf) = gfr%w_ff(:nf,:nf)*fv_metdet(:nf,:nf)*f(:nf,:nf)
    call dtrtrs('u', 't', 'n', nf2, 1, gfr%R, size(gfr%R,1), wrk, np2)
    call dtrtrs('u', 'n', 'n', nf2, 1, gfr%R, size(gfr%R,1), wrk, np2)
    g(:npi,:npi) = zero
    do fj = 1,gfr%nphys
       do fi = 1,gfr%nphys
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
             wrk(gi,gj) = g(gi,gj)/gfr%w_sgsg(gi,gj)
          end do
       end do 
       ! Interpolate from npi to np; if npi = np, this is just the Id matrix.
       do fj = 1,np
          do fi = 1,np
             accum = zero
             do gj = 1,npi
                do gi = 1,npi
                   accum = accum + gfr%interp(gi,gj,fi,fj)*wrk(gi,gj)
                end do
             end do
             ! Divide out the ref -> sphere jacobian.
             g(fi,fj) = accum/gll_metdet(fi,fj)
          end do
       end do
    else
       ! Finish the projection and divide out the ref -> sphere jacobian.
       do gj = 1,np
          do gi = 1,np
             g(gi,gj) = g(gi,gj)/(gfr%w_sgsg(gi,gj)*gll_metdet(gi,gj))
          end do
       end do
    end if
  end subroutine gfr_f2g_remapd

  subroutine gfr_print(gfr, elem, verbose)
    use element_mod, only: element_t

    type (GllFvRemap_t), intent(in) :: gfr
    type (element_t), intent(inout) :: elem(:)
    logical, intent(in) :: verbose

    real(kind=real_kind) :: a, b, rd
    integer :: ie

    if (verbose) then
       print *, 'npi', gfr%npi, 'nphys', gfr%nphys
       print *, 'w_ff', gfr%nphys, gfr%w_ff(:gfr%nphys, :gfr%nphys)
       print *, 'w_gg', np, gfr%w_gg(:np, :np)
       print *, 'w_sgsg', gfr%npi, gfr%w_sgsg(:gfr%npi, :gfr%npi)
       print *, 'M_gf', np, gfr%nphys, gfr%M_gf(:np, :np, :gfr%nphys, :gfr%nphys)
       print *, 'M_sgf', gfr%npi, gfr%nphys, gfr%M_sgf(:gfr%npi, :gfr%npi, :gfr%nphys, :gfr%nphys)
       print *, 'R', gfr%nphys, gfr%R(:gfr%nphys*gfr%nphys, :gfr%nphys*gfr%nphys)
       print *, 'interp', gfr%npi, np, gfr%interp(:gfr%npi, :gfr%npi, :np, :np)
    end if

    do ie = 1,nelemd
       a = sum(elem(ie)%metdet * gfr%w_gg)
       b = sum(gfr%fv_metdet(:,:,ie) * gfr%w_ff(:gfr%nphys, :gfr%nphys))
       rd = abs(b - a)/abs(a)
       if (rd > 1e-15) print *, ie, rd
    end do
  end subroutine gfr_print

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
       if (hybrid%masterthread) call gfr_init(nphys, elem)
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif

       call gfr_print(gfr, elem, .false.)

       ! This is meant to be called after threading ends.
       if (hybrid%masterthread) call gfr_finish()
#ifdef HORIZ_OPENMP
       !$omp barrier
#endif
    end do
  end subroutine gfr_test

end module gllfvremap_mod
