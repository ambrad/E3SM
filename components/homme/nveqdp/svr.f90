module svr_mod
  implicit none

  integer, parameter :: vert_remap_q_alg = 2, real_kind=8, nlev = 72

contains
  subroutine remap_Q_ppm(Qdp,nx,qsize,dp1,dp2)
    ! remap 1 field
    ! input:  Qdp   field to be remapped (NOTE: MASS, not MIXING RATIO)
    !         dp1   layer thickness (source)
    !         dp2   layer thickness (target)
    !
    ! output: remaped Qdp, conserving mass
    !
    integer,intent(in) :: nx,qsize
    real (kind=real_kind), intent(inout) :: Qdp(nx,nx,nlev,qsize)
    real (kind=real_kind), intent(in) :: dp1(nx,nx,nlev),dp2(nx,nx,nlev)
    ! Local Variables
    integer, parameter :: gs = 2                              !Number of cells to place in the ghost region
    real(kind=real_kind), dimension(       nlev+2 ) :: pio    !Pressure at interfaces for old grid
    real(kind=real_kind), dimension(       nlev+1 ) :: pin    !Pressure at interfaces for new grid
    real(kind=real_kind), dimension(       nlev+1 ) :: masso  !Accumulate mass up to each interface
    real(kind=real_kind), dimension(  1-gs:nlev+gs) :: ao     !Tracer value on old grid
    real(kind=real_kind), dimension(  1-gs:nlev+gs) :: dpo    !change in pressure over a cell for old grid
    real(kind=real_kind), dimension(  1-gs:nlev+gs) :: dpn    !change in pressure over a cell for old grid
    real(kind=real_kind), dimension(3,     nlev   ) :: coefs  !PPM coefficients within each cell
    real(kind=real_kind), dimension(       nlev   ) :: z1, z2
    real(kind=real_kind) :: ppmdx(10,0:nlev+1)  !grid spacings
    real(kind=real_kind) :: mymass, massn1, massn2
    integer :: i, j, k, q, kk, kid(nlev)

    do j = 1 , nx
       do i = 1 , nx

          pin(1)=0
          pio(1)=0
          do k=1,nlev
             dpn(k)=dp2(i,j,k)
             dpo(k)=dp1(i,j,k)
             pin(k+1)=pin(k)+dpn(k)
             pio(k+1)=pio(k)+dpo(k)
          enddo

          pio(nlev+2) = pio(nlev+1) + 1.  !This is here to allow an entire block
                                          !of k threads to run in the remapping
                                          !phase.  It makes sure there's an old
                                          !interface value below the domain that
                                          !is larger.
          pin(nlev+1) = pio(nlev+1)       !The total mass in a column does not
                                          !change.  Therefore, the pressure of
                                          !that mass cannot either.  Fill in the
                                          !ghost regions with mirrored
                                          !values. if vert_remap_q_alg is
                                          !defined, this is of no consequence.
          do k = 1 , gs
             dpo(1   -k) = dpo(       k)
             dpo(nlev+k) = dpo(nlev+1-k)
          enddo

          !Compute remapping intervals once for all tracers. Find the old grid
          !cell index in which the k-th new cell interface resides. Then
          !integrate from the bottom of that old cell to the new interface
          !location. In practice, the grid never deforms past one cell, so the
          !search can be simplified by this. Also, the interval of integration
          !is usually of magnitude close to zero or close to dpo because of
          !minimial deformation.  Numerous tests confirmed that the bottom and
          !top of the grids match to machine precision, so I set them equal to
          !each other.
          do k = 1 , nlev
             kk = k  !Keep from an order n^2 search operation by assuming the old cell index is close.
             !Find the index of the old grid cell in which this new cell's bottom interface resides.
             do while ( pio(kk) <= pin(k+1) )
                kk = kk + 1
             enddo
             kk = kk - 1                   !kk is now the cell index we're integrating over.
             if (kk == nlev+1) kk = nlev   !This is to keep the indices in bounds.
             !Top bounds match anyway, so doesn't matter what coefficients are used
             kid(k) = kk                   !Save for reuse
             z1(k) = -0.5D0                !This remapping assumes we're
                                           !starting from the left interface of
                                           !an old grid cell In fact, we're
                                           !usually integrating very little or
                                           !almost all of the cell in question
             !PPM interpolants are normalized to an independent
             z2(k) = ( pin(k+1) - ( pio(kk) + pio(kk+1) ) * 0.5 ) / dpo(kk)  
             !coordinate domain [-0.5,0.5].
          enddo

          !This turned out a big optimization, remembering that only parts of
          !the PPM algorithm depends on the data, namely the limiting. So
          !anything that depends only on the grid is pre-computed outside the
          !tracer loop.
          ppmdx(:,:) = compute_ppm_grids( dpo )

          !From here, we loop over tracers for only those portions which depend
          !on tracer data, which includes PPM limiting and mass accumulation
          do q = 1 , qsize
             !Accumulate the old mass up to old grid cell interface locations to
             !simplify integration during remapping. Also, divide out the grid
             !spacing so we're working with actual tracer values and can
             !conserve mass. The option for ifndef ZEROHORZ I believe is there
             !to ensure tracer consistency for an initially uniform field. I
             !copied it from the old remap routine.
             masso(1) = 0.
             do k = 1 , nlev
                ao(k) = Qdp(i,j,k,q)
                masso(k+1) = masso(k) + ao(k) !Accumulate the old mass. This
                                              !will simplify the remapping
                ao(k) = ao(k) / dpo(k)        !Divide out the old grid spacing
                                              !because we want the tracer mixing
                                              !ratio, not mass.
             enddo
             !Fill in ghost values. Ignored if vert_remap_q_alg == 2
             do k = 1 , gs
                if (vert_remap_q_alg == 3) then
                   ao(1   -k) = ao(1)
                   ao(nlev+k) = ao(nlev)
                elseif (vert_remap_q_alg == 2 .or. vert_remap_q_alg == 1) then   !Ignored if vert_remap_q_alg == 2
                   ao(1   -k) = ao(       k)
                   ao(nlev+k) = ao(nlev+1-k)
                endif
             enddo
             !Compute monotonic and conservative PPM reconstruction over every cell
             coefs(:,:) = compute_ppm( ao , ppmdx )
             !Compute tracer values on the new grid by integrating from the old
             !cell bottom to the new cell interface to form a new grid mass
             !accumulation. Taking the difference between accumulation at
             !successive interfaces gives the mass inside each cell. Since Qdp
             !is supposed to hold the full mass this needs no normalization.
             massn1 = 0.
             do k = 1 , nlev
                kk = kid(k)
                massn2 = masso(kk) + integrate_parabola( coefs(:,kk) , z1(k) , z2(k) ) * dpo(kk)
                Qdp(i,j,k,q) = massn2 - massn1
                massn1 = massn2
             enddo
          enddo
       enddo
    enddo
  end subroutine remap_Q_ppm

  !THis compute grid-based coefficients from Collela & Woodward 1984.
  function compute_ppm_grids( dx )   result(rslt)
    real(kind=real_kind), intent(in) :: dx(-1:nlev+2)  !grid spacings
    real(kind=real_kind)             :: rslt(10,0:nlev+1)  !grid spacings
    integer :: j
    integer :: indB, indE

    !Calculate grid-based coefficients for stage 1 of compute_ppm
    do j = 0 , nlev+1
       rslt( 1,j) = dx(j) / ( dx(j-1) + dx(j) + dx(j+1) )
       rslt( 2,j) = ( 2.*dx(j-1) + dx(j) ) / ( dx(j+1) + dx(j) )
       rslt( 3,j) = ( dx(j) + 2.*dx(j+1) ) / ( dx(j-1) + dx(j) )
    enddo

    !Caculate grid-based coefficients for stage 2 of compute_ppm
    do j = 0 , nlev
       rslt( 4,j) = dx(j) / ( dx(j) + dx(j+1) )
       rslt( 5,j) = 1. / sum( dx(j-1:j+2) )
       rslt( 6,j) = ( 2. * dx(j+1) * dx(j) ) / ( dx(j) + dx(j+1 ) )
       rslt( 7,j) = ( dx(j-1) + dx(j  ) ) / ( 2. * dx(j  ) + dx(j+1) )
       rslt( 8,j) = ( dx(j+2) + dx(j+1) ) / ( 2. * dx(j+1) + dx(j  ) )
       rslt( 9,j) = dx(j  ) * ( dx(j-1) + dx(j  ) ) / ( 2.*dx(j  ) +    dx(j+1) )
       rslt(10,j) = dx(j+1) * ( dx(j+1) + dx(j+2) ) / (    dx(j  ) + 2.*dx(j+1) )
    enddo
  end function compute_ppm_grids

  !This computes a limited parabolic interpolant using a net 5-cell stencil, but the stages of computation are broken up into 3 stages
  function compute_ppm( a , dx )    result(coefs)
    real(kind=real_kind), intent(in) :: a    (    -1:nlev+2)  !Cell-mean values
    real(kind=real_kind), intent(in) :: dx   (10,  0:nlev+1)  !grid spacings
    real(kind=real_kind) ::             coefs(0:2,   nlev  )  !PPM coefficients (for parabola)
    real(kind=real_kind) :: ai (0:nlev  )                     !fourth-order accurate, then limited interface values
    real(kind=real_kind) :: dma(0:nlev+1)                     !An expression from Collela's '84 publication
    real(kind=real_kind) :: da                                !Ditto
    ! Hold expressions based on the grid (which are cumbersome).
    real(kind=real_kind) :: dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10
    real(kind=real_kind) :: al, ar                            !Left and right interface values for cell-local limiting
    integer :: j
    integer :: indB, indE

    ! Stage 1: Compute dma for each cell, allowing a 1-cell ghost stencil below and above the domain
    do j = 0 , nlev+1
       da = dx(1,j) * ( dx(2,j) * ( a(j+1) - a(j) ) + dx(3,j) * ( a(j) - a(j-1) ) )
       dma(j) = minval( (/ abs(da) , 2. * abs( a(j) - a(j-1) ) , 2. * abs( a(j+1) - a(j) ) /) ) * sign(1.D0,da)
       if ( ( a(j+1) - a(j) ) * ( a(j) - a(j-1) ) <= 0. ) dma(j) = 0.
    enddo

    ! Stage 2: Compute ai for each cell interface in the physical domain (dimension nlev+1)
    do j = 0 , nlev
       ai(j) = a(j) + dx(4,j) * ( a(j+1) - a(j) ) + dx(5,j) * ( dx(6,j) * ( dx(7,j) - dx(8,j) ) &
            * ( a(j+1) - a(j) ) - dx(9,j) * dma(j+1) + dx(10,j) * dma(j) )
    enddo

    ! Stage 3: Compute limited PPM interpolant over each cell in the physical
    ! domain (dimension nlev) using ai on either side and ao within the cell.
    do j = 1 , nlev
       al = ai(j-1)
       ar = ai(j  )
       if ( (ar - a(j)) * (a(j) - al) <= 0. ) then
          al = a(j)
          ar = a(j)
       endif
       if ( (ar - al) * (a(j) - (al + ar)/2.) >  (ar - al)**2/6. ) al = 3.*a(j) - 2. * ar
       if ( (ar - al) * (a(j) - (al + ar)/2.) < -(ar - al)**2/6. ) ar = 3.*a(j) - 2. * al
       !Computed these coefficients from the edge values and cell mean in Maple. Assumes normalized coordinates: xi=(x-x0)/dx
       coefs(0,j) = 1.5 * a(j) - ( al + ar ) / 4.
       coefs(1,j) = ar - al
       ! coefs(2,j) = -6. * a(j) + 3. * ( al + ar )
       coefs(2,j) = 3. * (-2. * a(j) + ( al + ar ))
    enddo

    !If vert_remap_q_alg == 2, use piecewise constant in the boundaries, and
    !don't use ghost cells.
    if (vert_remap_q_alg == 2) then
       coefs(0,1:2) = a(1:2)
       coefs(1:2,1:2) = 0.
       coefs(0,nlev-1:nlev) = a(nlev-1:nlev)
       coefs(1:2,nlev-1:nlev) = 0.D0
    endif
  end function compute_ppm

  !Simple function computes the definite integral of a parabola in normalized
  !coordinates, xi=(x-x0)/dx, given two bounds. Make sure this gets inlined
  !during compilation.
  function integrate_parabola( a , x1 , x2 )    result(mass)
    real(kind=real_kind), intent(in) :: a(0:2)  !Coefficients of the parabola
    real(kind=real_kind), intent(in) :: x1      !lower domain bound for integration
    real(kind=real_kind), intent(in) :: x2      !upper domain bound for integration
    real(kind=real_kind)             :: mass
    mass = a(0) * (x2 - x1) + a(1) * (x2 * x2 - x1 * x1) / 0.2D1 + a(2) * (x2 * x2 * x2 - x1 * x1 * x1) / 0.3D1
  end function integrate_parabola
end module svr_mod

program main
  use svr_mod
  implicit none

#include "data.f90"
  call remap_Q_ppm(Qdp,1,1,dp1,dp2)
  print *, Qdp
end program main
