module seq_nlmap_mod

  !-----------------------------------------------------------------------------
  !
  ! Purpose: Apply a high-order linear map followed by a property restoring
  ! nonlinear global fixer. This module adds to the capabilities of seq_map_mod
  ! and extends its seq_map_avNormArr subroutine. The objective is to improve
  ! coarse-to-fine, conservative, monotone maps. Area-averaged (aave) maps
  ! produce grid imprint on the target fine grid because each coarse source cell
  ! has a constant field. This module applies a high-order linear map that
  ! produces smooth data on the fine target grid, then applies a nonlinear
  ! global fixer to restore conservation and bounds.
  !
  ! Following is a description of the algorithm and its properties,
  ! specializations of Alg 3.1 of the following reference for this use case:
  !   Bradley, A. M., Bosler, P. A., Guba, O., Taylor, M. A., and Barnett,
  !   G. A. (2019). Communication-efficient property preservation in tracer
  !   transport. SIAM Journal on Scientific Computing, 41(3), C161-C193.
  !
  ! Notation:
  !   A x is the matrix-vector product A times x.
  !   a x is a scalar times a vector.
  !   x'y is the dot product of x and y.
  !   non-zero-pattern(A) is a matrix of size A with 1 where A(i,j) /= 0
  !     and 0 otherwise.
  !   (In)equalities and / are applied element-wise.
  !   In practice:
  !     Am is the area-average conservative and monotone linear map.
  !     A  is the high-order linear map.
  !     f  is a vector of area fractions.
  ! On input:
  !   1. Am is a monotone and conservative linear operator, where s is the
  !      vector of src areas; t, tgt areas.
  !   2. non-zero-pattern(Am) in non-zero-pattern(A).
  !   3. 0 <= f <= 1.
  ! Definition of conservative:
  !   t'Am x = s'x
  ! Definition of monotone:
  !   l, u := bounds(Am, x)
  !     => l(i) := min(x where Am(i,:) /= 0)
  !        u(i) similarly
  !   l <= Am x <= u
  !   => 4. 0 <= A(i,j) <= 1
  ! With fractions:
  !   g := Am f
  !   ym := (Am (f x)) / g
  !   l, u := bounds(Am, x)
  !   Conservative:
  !     5. t'(g y) = t'(Am (f x)) = s'(f x)
  !   Monotone:
  !     c(j) := Am(i,j) f(j)
  !     0 <= c(j) <= 1 by 3 and 4.
  !     Let j vary over non-0s of Am(i,:).
  !     ym(i) = sum_j Am(i,j) f(j) x(j) / sum_j Am(i,j) f(j)
  !           = sum_j c(j) x(j) / sum_j c(j)
  !          <= max_j x(j) sum_j c(j) / sum_j c(j)
  !           = max_j x(j)
  !           = u(j)
  !     and similarly for l(j).
  !     => 6. l <= ym <= u
  ! Bounds:
  !   2 => 7. bounds(Am, x) in bounds(A, x)
  ! CAAS (Alg 3.1 in doi:10.1137/18M1165414) applied to this case:
  !   CAAS(Am, A, f, x)
  !     g := Am f
  !     M := t'(Am (f x))
  !     y0 := (A (f x)) / g
  !     l, u := bounds(A, x)
  !     y1 := max(l, min(u, y0))
  !     dM := M - t'(g y1)
  !     if dM >= 0
  !       w := u - y1
  !     else
  !       w := y1 - l
  !     y2 := y1 + (dM / (t g)'w) w
  !     return y2
  ! This algorithm has a nonempty constraint set because, by 5 and 6,
  !   ym := (Am (f x)) / g is conservative and in bounds.
  ! CAAS finds a solution if the constraint set is nonempty, proving y2 is
  ! conservative and in bounds. We can prove this in details, as follows:
  !   y2 is conservative:
  !     t'(g y2) = t'(g y1 + (dM / (t g)'w) g w)
  !              = M - dM + (dM / (t g)'w) t'(g w)
  !              = M
  !   y2 is in bounds:
  !     w >= 0 by construction.
  !     Assume dM >= 0; the other case is similar.
  !     8. M <= t'(g u):
  !       M = t'(Am (f x)) = t'(g ym) <= t'(g u) by 6 and 7
  !     dM <= (t g)'w:
  !       (t g)'w = (t g)'u - (t g)'y1 = (t g)'u + dM - M
  !         = dM + [(t g)'u - M] >= dM by 8
  !     => y2 = y1 + (dM / (t g)'w) w <= y1 + w = u => y2 <= u.
  ! In this proof, the crucial part is point 8; it is equivalent to the
  ! statement that the constraint is nonempty. This in turn makes
  !   0 <= (dM / (t g)'w) <= 1,
  ! permitting the final line to hold.
  !
  ! Usage:
  !   For any existing aave mapfile SRC2TGT_TMAPFILE, where T is 'F' or 'S',
  ! optionally provide a second, high-order map by specifying
  ! SRC2TGT_TMAPFILE_NONLINEAR. If T is 'F', then the fixer restores global
  ! mass, as defined by the SRC2TGT_TMAPFILE map, and local bounds. If T is 'S',
  ! then the fixer restores the bounds.
  !
  ! Author: A.M. Bradley, Mar,Apr-2023
  !
  !-----------------------------------------------------------------------------
  
  use shr_kind_mod     , only: R8 => SHR_KIND_R8, IN=>SHR_KIND_IN
  use shr_kind_mod     , only: CX => SHR_KIND_CX
  use shr_sys_mod
  use shr_const_mod
  use mct_mod
  use seq_comm_mct
  use component_type_mod
  use seq_map_type_mod
  use shr_reprosum_mod , only: shr_reprosum_calc
  use shr_infnan_mod   , only: shr_infnan_isnan, shr_infnan_isinf

  implicit none
  save
  private
#include <mpif.h>

  public :: seq_nlmap_check_matrices
  public :: seq_nlmap_avNormArr

contains

  subroutine seq_nlmap_avNormArr(mapper, av_i, av_o, norm_i, rList, norm)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   , intent(inout) :: mapper ! mapper
    type(mct_aVect) , intent(in)    :: av_i   ! input
    type(mct_aVect) , intent(inout) :: av_o   ! output
    real(r8)        , intent(in), optional :: norm_i(:) ! source "weight"
    character(len=*), intent(in), optional :: rList ! fields list
    logical         , intent(in), optional :: norm  ! normalize at end
    !
    ! Local variables
    !
    type(mct_aVect)        :: avp_i , avp_o, nl_avp_o
    integer(IN)            :: j,kf
    integer(IN)            :: lsize_i,lsize_o
    real(r8)               :: normval
    character(CX)          :: lrList,appnd
    logical                :: lnorm
    character(*),parameter :: subName = '(seq_nlmap_avNormArr) '
    character(len=*),parameter :: ffld = 'norm8wt'
    character(len=*), parameter :: afldname  = 'aream'
    character(len=128) :: msg
    logical :: nl_on, amroot, verbose
    integer(IN) :: mpicom, ierr, k, natt, nsum, nfld, kArea, lidata(2), gidata(2), i, n
    real(r8) :: tmp, area, lo, hi, y
    real(r8), allocatable, dimension(:) :: lmins, gmins, lmaxs, gmaxs, glbl_masses, gwts
    real(r8), allocatable, dimension(:,:) :: dof_masses, caas_wgt, oglims, lcl_lo, lcl_hi
    !-----------------------------------------------------

    nl_on = .true.
    verbose = .false.
    call seq_comm_setptrs(CPLID, mpicom=mpicom)
    amroot = seq_comm_iamroot(CPLID)

    lsize_i = mct_aVect_lsize(av_i)
    lsize_o = mct_aVect_lsize(av_o)

    lnorm = .true.
    if (present(norm)) then
       lnorm = norm
    endif

    if (present(norm_i)) then
       if (.not.lnorm) call shr_sys_abort(subname//' ERROR norm_i and norm = false')
       if (size(norm_i) /= lsize_i) call shr_sys_abort(subname//' ERROR size(norm_i) ne lsize_i')
    endif

    !--- create temporary avs for mapping ---

    if (lnorm .or. present(norm_i)) then
       appnd = ':'//ffld
    else
       appnd = ''
    endif
    if (present(rList)) then
       call mct_aVect_init(avp_i, rList=trim( rList)//trim(appnd), lsize=lsize_i)
       call mct_aVect_init(avp_o, rList=trim( rList)//trim(appnd), lsize=lsize_o)
       if (nl_on) then
          call mct_aVect_init(nl_avp_o, rList=trim( rList)//trim(appnd), lsize=lsize_o)
       end if
    else
       lrList = mct_aVect_exportRList2c(av_i)
       call mct_aVect_init(avp_i, rList=trim(lrList)//trim(appnd), lsize=lsize_i)
       lrList = mct_aVect_exportRList2c(av_o)
       call mct_aVect_init(avp_o, rList=trim(lrList)//trim(appnd), lsize=lsize_o)
       if (nl_on) then
          call mct_aVect_init(nl_avp_o, rList=trim(lrList)//trim(appnd), lsize=lsize_o)
       end if
    endif

    !--- copy av_i to avp_i and set ffld value to 1.0
    !--- then multiply all fields by norm_i if norm_i exists
    !--- this will do the right thing for the norm_i normalization

    call mct_aVect_copy(aVin=av_i, aVout=avp_i, VECTOR=mct_usevector)

    if (lnorm .or. present(norm_i)) then
       kf = mct_aVect_indexRA(avp_i,ffld)
       do j = 1,lsize_i
          avp_i%rAttr(kf,j) = 1.0_r8
       enddo

       if (present(norm_i)) then
          !$omp simd
          do j = 1,lsize_i
             avp_i%rAttr(:,j) = avp_i%rAttr(:,j)*norm_i(j)
          enddo
       endif
    endif

    !--- linear map ---

    if (mapper%esmf_map) then
       call shr_sys_abort(subname//' ERROR: esmf SMM not supported')
    else
       ! MCT based SMM
       call mct_sMat_avMult(avp_i, mapper%sMatp, avp_o, VECTOR=mct_usevector)
    endif

    !--- optional nonlinear map ---
    
    if (nl_on) then
       natt = size(avp_i%rAttr, 1)
       if (verbose) then
          if (amroot) then
             write(logunit, '(4A,3L2,I3)') 'nlmap> ', trim(mapper%nl_mapfile), ' ', &
                  trim(mapper%strategy), mapper%nl_conservative, lnorm, present(norm_i), natt
          end if
       end if
       do j = 1,lsize_i
          do k = 1,natt
             tmp = avp_i%rAttr(k,j)
             if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                write(logunit, '(a,a,l2,i5,i6,es23.15)') 'nlmap> inf/nan-1 ', &
                     trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
             end if
          end do
       end do
       if (lnorm) then
          kf = mct_aVect_indexRA(avp_i,ffld)
          if (kf /= natt) then
             call shr_sys_abort(subname// &
                  ' ERROR: Nonlinear map code expects weight field, '// &
                  'if present, to be in final AV column.')
          end if
          natt = natt - 1
       end if
       allocate(lcl_lo(natt,lsize_o), lcl_hi(natt,lsize_o))
       call sMat_avMult_and_calc_bounds(avp_i, mapper%nl_sMatp, lnorm, natt, &
            nl_avp_o, lcl_lo, lcl_hi)
       ! Mask high-order field against low-order. Occasionally an exact 0 in the
       ! low-order field will map the high-order field unnecessarily, but that's
       ! OK: it's a local reduction in order to one, not a wrong value.
       do j = 1,lsize_o
          do k = 1,natt
             if (avp_o%rAttr(k,j) == 0) then
                nl_avp_o%rAttr(k,j) = 0
                ! Need to set bounds to 0 so that the mass is not modified.
                lcl_lo(k,j) = 0
                lcl_hi(k,j) = 0
             end if
          end do
       end do
       if (mapper%nl_conservative) then
          ! Compute global bounds.
          allocate(lmins(natt), gmins(natt), lmaxs(natt), gmaxs(natt))
          lmins(:) =  1.e300_r8
          lmaxs(:) = -1.e300_r8
          do j = 1,lsize_o
             do k = 1,natt
                lmins(k) = min(lmins(k), lcl_lo(k,j))
                lmaxs(k) = max(lmaxs(k), lcl_hi(k,j))
             end do
          end do
          call mpi_allreduce(lmins, gmins, natt, MPI_DOUBLE_PRECISION, MPI_MIN, mpicom, ierr)
          call mpi_allreduce(lmaxs, gmaxs, natt, MPI_DOUBLE_PRECISION, MPI_MAX, mpicom, ierr)
          if (amroot .and. verbose) then
             do k = 1,natt
                write(logunit, '(a,i2,a,i2,es23.15,es23.15)') &
                     'nlmap> src-bnds ', k, '/', natt, gmins(k), gmaxs(k)
             end do
          end if
          if (verbose) then
             allocate(oglims(natt,2))
             lmins(:) =  1.e300_r8
             lmaxs(:) = -1.e300_r8
             do j = 1,lsize_o
                do k = 1,natt
                   tmp = nl_avp_o%rAttr(k,j)
                   lmins(k) = min(lmins(k), tmp)
                   lmaxs(k) = max(lmaxs(k), tmp)
                end do
             end do
             call mpi_allreduce(lmins, oglims(:,1), natt, MPI_DOUBLE_PRECISION, MPI_MIN, mpicom, ierr)
             call mpi_allreduce(lmaxs, oglims(:,2), natt, MPI_DOUBLE_PRECISION, MPI_MAX, mpicom, ierr)
             if (amroot) then
                do k = 1,natt
                   write(logunit, '(a,i2,a,i2,es23.15,es23.15)') &
                        'nlmap> pre-bnds ', k, '/', natt, oglims(k,1), oglims(k,2)
                end do
             end if
             deallocate(oglims)
          end if
          deallocate(lmins, lmaxs)
          ! Compute global mass in low-order and high-order fields.
          kArea = mct_aVect_indexRA(mapper%dom_cx_d%data, afldname)
          nsum = lsize_o
          nfld = 2*natt
          allocate(dof_masses(nsum,nfld), glbl_masses(nfld)) ! low- and high-order
          if (mct_aVect_lSize(mapper%dom_cx_d%data) /= lsize_o) then
             write(logunit, '(A,2I8)') 'nlmap> sizes do not match', &
                  lsize_o, mct_aVect_lSize(mapper%dom_cx_d%data)
             call shr_sys_abort(subname//' ERROR: nlmap> sizes do not match')
          end if
          do j = 1,lsize_o
             area = mapper%dom_cx_d%data%rAttr(kArea,j)
             dof_masses(j,     1:natt) =    avp_o%rAttr(:,j)*area
             dof_masses(j,natt+1:nfld) = nl_avp_o%rAttr(:,j)*area
          end do
          call shr_reprosum_calc(dof_masses, glbl_masses, nsum, nsum, nfld, commid=mpicom)
          deallocate(dof_masses)
          ! Check solution against local bounds.
          nsum = lsize_o
          nfld = 3*natt
          allocate(caas_wgt(nsum,nfld)) ! dm, cap low, cap high
          do j = 1,lsize_o
             area = mapper%dom_cx_d%data%rAttr(kArea,j)
             do k = 1,natt
                y = nl_avp_o%rAttr(k,j)
                lo = lcl_lo(k,j)
                hi = lcl_hi(k,j)
                if (lnorm) then
                   lo = lo*avp_o%rAttr(natt+1,j)
                   hi = hi*avp_o%rAttr(natt+1,j)
                end if
                if (y < lo) then
                   caas_wgt(j,k) = (y - lo)*area
                   y = lo
                else if (y > hi) then
                   caas_wgt(j,k) = (y - hi)*area
                   y = hi
                else
                   caas_wgt(j,k) = 0
                end if
                caas_wgt(j,  natt+k) = (y - lo)*area
                caas_wgt(j,2*natt+k) = (hi - y)*area
             end do
          end do
          allocate(gwts(nfld))
          call shr_reprosum_calc(caas_wgt, gwts, nsum, nsum, nfld, commid=mpicom)
          deallocate(caas_wgt)
          ! Combine clipping and global mass error into a single dm value.
          gwts(1:natt) = gwts(1:natt) + (glbl_masses(1:natt) - glbl_masses(natt+1:2*natt))
          if (verbose .and. amroot) then
             do k = 1,natt
                if (gwts(k) /= 0 .or. glbl_masses(k) /= 0) then
                   write(logunit, '(a,i2,a,i2,es23.15,es23.15,es23.15,es10.2)') &
                        'nlmap>  caas-dm ', k, '/', natt, &
                        ! true global mass
                        glbl_masses(k), &
                        ! dm due to global mass nonconservation in the linear map
                        glbl_masses(k) - glbl_masses(natt+k), &
                        ! dm due to clipping
                        gwts(k) - (glbl_masses(k) - glbl_masses(natt+k)), &
                        ! dm relative to true global mass
                        gwts(k)/abs(glbl_masses(k))
                end if
             end do
          end if
          ! Adjust high-order solution and set avp_o. The adjustment consists of a
          ! clip, if needed, and adding or removing mass up to the capacity.
          do j = 1,lsize_o
             do k = 1,natt
                tmp = nl_avp_o%rAttr(k,j)
                if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                   write(logunit, '(a,a,l2,i3,i6,es23.15)') 'nlmap> inf/nan0a ', &
                        trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
                end if
                tmp = avp_o%rAttr(k,j)
                if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                   write(logunit, '(a,a,l2,i3,i6,es23.15)') 'nlmap> inf/nan0b ', &
                        trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
                end if
             end do
          end do
          do k = 1,natt
             if (gwts(k) > 0) then
                tmp = gwts(2*natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      lo = lcl_lo(k,j)
                      hi = lcl_hi(k,j)
                      if (lnorm) then
                         lo = lo*avp_o%rAttr(natt+1,j)
                         hi = hi*avp_o%rAttr(natt+1,j)
                      end if
                      y = max(lo, min(hi, nl_avp_o%rAttr(k,j)))
                      avp_o%rAttr(k,j) = y + ((hi - y)/tmp)*gwts(k)
                   end do
                end if
             else if (gwts(k) < 0) then
                tmp = gwts(natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      lo = lcl_lo(k,j)
                      hi = lcl_hi(k,j)
                      if (lnorm) then
                         lo = lo*avp_o%rAttr(natt+1,j)
                         hi = hi*avp_o%rAttr(natt+1,j)
                      end if
                      y = max(lo, min(hi, nl_avp_o%rAttr(k,j)))
                      avp_o%rAttr(k,j) = y + ((y - lo)/tmp)*gwts(k)
                   end do
                end if
             end if
          end do
          deallocate(gwts, lcl_lo, lcl_hi)
          call mct_aVect_clean(nl_avp_o)
          do j = 1,lsize_o
             do k = 1,natt
                tmp = avp_o%rAttr(k,j)
                if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                   write(logunit, '(a,a,l2,i3,i6,es23.15)') 'nlmap> inf/nan 1 ', &
                        trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
                end if
             end do
          end do
          ! Clip for numerics, just against the global extrema.
          do j = 1,lsize_o
             do k = 1,natt
                if (avp_o%rAttr(k,j) == 0) cycle ! 0-mask
                avp_o%rAttr(k,j) = max(gmins(k), min(gmaxs(k), avp_o%rAttr(k,j)))
             end do
          end do
          if (verbose) then
             ! check global mass
             nsum = lsize_o
             allocate(dof_masses(nsum,natt), gwts(natt))
             do j = 1,lsize_o
                dof_masses(j,:) = avp_o%rAttr(:,j)*mapper%dom_cx_d%data%rAttr(kArea,j)
             end do
             call shr_reprosum_calc(dof_masses, gwts, nsum, nsum, natt, commid=mpicom)
             deallocate(dof_masses)
             if (amroot) then
                do k = 1,natt
                   if (gwts(k) /= 0 .or. glbl_masses(k) /= 0) then
                      tmp = (gwts(k) - glbl_masses(k))/abs(glbl_masses(k))
                      if (abs(tmp) < 1e-15) then
                         msg = ''
                      else if (abs(tmp) < 1e-13) then
                         msg = ' OK'
                      else
                         msg = ' ALARM'
                      end if
                      write(logunit, '(a,i2,a,i2,es23.15,es23.15,es10.2,a)') &
                           'nlmap> fin-mass ', k, '/', natt, glbl_masses(k), gwts(k), tmp, trim(msg)
                   end if
                end do
             end if
             deallocate(gwts)
             ! check bounds
             allocate(lmins(natt), lmaxs(natt), oglims(natt,2))
             lmins(:) =  1.e300_r8
             lmaxs(:) = -1.e300_r8
             do j = 1,lsize_o
                do k = 1,natt
                   tmp = avp_o%rAttr(k,j)
                   lmins(k) = min(lmins(k), tmp)
                   lmaxs(k) = max(lmaxs(k), tmp)
                end do
             end do
             call mpi_allreduce(lmins, oglims(:,1), natt, MPI_DOUBLE_PRECISION, MPI_MIN, mpicom, ierr)
             call mpi_allreduce(lmaxs, oglims(:,2), natt, MPI_DOUBLE_PRECISION, MPI_MAX, mpicom, ierr)
             if (amroot) then
                do k = 1,natt
                   if (oglims(k,1) >= gmins(k) .and. oglims(k,2) <= gmaxs(k)) then
                      msg = ''
                   else
                      msg = ' ALARM'
                   end if
                   write(logunit, '(a,i2,a,i2,es23.15,es23.15,a)') &
                        'nlmap> fin-bnds ', k, '/', natt, oglims(k,1), oglims(k,2), trim(msg)
                end do
             end if
             deallocate(lmins, lmaxs, oglims)
          end if
          deallocate(gmins, gmaxs, glbl_masses)
       else
          do j = 1,lsize_o
             do k = 1,natt
                lo = lcl_lo(k,j)
                hi = lcl_hi(k,j)
                if (lnorm) then
                   lo = lo*avp_o%rAttr(natt+1,j)
                   hi = hi*avp_o%rAttr(natt+1,j)
                end if
                avp_o%rAttr(k,j) = max(lo, min(hi, nl_avp_o%rAttr(k,j)))
             end do
          end do
       end if
       do j = 1,lsize_o
          do k = 1,natt
             tmp = avp_o%rAttr(k,j)
             if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                write(logunit, '(a,a,l2,i3,i6,es23.15)') 'nlmap> inf/nan 2 ', &
                     trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
             end if
          end do
       end do
    end if

    !--- renormalize avp_o by mapped norm_i  ---

    if (lnorm) then
       kf = mct_aVect_indexRA(avp_o,ffld)
       !$omp simd
       do j = 1,lsize_o
          normval = avp_o%rAttr(kf,j)
          if (normval /= 0.0_r8) then
             normval = 1.0_r8/normval
          endif
          avp_o%rAttr(:,j) = avp_o%rAttr(:,j)*normval
       enddo
    endif

    if (nl_on) then
       do j = 1,lsize_o
          do k = 1,natt
             tmp = avp_o%rAttr(k,j)
             if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                write(logunit, '(a,a,l2,i3,i6,es23.15)') 'nlmap> inf/nan 3 ', &
                     trim(mapper%mapfile), mapper%nl_conservative, k, j, tmp
             end if
          end do
       end do
    end if

    !--- copy back into av_o and we are done ---

    call mct_aVect_copy(aVin=avp_o, aVout=av_o, VECTOR=mct_usevector)

    call mct_aVect_clean(avp_i)
    call mct_aVect_clean(avp_o)

  end subroutine seq_nlmap_avNormArr

  subroutine sMat_avMult_and_calc_bounds(xAV, sMatPlus, lnorm, natt, yAV, lo, hi)
    ! Compute
    !     y = A*x
    !     l, u = bounds(A, x).
    ! l, u can then be used to compute
    !     y = clip(y, l, u).
    ! For each entry i, bounds(A, x) returns
    !     min/maxval(x such that A(i,:) is a non-0).
    ! That is, l(i), u(i) are bounds derived from the discrete domain of
    ! dependence of y(i).
    !   During initialization, strategy 'X' ('Xonly') was specified. Thus each
    ! y(i) has full access to its discrete domain of dependence.

    type (mct_aVect), intent(in)    :: xAV
    type (mct_sMatp), intent(inout) :: sMatPlus
    type (mct_aVect), intent(out)   :: yAV
    logical, intent(in) :: lnorm
    integer, intent(in) :: natt
    real(r8), dimension(:,:), intent(out) :: lo, hi

    type (mct_aVect) :: xPrimeAV, tmpav, lop, hip
    integer :: ierr, ne, irow, icol, iwgt, i, j, row, col, ysize
    real(r8) :: wgt, tmp

    ysize = mct_aVect_lsize(yAV)
    if (size(lo,2) /= ysize) then
       print *, 'nlmap> size(lo,1),ysize =',size(lo,2),ysize
       call shr_sys_abort('(seq_map_avNormArr) ERROR: lo,hi and y sizes do not match')
    end if

    ! x -> x'
    call mct_aVect_init(xPrimeAV, xAV, sMatPlus%XPrimeLength)
    call mct_aVect_zero(xPrimeAV)
    call mct_rearr_rearrange(xAV, xPrimeAV, sMatPlus%XToXPrime, &
         tag=sMatPlus%Tag, vector=mct_usevector, &
         alltoall=.true., handshake=.true.)
    ! y' = A x'
    call mct_aVect_init(tmpav, yAV, sMatPlus%YPrimeLength)
    call mct_aVect_zero(tmpav)
    call mct_sMat_avMult(xPrimeAV, sMatPlus%Matrix, tmpav, vector=mct_usevector)
    ! y' -> y
    call mct_rearr_rearrange(tmpav, yAV, sMatPlus%YPrimeToY, &
         tag=sMatPlus%Tag, sum=.true., vector=mct_usevector, &
         alltoall=.true., handshake=.true.)
    call mct_aVect_clean(tmpav)
    
    ! l',u' = bounds(A, x')
    call mct_aVect_init(lop, yAV, sMatPlus%YPrimeLength)
    call mct_aVect_init(hip, yAV, sMatPlus%YPrimeLength)
    lop%rAttr(:,:) =  1.e300_r8
    hip%rAttr(:,:) = -1.e300_r8
    ne = mct_sMat_lsize(sMatPlus%Matrix)
    irow = mct_sMat_indexIA(sMatPlus%Matrix,'lrow')
    icol = mct_sMat_indexIA(sMatPlus%Matrix,'lcol')
    iwgt = mct_sMat_indexRA(sMatPlus%Matrix,'weight')
    do i = 1, ne
       row = sMatPlus%Matrix%data%iAttr(irow,i)
       col = sMatPlus%Matrix%data%iAttr(icol,i)
       wgt = sMatPlus%Matrix%data%rAttr(iwgt,i)
       if (wgt == 0) cycle
       if (lnorm) then
          if (xPrimeAV%rAttr(natt+1,col) == 0.0_r8) cycle
       end if
       do j = 1, natt
          tmp = xPrimeAV%rAttr(j,col)
          if (lnorm) tmp = tmp / xPrimeAV%rAttr(natt+1,col)
          lop%rAttr(j,row) = min(lop%rAttr(j,row), tmp)
          hip%rAttr(j,row) = max(hip%rAttr(j,row), tmp)
       end do
    end do
    ! Set bounds to 0 if there are no valid matrix entries for this row.
    do i = 1, sMatPlus%YPrimeLength
       do j = 1, natt
          if (lop%rAttr(j,i) > hip%rAttr(j,i)) then
             lop%rAttr(j,i) = 0
             hip%rAttr(j,i) = 0
          end if
       end do
    end do

    call mct_aVect_clean(xPrimeAV, ierr)

    ! l',u' -> l,u
    call mct_aVect_init(tmpav, yAV, ysize)
    call mct_aVect_zero(tmpav)
    call mct_rearr_rearrange(lop, tmpav, sMatPlus%YPrimeToY, &
         tag=sMatPlus%Tag, sum=.true., vector=mct_usevector, &
         alltoall=.true., handshake=.true.)
    lo(:natt,:ysize) = tmpav%rAttr(:natt,:ysize)
    call mct_aVect_zero(tmpav)
    call mct_rearr_rearrange(hip, tmpav, sMatPlus%YPrimeToY, &
         tag=sMatPlus%Tag, sum=.true., vector=mct_usevector, &
         alltoall=.true., handshake=.true.)
    hi(:natt,:ysize) = tmpav%rAttr(:natt,:ysize)
    
    call mct_aVect_clean(tmpav)
    call mct_aVect_clean(lop)
    call mct_aVect_clean(hip)

  end subroutine sMat_avMult_and_calc_bounds

  subroutine seq_nlmap_check_matrices(m)
    type(seq_map), intent(in) :: m

    if (.not. m%nl_available) return

    
  end subroutine seq_nlmap_check_matrices

end module seq_nlmap_mod
