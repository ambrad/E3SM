module phys_grid_nbrhd_util
  ! These routines are in a separate module to resolve dependencies properly.

  use parallel_mod, only: par
  use spmd_utils, only: iam, masterproc, npes
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile, only: iulog
  use dimensions_mod, only: nelemd, np, npsq
  use constituents, only: pcnst
  use ppgrid, only: begchunk, endchunk, nbrhdchunk, pcols, pver
  use kinds, only: real_kind, int_kind
  use phys_grid, only: get_ncols_p, get_gcol_all_p
  use dyn_grid, only: get_horiz_grid_dim_d, get_horiz_grid_d, get_block_gcol_cnt_d, &
       get_block_gcol_d, get_gcol_block_d, fv_nphys
  use physics_types, only: physics_state
  use element_mod, only: element_t
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

  subroutine nbrhd_test_api(elem, state)
    type (element_t), intent(in) :: elem(:)
    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)

    real(r8), allocatable, dimension(:) :: lats_d, lons_d
    integer :: d1, d2, ngcols

    call get_horiz_grid_dim_d(d1, d2)
    ngcols = d1*d2
    allocate(lats_d(ngcols), lons_d(ngcols))
    call get_horiz_grid_d(ngcols, clat_d_out=lats_d, clon_d_out=lons_d)

    if (nbrhd_get_option_block_to_chunk_on()) &
         call test_api(lats_d, lons_d, elem, state, .true. )
    if (nbrhd_get_option_chunk_to_chunk_on()) &
         call test_api(lats_d, lons_d, elem, state, .false.)

    deallocate(lats_d, lons_d)
  end subroutine nbrhd_test_api

  subroutine test_api(lats_d, lons_d, elem, state, owning_blocks)
    real(r8), intent(in) :: lats_d(:), lons_d(:)
    type (element_t), intent(in) :: elem(:)
    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)
    logical, intent(in) :: owning_blocks

    real(r8), parameter :: none = -10000

    real(r8), allocatable :: lats(:,:), lons(:,:), sbuf(:), rbuf(:)
    integer, allocatable :: sptr(:,:), rptr(:), gcols(:), icols(:), used(:)
    real(r8) :: lat, lon, x, y, z, angle, max_angle
    integer :: nerr, ntest, bid, n, k, lid, ncol, icol, rcdsz, gcol, nphys, nphys_sq, &
         numlev, numrep, max_numlev, max_numrep, snrecs, rnrecs, j, p, lchnk, ptr, &
         nbrhd_pcnst, num_recv_col, lide, icole
    logical :: e

    if (masterproc) write(iulog,*) 'nbr> test_api', owning_blocks
    nerr = 0
    ntest = 0
    allocate(gcols(max(get_ncols_p(endchunk+nbrhdchunk), max(npsq, pcols))))

    do lid = begchunk, endchunk+nbrhdchunk
       ncol = get_ncols_p(lid)
       call get_gcol_all_p(lid, ncol, gcols)
       do icol = 1, ncol
          gcol = gcols(icol)
          e = test(nerr, ntest, state(lid)%lat(icol) == lats_d(gcol), 'lat')
          e = test(nerr, ntest, state(lid)%lon(icol) == lons_d(gcol), 'lon')
       end do
    end do

    ! Mimic the result of running the standard part of d_p_coupling: iam's
    ! columns' states get filled.
    do lid = begchunk, endchunk
       ncol = get_ncols_p(lid)
       call get_gcol_all_p(lid, pcols, gcols)
       do icol = 1, ncol
          gcol = gcols(icol)
          lat = lats_d(gcol)
          lon = lons_d(gcol)
          state(lid)%ps  (icol) = lat
          state(lid)%phis(icol) = lon
          do k = 1, pver
             state(lid)%T    (icol,k) = lat + k - 1
             state(lid)%omega(icol,k) = lon + k - 1
             state(lid)%u    (icol,k) = lat + k - 2
             state(lid)%v    (icol,k) = lon + k - 2
             do p = 1, pcnst
                state(lid)%q(icol,k,p) = p*(lat + lon) + k
             end do
          end do
       end do
    end do

    if (fv_nphys > 0) then
       nphys = fv_nphys
    else
       nphys = np
    end if
    nphys_sq = nphys*nphys
    nbrhd_pcnst = nbrhd_get_option_pcnst()
    e = test(nerr, ntest, nbrhd_pcnst >= 1 .and. nbrhd_pcnst <= pcnst, 'nbrhd_pcnst')
    rcdsz = 4 + nbrhd_pcnst

    if (owning_blocks) then
       call nbrhd_block_to_chunk_sizes(snrecs, rnrecs, max_numlev, max_numrep, num_recv_col)
    else
       call nbrhd_chunk_to_chunk_sizes(snrecs, rnrecs, max_numlev, max_numrep, num_recv_col)
    end if
    allocate(sbuf(rcdsz*snrecs), rbuf(rcdsz*rnrecs))

    allocate(sptr(0:max_numlev-1,max_numrep))
    if (owning_blocks) then
       ! Dynamics blocks -> send buf.
       do lid = 1, nelemd
          if (fv_nphys > 0) then
             ncol = nphys_sq
          else
             ncol = elem(lid)%idxP%NumUniquePts
          end if
          call get_block_gcol_d(elem(lid)%GlobalID, ncol, gcols)
          do icol = 1, ncol
             if (owning_blocks) then
                call nbrhd_block_to_chunk_send_pters(lid, icol, rcdsz, &
                     numlev, numrep, sptr)
             else
                call nbrhd_chunk_to_chunk_send_pters(lid, icol, rcdsz, &
                     numlev, numrep, sptr)
             end if
             gcol = gcols(icol)
             lat = lats_d(gcol)
             lon = lons_d(gcol)
             do j = 1, numrep
                ptr = sptr(0,j)
                sbuf(ptr+0) = lat
                sbuf(ptr+1) = lon
                sbuf(ptr+2:ptr+3) = 0.0_r8
                do k = 1, numlev-1
                   ptr = sptr(k,j)
                   sbuf(ptr+0) = lat + k - 1
                   sbuf(ptr+1) = lat + k - 2
                   sbuf(ptr+2) = lon + k - 2
                   sbuf(ptr+3) = lon + k - 1
                   do p = 1, nbrhd_pcnst
                      sbuf(ptr+3+p) = p*(lat + lon) + k
                   end do
                end do
             end do
          end do
       end do
    else
       ! Owned state -> send buf.
       do lid = begchunk, endchunk
          ncol = get_ncols_p(lid)
          do icol = 1, ncol
             call nbrhd_chunk_to_chunk_send_pters(lid, icol, rcdsz, &
                  numlev, numrep, sptr)
             do j = 1, numrep
                ptr = sptr(0,j)
                sbuf(ptr+0) = state(lid)%ps  (icol)
                sbuf(ptr+1) = state(lid)%phis(icol)
                sbuf(ptr+2:ptr+3) = 0.0_r8
                do k = 1, numlev-1
                   ptr = sptr(k,j)
                   sbuf(ptr+0) = state(lid)%T(icol,k)
                   sbuf(ptr+1) = state(lid)%u(icol,k)
                   sbuf(ptr+2) = state(lid)%v(icol,k)
                   sbuf(ptr+3) = state(lid)%omega(icol,k)
                   do p = 1, nbrhd_pcnst
                      sbuf(ptr+3+p) = state(lid)%q(icol,k,p)
                   end do
                end do
             end do
          end do
       end do
    end if
    deallocate(sptr, gcols)

    if (owning_blocks) then
       call nbrhd_transpose_block_to_chunk(rcdsz, sbuf, rbuf)
    else
       call nbrhd_transpose_chunk_to_chunk(rcdsz, sbuf, rbuf)
    end if

    ! Receive buf -> neighborhood columns' states.
    allocate(rptr(0:max_numlev-1))
    lchnk = endchunk+nbrhdchunk
    do icol = 1, num_recv_col
       if (owning_blocks) then
          call nbrhd_block_to_chunk_recv_pters(icol, rcdsz, numlev, rptr)
       else
          call nbrhd_chunk_to_chunk_recv_pters(icol, rcdsz, numlev, rptr)
       end if
       ptr = rptr(0)
       state(lchnk)%ps  (icol) = rbuf(ptr+0)
       state(lchnk)%phis(icol) = rbuf(ptr+1)
       do k = 1, numlev-1
          ptr = rptr(k)
          state(lchnk)%T    (icol,k) = rbuf(ptr+0)
          state(lchnk)%u    (icol,k) = rbuf(ptr+1)
          state(lchnk)%v    (icol,k) = rbuf(ptr+2)
          state(lchnk)%omega(icol,k) = rbuf(ptr+3)
          do p = 1, nbrhd_pcnst
             state(lchnk)%q(icol,k,p) = rbuf(ptr+3+p)
          end do
       end do
    end do
    deallocate(rptr)

    deallocate(sbuf, rbuf)

    call nbrhd_copy_states(state)

    ! Check that we have the expected values in all states.
    do lid = begchunk, endchunk+nbrhdchunk
       do icol = 1, state(lid)%ncol
          lat = state(lid)%lat(icol)
          lon = state(lid)%lon(icol)
          e = test(nerr, ntest, state(lid)%ps  (icol) == lat, 'ps')
          e = test(nerr, ntest, state(lid)%phis(icol) == lon, 'zs')
          do k = 1, pver
             e = test(nerr, ntest, state(lid)%T    (icol,k) == lat + k - 1, 'T'  )
             e = test(nerr, ntest, state(lid)%omega(icol,k) == lon + k - 1, 'om' )
             e = test(nerr, ntest, state(lid)%u    (icol,k) == lat + k - 2, 'uv1')
             e = test(nerr, ntest, state(lid)%v    (icol,k) == lon + k - 2, 'uv2')
             do p = 1, nbrhd_pcnst
                e = test(nerr, ntest, state(lid)%q(icol,k,p) == p*(lat + lon) + k, 'q')
             end do
          end do
       end do
    end do

    ! Check neighborhoods.
    max_angle = nbrhd_get_option_angle()
    lide = endchunk+nbrhdchunk
    allocate(icols(128), used(state(lide)%ncol))
    used(:) = 0
    do lid = begchunk, endchunk
       do icol = 1, state(lid)%ncol
          call latlon2xyz(state(lid)%lat(icol), state(lid)%lon(icol), x, y, z)
          n = nbrhd_get_nbrhd_size(lid, icol)
          if (n > size(icols)) then
             deallocate(icols)
             allocate(icols(2*n))
          end if
          call nbrhd_get_nbrhd(lid, icol, icols)
          do k = 1, n
             icole = icols(k)
             used(icole) = used(icole) + 1
             angle = unit_sphere_angle(x, y, z, &
                  state(lide)%lat(icole), state(lide)%lon(icole))
             e = test(nerr, ntest, angle <= max_angle, 'angle')
          end do
       end do
    end do
    e = test(nerr, ntest, all(used > 0), 'all nbrhd cols used')
    deallocate(icols, used)

    if (nerr > 0) write(iulog,*) 'nbr> test_api FAIL', nerr, ntest
  end subroutine test_api

  function test(nerr, ntest, cond, message) result(out)
    integer, intent(inout) :: nerr, ntest
    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    ntest = ntest + 1
    if (.not. cond) then
       write(iulog,*) 'nbr> test ', trim(message)
       nerr = nerr + 1
    end if
    out = cond
  end function test

end module phys_grid_nbrhd_util
