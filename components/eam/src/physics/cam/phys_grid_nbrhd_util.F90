module phys_grid_nbrhd_util
  ! These routines are in a separate module to resolve dependencies
  ! properly. See phys_grid_nbrhd.
  !
  ! AMB 2020/09 Initial

  use parallel_mod, only: par
  use spmd_utils, only: iam, masterproc, npes
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile, only: iulog
  use cam_abortutils, only: endrun
  use dimensions_mod, only: nelemd, npsq
  use ppgrid, only: begchunk, endchunk, nbrhdchunk, pcols, pver
  use kinds, only: real_kind, int_kind
  use phys_grid, only: get_ncols_p, get_gcol_all_p
  use physics_types, only: physics_state
  use phys_grid_nbrhd ! all public routines

  implicit none
  private

  public :: &
       ! API
       nbrhd_d_p_coupling, &
       nbrhd_p_p_coupling, &
       ! For testing
       nbrhd_test_api
  
contains

  subroutine nbrhd_d_p_coupling(ps, zs, T, om, uv, q, state)
    use dyn_grid, only: get_block_gcol_cnt_d

    ! Dynamics blocks -> owned-chunk columns' neighborhods, for use in
    ! d_p_coupling.

    use parallel_mod, only: par
    use perf_mod, only: t_startf, t_stopf

    real(r8), intent(in) :: ps(:,:), zs(:,:), T(:,:,:), om(:,:,:), uv(:,:,:,:), q(:,:,:,:)
    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)

    real(r8), allocatable :: bbuf(:), cbuf(:)
    integer, allocatable :: bptr(:,:), cptr(:)
    integer :: nbrhd_pcnst, rcdsz, bnrecs, cnrecs, max_numlev, max_numrep, num_recv_col, &
         numlev, numrep, ie, bid, ncol, j, k, p, lchnk, icol, ptr, ierr

    call t_startf('dpcopy_nbrhd')

    nbrhd_pcnst = nbrhd_get_option_pcnst()
    rcdsz = 4 + nbrhd_pcnst
    call nbrhd_block_to_chunk_sizes(bnrecs, cnrecs, max_numlev, max_numrep, num_recv_col)

    allocate(bbuf(rcdsz*bnrecs), cbuf(rcdsz*cnrecs), stat=ierr)
    if (ierr /= 0) then
       call endrun('phys_grid_nbrhd_util::nbrhd_d_p_coupling: alloc b,cbuf failed')
    end if

    if (par%dynproc) then
       allocate(bptr(0:max_numlev-1,max_numrep), stat=ierr)
       if (ierr /= 0) then
          call endrun('phys_grid_nbrhd_util::nbrhd_d_p_coupling: alloc bptr failed')
       end if
       do ie = 1, nelemd
          bid = nbrhd_get_ie2bid(ie)
          ncol = get_block_gcol_cnt_d(bid)
          do icol = 1, ncol
             call nbrhd_block_to_chunk_send_pters(ie, icol, rcdsz, &
                  numlev, numrep, bptr)
             do j = 1, numrep
                ptr = bptr(0,j)
                bbuf(ptr+0) = ps(icol,ie)
                bbuf(ptr+1) = zs(icol,ie)
                bbuf(ptr+2:ptr+3) = 0.0_r8
                do k = 1, numlev-1
                   ptr = bptr(k,j)
                   bbuf(ptr+0) =  T(icol,  k,ie)
                   bbuf(ptr+1) = uv(icol,1,k,ie)
                   bbuf(ptr+2) = uv(icol,2,k,ie)
                   bbuf(ptr+3) = om(icol,  k,ie)
                   do p = 1, nbrhd_pcnst
                      bbuf(ptr+3+p) = q(icol,k,p,ie)
                   end do
                end do
             end do
          end do
       end do
       deallocate(bptr)
    else
       bbuf(:) = 0._r8
    end if

    call t_startf('nbrhd_block_to_chunk')
    call nbrhd_transpose_block_to_chunk(rcdsz, bbuf, cbuf)
    call t_stopf ('nbrhd_block_to_chunk')

    allocate(cptr(0:max_numlev-1), stat=ierr)
    if (ierr /= 0) then
       call endrun('phys_grid_nbrhd_util::nbrhd_d_p_coupling: alloc cptr failed')
    end if
    lchnk = endchunk+nbrhdchunk
    do icol = 1, num_recv_col
       call nbrhd_block_to_chunk_recv_pters(icol, rcdsz, numlev, cptr)
       ptr = cptr(0)
       state(lchnk)%ps  (icol) = cbuf(ptr+0)
       state(lchnk)%phis(icol) = cbuf(ptr+1)
       do k = 1, numlev-1
          ptr = cptr(k)
          state(lchnk)%T    (icol,k) = cbuf(ptr+0)
          state(lchnk)%u    (icol,k) = cbuf(ptr+1)
          state(lchnk)%v    (icol,k) = cbuf(ptr+2)
          state(lchnk)%omega(icol,k) = cbuf(ptr+3)
          do p = 1, nbrhd_pcnst
             state(lchnk)%q(icol,k,p) = cbuf(ptr+3+p)
          end do
       end do
    end do
    deallocate(cptr)

    deallocate(bbuf, cbuf)

    call nbrhd_copy_states(state)
    call t_stopf('dpcopy_nbrhd')
  end subroutine nbrhd_d_p_coupling

  subroutine nbrhd_p_p_coupling(state)
    ! Owned-chunk -> owned-chunk columns' neighborhods, for use during a physics
    ! time step, if this need ever arises.

    use perf_mod, only: t_startf, t_stopf

    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)

    real(r8), allocatable :: sbuf(:), rbuf(:)
    integer, allocatable :: sptr(:,:), rptr(:)
    integer :: nbrhd_pcnst, rcdsz, snrecs, rnrecs, max_numlev, max_numrep, num_recv_col, &
         numlev, numrep, lid, ncol, icol, j, k, p, lchnk, ptr, ierr

    call t_startf('ppcopy_nbrhd')
    nbrhd_pcnst = nbrhd_get_option_pcnst()
    rcdsz = 4 + nbrhd_pcnst
    call nbrhd_chunk_to_chunk_sizes(snrecs, rnrecs, max_numlev, max_numrep, num_recv_col)
    allocate(sbuf(rcdsz*snrecs), rbuf(rcdsz*rnrecs), stat=ierr)
    if (ierr /= 0) then
       call endrun('phys_grid_nbrhd_util::nbrhd_p_p_coupling: alloc s,rbuf failed')
    end if

    allocate(sptr(0:max_numlev-1,max_numrep), stat=ierr)
    if (ierr /= 0) then
       call endrun('phys_grid_nbrhd_util::nbrhd_p_p_coupling: alloc sptr failed')
    end if
    do lid = begchunk, endchunk
       ncol = get_ncols_p(lid)
       do icol = 1, ncol
          call nbrhd_chunk_to_chunk_send_pters(lid, icol, rcdsz, numlev, numrep, sptr)
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
    deallocate(sptr)

    call t_startf('nbrhd_chunk_to_chunk')
    call nbrhd_transpose_chunk_to_chunk(rcdsz, sbuf, rbuf)
    call t_stopf('nbrhd_chunk_to_chunk')

    allocate(rptr(0:max_numlev-1), stat=ierr)
    if (ierr /= 0) then
       call endrun('phys_grid_nbrhd_util::nbrhd_p_p_coupling: alloc rptr failed')
    end if
    lchnk = endchunk+nbrhdchunk
    do icol = 1, num_recv_col
       call nbrhd_chunk_to_chunk_recv_pters(icol, rcdsz, numlev, rptr)
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
    call t_stopf('ppcopy_nbrhd')
  end subroutine nbrhd_p_p_coupling

  subroutine nbrhd_copy_states(state)
    ! Copy state from normal chunks to the extra neighborhood chunk. These are
    ! neighborhood columns whose data already exist on this pe.

    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)

    integer (int_kind) :: n, i, lchnk, lchnke, icol, icole, pcnst

    lchnke = endchunk+nbrhdchunk
    pcnst = nbrhd_get_option_pcnst()
    n = nbrhd_get_num_copies()
    do i = 1, n
       call nbrhd_get_copy_idxs(i, lchnk, icol, icole)
       state(lchnke)%ps   (icole  ) = state(lchnk)%ps   (icol  )
       state(lchnke)%phis (icole  ) = state(lchnk)%phis (icol  )
       state(lchnke)%T    (icole,:) = state(lchnk)%T    (icol,:)
       state(lchnke)%u    (icole,:) = state(lchnk)%u    (icol,:)
       state(lchnke)%v    (icole,:) = state(lchnk)%v    (icol,:)
       state(lchnke)%omega(icole,:) = state(lchnk)%omega(icol,:)
       state(lchnke)%q(icole,:,1:pcnst) = state(lchnk)%q(icol,:,1:pcnst)
    end do
  end subroutine nbrhd_copy_states

  subroutine nbrhd_test_api(state)
    use dyn_grid, only: get_horiz_grid_dim_d, get_horiz_grid_d

    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)

    real(r8), allocatable, dimension(:) :: lats_d, lons_d
    integer :: d1, d2, ngcols

    if (nbrhd_get_option_test() == 0) return

    call get_horiz_grid_dim_d(d1, d2)
    ngcols = d1*d2
    allocate(lats_d(ngcols), lons_d(ngcols))
    call get_horiz_grid_d(ngcols, clat_d_out=lats_d, clon_d_out=lons_d)

    if (nbrhd_get_option_block_to_chunk_on()) &
         call test_api(lats_d, lons_d, state, .true. )
    if (nbrhd_get_option_chunk_to_chunk_on()) &
         call test_api(lats_d, lons_d, state, .false.)

    deallocate(lats_d, lons_d)
  end subroutine nbrhd_test_api

  subroutine test_api(lats_d, lons_d, state, owning_blocks)
    use dyn_grid, only: get_block_gcol_cnt_d, get_block_gcol_d
    use constituents, only: pcnst

    real(r8), parameter :: none = -10000._r8

    real(r8), intent(in) :: lats_d(:), lons_d(:)
    type (physics_state), intent(inout) :: state(begchunk:endchunk+nbrhdchunk)
    logical, intent(in) :: owning_blocks

    real(r8), allocatable :: lats(:,:), lons(:,:), sbuf(:), rbuf(:), &
         ps(:,:), zs(:,:), T(:,:,:), om(:,:,:), uv(:,:,:,:), q(:,:,:,:)
    integer, allocatable :: gcols(:), icols(:), used(:)
    real(r8) :: lat, lon, x, y, z, angle, max_angle
    integer ::  nerr, ntest, ie, bid, ncol, icol, lid, gcol, k, p, nbrhd_pcnst, &
         n, lide, icole
    logical :: e

    if (masterproc) write(iulog,*) 'nbr> test_api', owning_blocks
    
    ! Zorch communicated state values.
    do lid = begchunk, endchunk+nbrhdchunk
       state(lid)%ps  (:) = none
       state(lid)%phis(:) = none
       state(lid)%T    (:,:) = none
       state(lid)%u    (:,:) = none
       state(lid)%v    (:,:) = none
       state(lid)%omega(:,:) = none
       state(lid)%q(:,:,:) = none
    end do

    nerr = 0
    ntest = 0
    allocate(gcols(max(get_ncols_p(endchunk+nbrhdchunk), max(npsq, pcols))))

    ! Check that all states have the expected (lat,lon).
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
    ! columns' states get filled. Inject artificial, checkable values.
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

    nbrhd_pcnst = nbrhd_get_option_pcnst()
    e = test(nerr, ntest, nbrhd_pcnst >= 1 .and. nbrhd_pcnst <= pcnst, 'nbrhd_pcnst')

    ! Carry out a nbrhd comm round.
    if (owning_blocks) then
       allocate(ps(npsq,nelemd), zs(npsq,nelemd), &
                T(npsq,pver,nelemd), om(npsq,pver,nelemd), &
                uv(npsq,2,pver,nelemd), &
                q(npsq,pver,nbrhd_pcnst,nelemd))
       do ie = 1, nelemd
          bid = nbrhd_get_ie2bid(ie)
          ncol = get_block_gcol_cnt_d(bid)
          call get_block_gcol_d(bid, ncol, gcols)
          do icol = 1, ncol
             gcol = gcols(icol)
             lat = lats_d(gcol)
             lon = lons_d(gcol)
             ps(icol,ie) = lat
             zs(icol,ie) = lon
             do k = 1, pver
                T (icol,k,  ie) = lat + k - 1
                om(icol,k,  ie) = lon + k - 1
                uv(icol,1,k,ie) = lat + k - 2
                uv(icol,2,k,ie) = lon + k - 2
                do p = 1, nbrhd_pcnst
                   q(icol,k,p,ie) = p*(lat + lon) + k
                end do
             end do
          end do
       end do
       call nbrhd_d_p_coupling(ps, zs, T, om, uv, q, state)
       deallocate(ps, zs, T, om, uv, q)
    else
       call nbrhd_p_p_coupling(state)
    end if

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

    if (nerr > 0) write(iulog,*) 'nbr> test_api FAIL', owning_blocks, nerr, ntest
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
