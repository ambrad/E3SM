module phys_grid_nbrhd
  ! Communication schedule and data structures to supplement a column with data
  ! in its neighborhood for use in scale-aware physics parameterizations.
  !
  ! AMB 2020/09 Initial

  use spmd_utils, only: iam, masterproc, npes
  use shr_kind_mod, only: r8 => shr_kind_r8, r4 => shr_kind_r4
  use ppgrid, only: pver, begchunk, endchunk, nbrhdchunk
  use m_MergeSorts, only: IndexSet, IndexSort
  use phys_grid_types

  implicit none
  private

  type :: PhysGridData ! collect phys_grid module data we need
     integer :: clat_p_tot, nlcols, ngcols, ngcols_p
     integer, pointer, dimension(:) :: lat_p, lon_p, latlon_to_dyn_gcol_map, clat_p_idx
     real(r8), pointer, dimension(:) :: clat_p, clon_p
  end type PhysGridData

  type :: SparseTriple
     ! xs(i) maps to ys(yptr(i):yptr(i+1)-1)
     integer, pointer :: xs(:), yptr(:), ys(:)
  end type SparseTriple

  type :: IdMap
     integer, allocatable :: id1(:), id2(:)
  end type IdMap

  type :: Offsets
     integer :: ncol
     ! Offsets for the columns in a block. Each one can have a different number
     ! of repetitions in the send buffer than the others.
     integer, allocatable :: numlev(:), numrep(:)
     ! os(col(i):col(i+1)-1) are offsets for all repetitions of a column.
     integer, allocatable :: col(:), os(:)
  end type Offsets

  type :: CommData
     integer, allocatable, dimension(:) :: sndcnts, sdispls, rcvcnts, rdispls, &
          pdispls, dp_coup_proc
     integer :: lopt, prev_record_size, dp_coup_steps
  end type CommData

  type :: ColumnNeighborhoods
     ! A neighborhood is a list of gcols neighboring a column. Neighborhoods,
     ! plural, is a list of these of lists.
     integer :: verbose
     ! Radian angle defining neighborhood. Defines max between a column center
     ! and column centers within its neighborhood.
     real(r8) :: max_angle
     ! For each gcol in iam's chunks, the list of gcols in its neighborhood.
     type (SparseTriple) :: chk_nbrhds
     ! blk_offset is index using local block ID, not global as in phys_grid.
     integer, allocatable, dimension(:) :: blk_num, chk_num
     integer :: blk_nrecs, chk_nrecs, max_numrep, max_numlev
     type (Offsets), allocatable, dimension(:) :: blk_offset
     integer, allocatable :: chk_numlev(:), chk_offset(:)
     type (CommData) :: comm_data
  end type ColumnNeighborhoods

  type (ColumnNeighborhoods), private :: cns

  public :: &
       ! phys_grid initialization
       nbrhd_init, &
       ! dp_coupling communication
       nbrhd_get_nrecs, &
       nbrhd_block_to_chunk_send_sizes, &
       nbrhd_block_to_chunk_send_pters, &
       nbrhd_transpose_block_to_chunk, &
       nbrhd_block_to_chunk_recv_pters, &
       ! API for parameterizations
       nbrhd_get_nbrhd_size, &
       nbrhd_get_nbrhd

contains

  subroutine nbrhd_init(clat_p_tot, clat_p_idx, clat_p, clon_p, lat_p, lon_p, &
       latlon_to_dyn_gcol_map, nlcols, ngcols, ngcols_p, chunks, knuhcs, phys_alltoall)

    integer, intent(in) :: clat_p_tot, nlcols, ngcols, ngcols_p, phys_alltoall
    integer, target, dimension(:), intent(in) :: clat_p_idx, lat_p, lon_p, &
         latlon_to_dyn_gcol_map
    real(r8), target, dimension(:), intent(in) :: clat_p, clon_p
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)

    type (PhysGridData) :: gd
    type (SparseTriple) :: cpe2nbrs, dpe2nbrs
    integer :: bnrec, cnrec
    logical :: e

    call run_unit_tests()

    cns%verbose = 1
    nbrhdchunk = 1
    cns%max_angle = 0.06d0

    gd%clat_p_tot = clat_p_tot
    gd%nlcols = nlcols; gd%ngcols = ngcols; gd%ngcols_p = ngcols_p
    gd%lat_p => lat_p; gd%lon_p => lon_p
    gd%clat_p_idx => clat_p_idx; gd%latlon_to_dyn_gcol_map => latlon_to_dyn_gcol_map
    gd%clat_p => clat_p; gd%clon_p => clon_p

    call find_chunk_nbrhds(cns, gd, chunks, cns%chk_nbrhds)
    call make_cpe2nbrs(cns, gd, chunks, knuhcs, cpe2nbrs)
    call make_dpe2nbrs(cns, gd, cns%chk_nbrhds, dpe2nbrs)
    call make_comm_schedule(cns, gd, cpe2nbrs, dpe2nbrs)
    call init_comm_data(cns, cns%comm_data, phys_alltoall)

    call SparseTriple_deallocate(cpe2nbrs)
    call SparseTriple_deallocate(dpe2nbrs)

    if (cns%verbose > 0) then
       call nbrhd_get_nrecs(bnrec, cnrec)
       print *,'amb> nrec', bnrec, cnrec
       call test_comm_schedule(cns)
    end if
  end subroutine nbrhd_init

  subroutine nbrhd_get_nrecs(block_buf_nrecs, chunk_buf_nrecs)
    integer, intent(out) :: block_buf_nrecs, chunk_buf_nrecs

    block_buf_nrecs = cns%blk_nrecs
    chunk_buf_nrecs = cns%chk_nrecs
  end subroutine nbrhd_get_nrecs

  subroutine nbrhd_block_to_chunk_send_sizes(max_numlev, max_numrep)
    ! Get max numlev, numrep to size pter arrays for *_pters routines.

    integer, intent(out) :: max_numlev, max_numrep

    max_numlev = cns%max_numlev
    max_numrep = cns%max_numrep
  end subroutine nbrhd_block_to_chunk_send_sizes

  subroutine nbrhd_block_to_chunk_send_pters(ie, icol, rcdsz, numlev, numrep, ptr)
    ! Set up the pointer array for column icol of block having local block ID
    ! ie. rcdsz is the record size. On output, ptr(1:numlev, 1:numrep) is filled
    ! with offsets. Since the map from dynamics blocks to chunks is 1-many,
    ! numrep is >= 1 and not simply 1. This is different than in the basic
    ! phys_grid send_pters routine.

    integer, intent(in) :: ie, icol, rcdsz
    integer, intent(out) :: numlev, numrep
    integer, intent(out) :: ptr(:,:) ! >= max_numlev x >= max_numrep

    integer :: i, j, k
    logical :: e

    e = assert(ie >= 1 .and. ie <= size(cns%blk_offset), 'send_pters: ie')
    e = assert(icol >= 1 .and. icol <= cns%blk_offset(ie)%ncol, 'send_pters: icol')
    numlev = cns%blk_offset(ie)%numlev(icol)
    numrep = cns%blk_offset(ie)%numrep(icol)
    ptr(:,:) = -1
    j = cns%blk_offset(ie)%col(icol)
    do i = 1, numrep
       ptr(1,i) = cns%blk_offset(ie)%os(j+i-1) * rcdsz
       do k = 2, numlev
          ptr(k,i) = ptr(i,1) + rcdsz*(k-1)
       end do
    end do
  end subroutine nbrhd_block_to_chunk_send_pters

  subroutine nbrhd_transpose_block_to_chunk(rcdsz, blk_buf, chk_buf)
    ! If running on just one pe or SPMD is not defined, then there are no comm
    ! data, so this routine does nothing, and comm data have size 0.

#if defined SPMD
    use spmd_utils, only: mpicom, altalltoallv
    use mpishorthand, only: mpir8
#endif

    integer, intent(in) :: rcdsz
    real(r8), intent(in) :: blk_buf(rcdsz*cns%blk_nrecs)
    real(r8), intent(out) :: chk_buf(rcdsz*cns%chk_nrecs)

#if defined SPMD
    integer, parameter :: msgtag = 6042

    integer :: ssz, rsz, lwindow

    call make_comm_data(cns, cns%comm_data, rcdsz)
    ssz = rcdsz*cns%blk_nrecs
    rsz = rcdsz*cns%chk_nrecs
    lwindow = -1
    call altalltoallv(cns%comm_data%lopt, iam, npes, &
         cns%comm_data%dp_coup_steps, cns%comm_data%dp_coup_proc, &
         blk_buf, ssz, cns%comm_data%sndcnts, cns%comm_data%sdispls, mpir8, &
         chk_buf, rsz, cns%comm_data%rcvcnts, cns%comm_data%rdispls, mpir8, &
         msgtag, cns%comm_data%pdispls, mpir8, lwindow, mpicom)
#endif
  end subroutine nbrhd_transpose_block_to_chunk

  subroutine nbrhd_block_to_chunk_recv_pters(icol, rcdsz, numlev, ptr)
    integer, intent(in) :: icol, rcdsz
    integer, intent(out) :: numlev
    integer, intent(out) :: ptr(:) ! >= max_numlev

    integer :: k
    logical :: e
    
    e = assert(icol >= 1 .and. icol <= size(cns%chk_offset), 'recv_pters: icol')
    numlev = cns%chk_numlev(icol)
    ptr(1) = cns%chk_offset(icol) * rcdsz
    do k = 2, numlev
       ptr(k) = ptr(1) + rcdsz*(k-1)
    end do
  end subroutine nbrhd_block_to_chunk_recv_pters

  function nbrhd_get_nbrhd_size(col_chunk_idx, col_col_idx) result(n)
    integer, intent(in) :: col_chunk_idx, col_col_idx
    integer :: n
    n = -1
  end function nbrhd_get_nbrhd_size

  subroutine nbrhd_get_nbrhd(col_chunk_idx, col_col_idx, chunk_idxs, col_idxs)
    integer, intent(in) :: col_chunk_idx, col_col_idx
    integer, dimension(:), intent(out) :: chunk_idxs, col_idxs
    chunk_idxs = -1
    col_idxs = -1
  end subroutine nbrhd_get_nbrhd

  !> -------------------------------------------------------------------
  !> Private routines.

  subroutine find_chunk_nbrhds(cns, gd, chunks, cnbrhds)
    ! For each gcol in an iam-owning chunk, find its list of neighbors as
    ! sorted gcols.

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (SparseTriple), intent(out) :: cnbrhds

    integer, allocatable :: idxs(:), xs(:)
    integer :: nchunks, cid, ncols, gcol, i, cap, lcolid, cnt, ptr
    real(r8) :: lat, lon, xi, yi, zi, angle
    logical :: e

    if (cns%verbose > 0) print *, 'amb> nlcols', gd%nlcols
    nchunks = size(chunks)
    cap = gd%nlcols
    allocate(cnbrhds%xs(gd%nlcols), cnbrhds%yptr(gd%nlcols+1), cnbrhds%ys(cap))
    ! Get sorted iam-owning chunks' gcols.
    lcolid = 1
    do cid = 1, nchunks
       if (chunks(cid)%owner /= iam) cycle
       ncols = chunks(cid)%ncols
       e = assert(ncols >= 1, 'ncols')
       do i = 1, ncols
          cnbrhds%xs(lcolid) = chunks(cid)%gcol(i)
          lcolid = lcolid + 1
       end do
    end do
    e = assert(lcolid-1 == gd%nlcols, 'lcolid post')
    allocate(idxs(gd%nlcols), xs(gd%nlcols))
    call IndexSet(gd%nlcols, idxs)
    xs(:) = cnbrhds%xs(:)
    call IndexSort(gd%nlcols, idxs, xs)
    do i = 1, gd%nlcols
       cnbrhds%xs(i) = xs(idxs(i))
    end do
    deallocate(idxs, xs)
    ! Get each gcol's neighborhood.
    cnbrhds%yptr(1) = 1
    do lcolid = 1, gd%nlcols
       cnt = 0
       ptr = cnbrhds%yptr(lcolid)
       gcol = cnbrhds%xs(lcolid)
       cnt = find_gcol_nbrhd(gd, cns%max_angle, gcol, cnbrhds%ys, ptr, cap)
       cnbrhds%yptr(lcolid+1) = cnbrhds%yptr(lcolid) + cnt
    end do
    call array_realloc(cnbrhds%ys, cnbrhds%yptr(gd%nlcols+1)-1, cnbrhds%yptr(gd%nlcols+1)-1)
  end subroutine find_chunk_nbrhds

  subroutine make_cpe2nbrs(cns, gd, chunks, knuhcs, cpe2nbrs, exclude_existing_in)
    ! Make the map of chunk-owning pe to gcols, where the gcols are neighbors
    ! of a column in a chunk, each gcol belongs to a block that iam owns, and
    ! neighbor is defined in find_gcol_nbrhd. On output, cpe2nbrs has these
    ! entries:
    !     xs: sorted list of chunk-owning pes;
    !     yptr: pointers into ys;
    !     ys: for each pe, the list of iam-owning blocks' neighbors, as
    !         sorted gcols.
    ! If exclude_existing_in, which is true by default, exclude from ys any
    ! gcol that is already available to the pe from the regular comm pattern.

    use dyn_grid, only: get_gcol_block_cnt_d, get_block_owner_d, get_gcol_block_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    type (SparseTriple), intent(out) :: cpe2nbrs
    logical, optional, intent(in) :: exclude_existing_in

    integer, parameter :: cap_init = 128

    integer :: ptr, cap, gcol, block_cnt, blockids(1), bcids(1), cnt, ucnt, &
         i, j, pe, prev, acap, aptr, ng, max_ng, jprev, nupes, chunk_owner
    integer, pointer :: nbrhd(:), apes(:), agcols(:)
    integer, allocatable, dimension(:) :: idxs, pes, upes, gcols, ugcols, gidxs
    logical :: e, same, exclude_existing

    exclude_existing = .true.
    if (present(exclude_existing_in)) exclude_existing = exclude_existing_in
    ! Collect (pe,gcol) pairs where gcol is in one of my owned blocks and pe is
    ! the chunk owner.
    cap = cap_init
    acap = cap_init
    allocate(nbrhd(cap), idxs(cap), pes(cap), upes(cap), apes(acap), agcols(acap))
    aptr = 1
    do gcol = 1, gd%ngcols
       block_cnt = get_gcol_block_cnt_d(gcol)
       e = assert(block_cnt == 1, 'only block_cnt=1 is supported')
       call get_gcol_block_d(gcol, block_cnt, blockids, bcids)
       if (get_block_owner_d(blockids(1)) /= iam) cycle
       cnt = find_gcol_nbrhd(gd, cns%max_angle, gcol, nbrhd, 1, cap)
       if (cnt == 0) cycle
       if (cap > size(pes)) then
          deallocate(idxs, pes, upes)
          allocate(idxs(cap), pes(cap), upes(cap))
       end if
       e = assert(cnt <= cap, 'cap')
       do i = 1, cnt
          e = assert(nbrhd(i) >= 1 .and. nbrhd(i) <= gd%ngcols, 'nbrhd(i)')
          pes(i) = chunks(knuhcs(nbrhd(i))%chunkid)%owner
       end do
       call IndexSet(cnt, idxs(1:cnt))
       call IndexSort(cnt, idxs(1:cnt), pes(1:cnt))
       prev = pes(1)
       ucnt = 1
       upes(ucnt) = prev
       do i = 2, cnt
          if (pes(i) == prev) cycle
          prev = pes(i)
          ucnt = ucnt + 1
          upes(ucnt) = prev
       end do
       if (aptr + ucnt - 1 > acap) then
          acap = max(2*acap, aptr + ucnt - 1)
          e = assert(size(apes) >= aptr-1, 'apes size')
          call array_realloc(apes, aptr-1, acap)
          call array_realloc(agcols, aptr-1, acap)
       end if
       if (exclude_existing) chunk_owner = chunks(knuhcs(gcol)%chunkid)%owner
       do i = 1, ucnt
          if (exclude_existing .and. upes(i) == chunk_owner) cycle
          apes(aptr) = upes(i)
          agcols(aptr) = gcol
          aptr = aptr + 1
       end do
    end do
    deallocate(nbrhd, idxs, pes, upes)
    cnt = aptr - 1
    ! If we didn't find any, return with empty output.
    if (cnt == 0) then
       call SparseTriple_nullify(cpe2nbrs)
       return
    end if
    ! Count the number of unique pes and the max number of unique gcols per pe.
    allocate(idxs(cnt))
    call IndexSet(cnt, idxs)
    call IndexSort(cnt, idxs, apes)
    ucnt = 0
    prev = -1
    max_ng = 1
    do i = 1, cnt
       same = apes(idxs(i)) == prev
       if (same) ng = ng + 1
       if (.not. same .or. i == cnt) max_ng = max(max_ng, ng)
       if (same) cycle
       prev = apes(idxs(i))
       ucnt = ucnt + 1
       ng = 1
    end do
    nupes = ucnt
    ! Collect unique pes, set up pe -> unique gcols pointers, and collect
    ! unique gcols per pe.
    cap = cap_init
    allocate(cpe2nbrs%xs(nupes), cpe2nbrs%yptr(nupes+1), cpe2nbrs%ys(cap), &
         gidxs(max_ng), gcols(max_ng), ugcols(max_ng))
    cpe2nbrs%yptr(1) = 1
    i = 1
    do ucnt = 1, nupes
       cpe2nbrs%xs(ucnt) = apes(idxs(i))
       ng = 0
       do while (i <= cnt .and. apes(idxs(i)) == cpe2nbrs%xs(ucnt))
          ng = ng + 1
          gcols(ng) = agcols(idxs(i))
          i = i + 1
       end do
       call IndexSet(ng, gidxs)
       call IndexSort(ng, gidxs, gcols)
       cpe2nbrs%yptr(ucnt+1) = cpe2nbrs%yptr(ucnt)
       jprev = -1
       ptr = 0
       do j = 1, ng
          if (gcols(gidxs(j)) == jprev) cycle
          jprev = gcols(gidxs(j))
          ptr = ptr + 1
          ugcols(ptr) = jprev
          cpe2nbrs%yptr(ucnt+1) = cpe2nbrs%yptr(ucnt+1) + 1
       end do
       if (cpe2nbrs%yptr(ucnt+1)-1 > cap) then
          cap = max(2*cap, cpe2nbrs%yptr(ucnt+1)-1)
          call array_realloc(cpe2nbrs%ys, cpe2nbrs%yptr(ucnt)-1, cap)
       end if
       e = assert(cpe2nbrs%yptr(ucnt+1) - cpe2nbrs%yptr(ucnt) == ptr, 'yptr and ptr')
       cpe2nbrs%ys(cpe2nbrs%yptr(ucnt):cpe2nbrs%yptr(ucnt+1)-1) = ugcols(1:ptr)
    end do
    cnt = cpe2nbrs%yptr(nupes+1)-1
    call array_realloc(cpe2nbrs%ys, cnt, cnt) ! compact memory
    deallocate(apes, agcols, idxs, gidxs, gcols, ugcols)
    if (cns%verbose > 0) then
       print *,'amb> cpe2nbrs #pes',size(cpe2nbrs%xs)
       if (cns%verbose > 1) then
          do i = 1, size(cpe2nbrs%xs)
             print *,'amb> pe',cpe2nbrs%xs(i),cpe2nbrs%yptr(i+1)-cpe2nbrs%yptr(i)
          end do
       end if
    end if
  end subroutine make_cpe2nbrs

  subroutine make_dpe2nbrs(cns, gd, cnbrhds, dpe2nbrs, exclude_existing_in)
    ! Make the map of block-owning pe to gcols, where the gcols are neighbors
    ! of a column in a block, each gcol belongs to a chunk that iam owns, and
    ! neighbor is defined in find_gcol_nbrhd. On output, dpe2nbrs has these
    ! entries:
    !     xs: sorted list of block-owning pes;
    !     yptr: pointers into ys;
    !     ys: for each pe, the list of iam-owning chunks' neighbors, as
    !         sorted gcols.

    use dyn_grid, only: get_gcol_block_cnt_d, get_block_owner_d, get_gcol_block_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (SparseTriple), intent(in) :: cnbrhds
    type (SparseTriple), intent(out) :: dpe2nbrs
    logical, optional, intent(in) :: exclude_existing_in

    integer, pointer, dimension(:) :: unbrs, wrk(:)
    integer, allocatable, dimension(:) :: pes, idxs
    integer :: i, j, k, n, gcol, block_cnt, blockids(1), bcids(1), cnt, prev
    logical :: e, exclude_existing

    exclude_existing = .true.
    if (present(exclude_existing_in)) exclude_existing = exclude_existing_in
    e = assert(size(cnbrhds%yptr)-1 == gd%nlcols, 'nlcols')
    ! Unlike for cpe2nbrs, we are filling a 1-1 map, so we just need the unique
    ! neighbor gcols.
    call make_unique(cnbrhds%yptr(gd%nlcols+1)-1, cnbrhds%ys, unbrs)
    if (exclude_existing) then
       ! Filter out any gcol that iam's chunks' own, as these are already going
       ! to be communicated.
       allocate(wrk(size(unbrs)))
       wrk(:) = unbrs(:)
       n = 0
       do i = 1, size(unbrs)
          if (SparseTriple_in_xs(cnbrhds, wrk(i)) /= -1) cycle
          n = n + 1
          unbrs(n) = wrk(i)
       end do
       deallocate(wrk)
    else
       n = size(unbrs)
    end if
    if (cns%verbose > 0) &
         print *,'amb> dpe2nbrs', cnbrhds%yptr(gd%nlcols+1)-1, size(unbrs), n
    ! For each gcol, get the pe of owning block.
    allocate(pes(n), idxs(n))
    do i = 1, n
       gcol = unbrs(i)
       block_cnt = get_gcol_block_cnt_d(gcol)
       e = assert(block_cnt == 1, 'only block_cnt=1 is supported')
       call get_gcol_block_d(gcol, block_cnt, blockids, bcids)
       pes(i) = get_block_owner_d(blockids(1))
       e = assert(pes(i) >= 0 .and. pes(i) <= npes-1, 'dpe2nbrs pes')
    end do
    ! Count unique pes.
    call IndexSet(n, idxs)
    call IndexSort(n, idxs, pes)
    cnt = 0
    prev = -1
    do i = 1, n
       if (pes(idxs(i)) == prev) cycle
       cnt = cnt + 1
       prev = pes(idxs(i))
    end do
    ! Fill dpe2nbrs.
    allocate(dpe2nbrs%xs(cnt), dpe2nbrs%yptr(cnt+1), dpe2nbrs%ys(n))
    j = 1
    i = 1
    dpe2nbrs%yptr(j) = 1
    do j = 1, cnt
       dpe2nbrs%xs(j) = pes(idxs(i))
       k = 0
       do while (i <= n .and. pes(idxs(i)) == dpe2nbrs%xs(j))
          dpe2nbrs%ys(dpe2nbrs%yptr(j)+k) = unbrs(idxs(i))
          k = k + 1
          i = i + 1
       end do
       dpe2nbrs%yptr(j+1) = dpe2nbrs%yptr(j) + k
    end do
    e = assert(dpe2nbrs%yptr(cnt+1) == n+1, 'dpe2nbrs%yptr post')
    ! Sort each set of gcols.
    do j = 1, cnt
       n = dpe2nbrs%yptr(j+1) - dpe2nbrs%yptr(j)
       call IndexSet(n, idxs(1:n))
       unbrs(1:n) = dpe2nbrs%ys(dpe2nbrs%yptr(j):dpe2nbrs%yptr(j+1)-1)
       call IndexSort(n, idxs(1:n), unbrs(1:n))
       do i = 1, n
          dpe2nbrs%ys(dpe2nbrs%yptr(j)+i-1) = unbrs(idxs(i))
       end do
    end do
    deallocate(unbrs, idxs, pes)
    if (cns%verbose > 0) then
       print *,'amb> dpe2nbrs #pes',size(dpe2nbrs%xs)
       if (cns%verbose > 1) then
          do i = 1, size(dpe2nbrs%xs)
             print *,'amb> pe',dpe2nbrs%xs(i),dpe2nbrs%yptr(i+1)-dpe2nbrs%yptr(i)
          end do
       end if
    end if
  end subroutine make_dpe2nbrs

  function find_gcol_nbrhd(gd, max_angle, gcol, nbrhd, ptr, cap) result(cnt)
    ! Find all columns having center within max_angle of gcol. Append entries
    ! to nbrhd(ptr:), reallocating as necessary. cap is nbrhd's capacity at
    ! input, and it is updated when reallocation occurs. The gcols list is
    ! sorted.

    type (PhysGridData), intent(in) :: gd
    real(r8), intent(in) :: max_angle
    integer, intent(in) :: gcol, ptr
    integer, pointer, intent(inout) :: nbrhd(:)
    integer, intent(inout) :: cap

    integer, allocatable, dimension(:) :: idxs, buf
    integer :: cnt, j_lo, j_up, j, jl_lim, jl, jgcol, new_cap
    real(r8) :: lat, xi, yi, zi, angle
    logical :: e

    ! Get latitude range to search.
    lat = gd%clat_p(gd%lat_p(gcol))
    call latlon2xyz(lat, gd%clon_p(gd%lon_p(gcol)), xi, yi, zi)
    j_lo = upper_bound_or_in_range(gd%clat_p_tot, gd%clat_p, lat - max_angle)
    if (j_lo > 1) j_lo = j_lo - 1
    j_up = upper_bound_or_in_range(gd%clat_p_tot, gd%clat_p, lat + max_angle, j_lo)
    cnt = 0
    ! Check each point within this latitude range for distance.
    do j = j_lo, j_up
       if (j < gd%clat_p_tot) then
          jl_lim = gd%clat_p_idx(j+1) - gd%clat_p_idx(j)
       else
          jl_lim = gd%ngcols_p - gd%clat_p_idx(j) + 1
       end if
       do jl = 1, jl_lim
          e = assert(gd%clat_p_idx(j) + jl - 1 <= gd%ngcols_p, &
                     'gcol_nbrhd jgcol access')
          jgcol = gd%latlon_to_dyn_gcol_map(gd%clat_p_idx(j) + jl - 1)
          if (jgcol == -1 .or. jgcol == gcol) cycle
          angle = unit_sphere_angle(xi, yi, zi, &
               gd%clat_p(gd%lat_p(jgcol)), gd%clon_p(gd%lon_p(jgcol)))
          if (angle > max_angle) cycle
          if (ptr + cnt > cap) then
             new_cap = max(2*cap, ptr + cnt)
             call array_realloc(nbrhd, ptr+cnt-1, new_cap)
             cap = new_cap
          end if
          nbrhd(ptr+cnt) = jgcol
          cnt = cnt + 1
       end do
    end do
    ! Sort.
    allocate(idxs(cnt), buf(cnt))
    buf(1:cnt) = nbrhd(ptr:ptr+cnt-1)
    call IndexSet(cnt, idxs)
    call IndexSort(cnt, idxs, buf)
    do j = 1, cnt
       nbrhd(ptr+j-1) = buf(idxs(j))
    end do
    deallocate(idxs, buf)
  end function find_gcol_nbrhd

  subroutine make_comm_schedule(cns, gd, cpe2nbrs, dpe2nbrs)
    use dyn_grid, only: get_gcol_block_cnt_d, get_block_gcol_cnt_d, get_block_owner_d, &
         get_block_lvl_cnt_d, get_gcol_block_d

    type (ColumnNeighborhoods), intent(inout) :: cns
    type (PhysGridData), intent(in) :: gd
    type (SparseTriple), intent(in) :: cpe2nbrs, dpe2nbrs

    integer, allocatable :: ie2gid(:)
    type (IdMap) :: gid2ie
    integer :: i, j, k, ie, gid, nlblk, gcol, blockid(1), bcid(1), pecnt, glbcnt, &
         pe, numlev, ptr
    logical :: e

    ! We use local block IDs to keep our persistent arrays small. Get global <->
    ! local block ID maps.
    call get_local_blocks(ie2gid, gid2ie)
    nlblk = size(ie2gid)

    allocate(cns%blk_num(0:npes-1), cns%chk_num(0:npes-1), &
         cns%blk_offset(nlblk))

    ! Get column counts.
    do ie = 1, nlblk
       gid = ie2gid(i)
       k = get_block_gcol_cnt_d(gid)
       cns%blk_offset(ie)%ncol = k
       ! numrep is redundant wrt col, but it's useful
       allocate(cns%blk_offset(ie)%numlev(k), cns%blk_offset(ie)%numrep(k), &
            cns%blk_offset(ie)%col(k+1))
       cns%blk_offset(ie)%numrep(:) = 0
    end do
    ! Get repetition counts. A gcol in a block is in general in the
    ! neighborhoods of multiple chunks' gcols on multiple pes.
    do i = 1, size(cpe2nbrs%xs)
       do j = cpe2nbrs%yptr(i), cpe2nbrs%yptr(i+1)-1
          gcol = cpe2nbrs%ys(j) ! gcol in a chunk on pe
          call get_gcol_block_d(gcol, 1, blockid, bcid)
          e = assert(get_block_owner_d(blockid(1)) == iam, &
                     'comm_schedule: gcol is owned')
          k = binary_search(nlblk, gid2ie%id1, blockid(1))
          e = assert(k >= 1, 'comm_schedule: blockid is in map')
          ie = gid2ie%id2(k) ! local block ID providing data to the chunk
          e = assert(bcid(1) >= 1 .and. bcid(1) <= cns%blk_offset(ie)%ncol, &
                     'comm_schedule: bcid is in range')
          call incr(cns%blk_offset(ie)%numrep(bcid(1)+1))
       end do
    end do
    ! Allocate offset arrays.
    do ie = 1, nlblk
       gid = ie2gid(i)
       allocate(cns%blk_offset(ie)%os(sum(cns%blk_offset(ie)%numrep)))
       cns%blk_offset(ie)%col(1) = 1
       do i = 1, cns%blk_offset(ie)%ncol
          cns%blk_offset(ie)%col(i+1) = cns%blk_offset(ie)%col(i) + &
               cns%blk_offset(ie)%numrep(i)
       end do
       cns%blk_offset(ie)%numrep(:) = 0
    end do
    ! Get offsets and send counts.
    glbcnt = 0
    cns%blk_num(:) = 0
    cns%max_numrep = 0
    cns%max_numlev = 0
    do i = 1, size(cpe2nbrs%xs)
       pecnt = 0
       do j = cpe2nbrs%yptr(i), cpe2nbrs%yptr(i+1)-1
          gcol = cpe2nbrs%ys(j)
          call get_gcol_block_d(gcol, 1, blockid, bcid)
          k = binary_search(nlblk, gid2ie%id1, blockid(1))
          ie = gid2ie%id2(k)
          ptr = cns%blk_offset(ie)%col(bcid(1))
          k = cns%blk_offset(ie)%numrep(bcid(1))
          cns%blk_offset(ie)%os(ptr+k) = glbcnt
          cns%blk_offset(ie)%numrep(bcid(1)) = k + 1
          cns%max_numrep = max(cns%max_numrep, k + 1)
          numlev = get_block_lvl_cnt_d(gid, bcid(1))
          cns%blk_offset(ie)%numlev(bcid(1)) = numlev
          cns%max_numlev = max(cns%max_numlev, numlev)
          glbcnt = glbcnt + numlev
          pecnt = pecnt + numlev
       end do
       cns%blk_num(cpe2nbrs%xs(i)) = pecnt
    end do
    cns%blk_nrecs = glbcnt

    deallocate(ie2gid, gid2ie%id1, gid2ie%id2)

    allocate(cns%chk_offset(size(dpe2nbrs%ys)), cns%chk_numlev(size(dpe2nbrs%ys)))
    glbcnt = 0
    cns%chk_num(:) = 0
    do i = 1, size(dpe2nbrs%xs)
       pe = dpe2nbrs%xs(i)
       pecnt = 0
       do j = dpe2nbrs%yptr(i), dpe2nbrs%yptr(i+1)-1
          gcol = dpe2nbrs%ys(j)
          call get_gcol_block_d(gcol, 1, blockid, bcid)
          e = assert(get_block_owner_d(blockid(1)) == pe, &
                     'comm_schedule: gcol pe association')
          cns%chk_offset(j) = glbcnt
          numlev = get_block_lvl_cnt_d(blockid(1), bcid(1))
          cns%chk_numlev(j) = numlev
          glbcnt = glbcnt + numlev
          pecnt = pecnt + numlev
       end do
       cns%chk_num(pe) = pecnt
    end do
    cns%chk_nrecs = glbcnt
  end subroutine make_comm_schedule

  subroutine init_comm_data(cns, cd, phys_alltoall)
    use spmd_utils, only: pair, ceil2

    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommData), intent(out) :: cd
    integer, intent(in) :: phys_alltoall

    integer :: i, j, pe
    logical :: e

    cd%lopt = phys_alltoall
    if (cd%lopt < 0 .or. cd%lopt >= 4 .or. cd%lopt == 2) cd%lopt = 1
    cd%prev_record_size = -1
    allocate(cd%sndcnts(0:npes-1), cd%sdispls(0:npes-1), cd%rcvcnts(0:npes-1), &
         cd%rdispls(0:npes-1), cd%pdispls(0:npes-1))

    do j = 1, 2 ! count, then fill
       cd%dp_coup_steps = 0       
       do i = 1, ceil2(npes)-1
          pe = pair(npes, i, iam) ! pseudo-randomize order of comm partner pes
          if (pe < 0) cycle
          if (cns%blk_num(pe) > 0 .or. cns%chk_num(pe) > 0) then
             cd%dp_coup_steps = cd%dp_coup_steps + 1
             if (j == 2) cd%dp_coup_proc(cd%dp_coup_steps) = pe
          end if
       end do
       if (j == 1) allocate(cd%dp_coup_proc(cd%dp_coup_steps))
    end do

    if (cns%verbose > 0) print *,'amb> dp_coup_steps', cd%dp_coup_steps
  end subroutine init_comm_data

  subroutine make_comm_data(cns, cd, rcdsz)
#if defined SPMD
    use spmd_utils, only: mpicom
#endif

    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommData), intent(inout) :: cd
    integer, intent(in) :: rcdsz

    integer :: pe
    logical :: e
    
    e = assert(rcdsz >= 1, 'comm_data: valid record size')
    if (rcdsz == cd%prev_record_size) return
    cd%prev_record_size = rcdsz
    
    cd%sdispls(0) = 0
    cd%sndcnts(0) = rcdsz*cns%blk_num(0)
    do pe = 1, npes-1
       cd%sdispls(pe) = cd%sdispls(pe-1) + cd%sndcnts(pe-1)
       cd%sndcnts(pe) = rcdsz*cns%blk_num(pe)
    enddo

    cd%rdispls(0) = 0
    cd%rcvcnts(0) = rcdsz*cns%chk_num(0)
    do pe = 1, npes-1
       cd%rdispls(pe) = cd%rdispls(pe-1) + cd%rcvcnts(pe-1)
       cd%rcvcnts(pe) = rcdsz*cns%chk_num(pe)
    enddo

#if defined SPMD
    call mpialltoallint(cd%rdispls, 1, cd%pdispls, 1, mpicom)
#endif
  end subroutine make_comm_data

  subroutine get_local_blocks(ie2gid, gid2ie)
    ! id2gid is a list of iam's owned global block IDs in block local ID
    ! order. gid2ie%id1 is the sorted list of global block IDs, and gid2ie%id2
    ! is the list of corresponding local IDs.

    use dyn_grid, only: get_block_bounds_d, get_gcol_block_cnt_d, get_gcol_block_d, &
         get_block_gcol_d, get_block_owner_d, get_block_gcol_cnt_d

    integer, allocatable, intent(out) :: ie2gid(:)
    type (IdMap), intent(out) :: gid2ie

    integer, allocatable :: gcols(:)
    integer :: bf, bl, bid, cnt, pe, nid, blockid(1), bcid(1), ie(1), ngcols, i
    logical :: e

    call get_block_bounds_d(bf, bl)
    cnt = 0
    do bid = bf, bl
       pe = get_block_owner_d(bid)
       if (pe == iam) cnt = cnt + 1
    end do
    allocate(ie2gid(cnt), gcols(128))
    ! This seems a bit convoluted, but I'm not seeing an easier way to get block
    ! IDs in local ID order.
    do bid = bf, bl
       pe = get_block_owner_d(bid)
       if (pe /= iam) cycle
       ngcols = get_block_gcol_cnt_d(bid)
       if (ngcols < size(gcols)) then
          deallocate(gcols)
          allocate(gcols(ngcols))
       end if
       call get_block_gcol_d(bid, ngcols, gcols)
       nid = get_gcol_block_cnt_d(gcols(1))
       e = assert(nid == 1, 'only nid=1 is supported')
       call get_gcol_block_d(gcols(1), nid, blockid, bcid, ie)
       e = assert(ie(1) >= 1 .and. ie(1) <= cnt, 'ie in bounds')
       ie2gid(ie(1)) = bid
    end do
    deallocate(gcols)
    ! Now the opposite direction.
    allocate(gid2ie%id1(cnt), gid2ie%id2(cnt))
    call IndexSet(cnt, gid2ie%id2)
    gid2ie%id1(:) = ie2gid(:)
    call IndexSort(cnt, gid2ie%id2, gid2ie%id1)
    do i = 1, cnt
       gid2ie%id1(i) = ie2gid(gid2ie%id2(i))
    end do
  end subroutine get_local_blocks

  subroutine test_comm_schedule(cns)
    type (ColumnNeighborhoods), intent(in) :: cns
  end subroutine test_comm_schedule

  !> -------------------------------------------------------------------
  !> General utilities.

  function test(cond, message) result(out)
    ! Assertion that is always enabled, for use in unit tests.
    
    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    if (.not. cond) print *, 'amb> assert ', trim(message)
    out = cond
  end function test

  function assert(cond, message) result(out)
    ! Assertion that can be disabled.

    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    out = test(cond, message)
  end function assert

  function reldif(a, b) result(r)
    real(r8), intent(in) :: a, b
    real(r8) :: r

    r = abs(b - a)
    if (a == 0) return
    r = r/abs(a)
  end function reldif

  subroutine latlon2xyz(lat,lon,x,y,z)
    real(r8), intent(in) :: lat,lon
    real(r8), intent(out) :: x,y,z

    real(r8) :: sinl, cosl

    sinl = sin(lat)
    cosl = cos(lat);
    x = cos(lon)*cosl
    y = sin(lon)*cosl
    z = sinl
  end subroutine latlon2xyz

  function unit_sphere_angle(x1,y1,z1,lat,lon) result(angle)
    ! Angle between (x1,y1,z1) and (lat,lon).

    real(r8), intent(in) :: x1,y1,z1,lat,lon

    real(r8) :: x2,y2,z2,angle

    call latlon2xyz(lat,lon,x2,y2,z2)
    ! atan2(|v1 x v2|, v1 . v2)
    angle = atan2(sqrt((y1*z2 - y2*z1)**2 + (x2*z1 - x1*z2)**2 + (x1*y2 - x2*y1)**2), &
         x1*x2 + y1*y2 + z1*z2)
  end function unit_sphere_angle

  function upper_bound_or_in_range(n, a, val, k_in) result (k)
    ! Find k such that
    !   if k > 1 then a(k-1) <= val
    !   if k < n then           val < a(k)
    ! where a(1:n) has unique elements and is ascending. k_in is an optional
    ! hint.

    integer, intent(in) :: n
    integer, intent(in), optional :: k_in
    real(r8), intent(in) :: a(n), val

    integer :: lo, hi, k
    logical :: e

    k = 1
    if (present(k_in) .and. k_in >= 1 .and. k_in <= n) k = k_in
    if (val < a(k)) then
       lo = 1
       hi = k
    else
       lo = k
       hi = n
    end if
    do while (hi > lo + 1)
       k = (lo + hi)/2
       e = assert(k > lo .and. k < hi, 'upper_bound_or_in_range k')
       if (val < a(k)) then
          hi = k
       else
          lo = k
       end if
    end do
    k = hi
    e = assert((k == 1 .or. a(k-1) <= val) .and. (k == n .or. val < a(k)), &
         'upper_bound_or_in_range post')
  end function upper_bound_or_in_range

  function binary_search(n, a, val, k_in) result (k)
    ! Find position of val in a(1:n), or return -1 if val is not in a. k_in is
    ! an optional hint.

    integer, intent(in) :: n, a(n), val
    integer, intent(in), optional :: k_in

    integer :: lo, hi, k
    logical :: e

    k = 1
    if (present(k_in) .and. k_in >= 1 .and. k_in <= n) k = k_in
    if (val < a(k)) then
       lo = 1
       hi = k
    else
       lo = k
       hi = n
    end if
    do while (hi > lo + 1)
       k = (lo + hi)/2
       if (val < a(k)) then
          hi = k
       else
          lo = k
          if (a(k) == val) exit
       end if
    end do
    if (a(lo) == val) then
       k = lo
    else if (a(hi) == val) then
       k = hi
    else
       k = -1
    end if
    e = assert(k == -1 .or. (k >= 1 .and. k <= n), 'binary_search post')
  end function binary_search

  subroutine array_realloc(a, n, n_new)
    ! Reallocate a to size n_new, preserving the first min(n,n_new) values.

    integer, pointer, intent(inout) :: a(:)
    integer, intent(in) :: n, n_new

    integer, allocatable :: buf(:)
    integer :: i, n_min
    logical :: e

    e = assert(n_new >= 1 .and. size(a) >= n, 'array_realloc size(a)')
    if (n == 0) then
       deallocate(a)
       allocate(a(n_new))
       return
    end if
    n_min = min(n, n_new)
    allocate(buf(n_min))
    buf(1:n_min) = a(1:n_min)
    deallocate(a)
    allocate(a(n_new))
    a(1:n_min) = buf(1:n_min)
    deallocate(buf)
  end subroutine array_realloc

  subroutine make_unique(n, a, ua)
    ! On exit, ua is the sorted list of unique entries in a(1:n)

    use m_MergeSorts, only: IndexSet, IndexSort

    integer, intent(in) :: n, a(n)
    integer, pointer, intent(out) :: ua(:)

    integer, allocatable :: idxs(:)
    integer :: cnt, prev, i
    logical :: e

    e = assert(n > 0, 'make_unique: n > 0')
    allocate(idxs(n))
    call IndexSet(n, idxs)
    call IndexSort(n, idxs, a)
    ! Count unique entries.
    cnt = 1
    prev = a(idxs(1))
    do i = 2, n
       if (a(idxs(i)) == prev) cycle
       cnt = cnt + 1
       prev = a(idxs(i))
    end do
    ! Fill unique list.
    allocate(ua(cnt))
    cnt = 1
    prev = a(idxs(1))
    ua(cnt) = prev
    do i = 2, n
       if (a(idxs(i)) == prev) cycle
       cnt = cnt + 1
       prev = a(idxs(i))
       ua(cnt) = prev
    end do
    deallocate(idxs)
  end subroutine make_unique

  subroutine SparseTriple_nullify(st)
    type (SparseTriple), intent(out) :: st
    st%xs => null()
    st%yptr => null()
    st%ys => null()
  end subroutine SparseTriple_nullify

  subroutine SparseTriple_deallocate(st)
    type (SparseTriple), intent(out) :: st
    deallocate(st%xs, st%yptr, st%ys)
    call SparseTriple_nullify(st)
  end subroutine SparseTriple_deallocate

  function SparseTriple_in_xs(st, x) result(k)
    ! Find the position of x in st%xs(:) or -1 if not present.
    type (SparseTriple), intent(in) :: st
    integer, intent(in) :: x
    integer :: k
    k = binary_search(size(st%xs), st%xs, x, 1)
  end function SparseTriple_in_xs

  subroutine incr(i)
    integer, intent(inout) :: i
    i = i + 1
  end subroutine incr

  subroutine run_unit_tests()
    ! Unit tests for helper routines.

    use shr_const_mod, only: pi => shr_const_pi

    integer, parameter :: n = 4, b(n) = (/ -2, -1, 3, 7 /), &
         m = 6, c(m) = (/ 1, 3, 2, -1, 2, 3 /)
    real(r8), parameter :: a(n) = (/ -1.0_r8, 1.0_r8, 1.5_r8, 3.0_r8 /), &
         tol = epsilon(1.0_r8)

    integer, pointer :: uc(:)
    real(r8) :: lat1, lon1, lat2, lon2, x1, y1, z1, x2, y2, z2, angle
    integer :: k
    logical :: e

    k = upper_bound_or_in_range(n, a, -1.0_r8); e = test(k == 2, 'uboir 1')
    k = upper_bound_or_in_range(n, a, -1.0_r8, 2); e = test(k == 2, 'uboir 2')
    k = upper_bound_or_in_range(n, a, 3.0_r8, 2); e = test(k == n, 'uboir 3')
    k = upper_bound_or_in_range(n, a, 3.0_r8, -11); e = test(k == n, 'uboir 4')
    k = upper_bound_or_in_range(n, a, 1.2_r8); e = test(k == 3, 'uboir 5')

    k = binary_search(n, b, -22, -5); e = test(k == -1, 'binsrc 1')
    k = binary_search(n, b, -2); e = test(k == 1, 'binsrc 2')
    k = binary_search(n, b, 3, 2); e = test(k == 3, 'binsrc 3')
    k = binary_search(n, b, 7); e = test(k == 4, 'binsrc 4')
    k = binary_search(n, b, 7, 4); e = test(k == 4, 'binsrc 5')
    k = binary_search(n, b, 7, 5); e = test(k == 4, 'binsrc 6')
    k = binary_search(n, b, 0, 15); e = test(k == -1, 'binsrc 7')

    lat1 = -pi/3
    lon1 = pi/2
    call latlon2xyz(lat1, lon1, x1, y1, z1)
    lat2 = lat1 - 0.1_r8
    lon2 = lon1
    angle = unit_sphere_angle(x1, y1, z1, lat2, lon2)
    e = test(reldif(0.1_r8,angle) <= 10*tol, 'usa 1')

    call make_unique(m, c, uc)
    e = test(size(uc) == 4, 'uc length')
    e = test(uc(1) == -1, 'first(uc)')
    e = test(uc(4) == 3, 'last(uc)')
    do k = 2, size(uc)
       e = test(uc(k) > uc(k-1), 'uc sorted')
    end do
    deallocate(uc)

    k = 3
    call incr(k)
    e = test(k == 4, 'incr')
  end subroutine run_unit_tests

end module phys_grid_nbrhd
