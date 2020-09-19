module phys_grid_nbrhd
  ! Communication schedule and data structures to supplement a column with data
  ! in its neighborhood for use in scale-aware physics parameterizations.
  !
  ! Parameterization usage.
  !   In a parameterization, to get a column's neighborhood, make these calls,
  ! where icol is the current column of the chunk:
  !     nbrhd_size = nbrhd_get_nbrhd_size(state%lchnk, icol)
  !     ! size(icols) must be >= nbrhd_size
  !     call nbrhd_get_nbrhd(state%lchnk, icol, icols)
  ! Now access data in the neighborhood like this:
  !     state_nbrhd%X(icols(i),...),
  ! where state_nbrhd is the extra physics_state containing neighborhood data
  ! and 1 <= i <= nbrhd_size and X is a physics_state field.
  !   Not all state is necessarily available for communication efficiency. Call
  !     nbrhd_pcnst = nbrhd_get_option_pcnst()
  ! to determine the range 1:nbrhd_pcnst of 1:pcnst for which there are
  ! neighborhood data. T, om, uv, ps, phis are always available.
  !
  ! Namelist.
  !   This module has the following namelist options:
  !     &cam_inparm
  !      phys_nbrhd_degrees = 0.0d0   ! Radius of neighborhood in degrees
  !      phys_nbrhd_pcnst = pcnst     ! Share 1:phys_nbrhd_pcnst tracers
  !      phys_nbrhd_verbose = 0       ! 0 (none), 1 (a little), 2, 3 (a lot)
  !      phys_nbrhd_test = 0          ! 0 (none), 1, 2 (comprehensive tests)
  !     /
  ! The default value 0 of phys_nbrhd_degrees disables the neighborhood
  ! feature. When enabled, phys_nbrhd_pcnst must be >= 1 and <=
  ! pcnst. phys_nbrhd_verbose = 1 or 2 shows neighborhood size and rank
  ! connectivity data, with more data output for 2. phys_nbrhd_test = 1 runs
  ! small unit tests, and phys_nbrhd_test = 2 runs several communication rounds
  ! to test for expected communicated values.
  !
  ! phys_nbrhd_degrees is the most important parameter. It specifies the
  ! neighborhood radius in degrees. For physics column i, any physics column j
  ! such that angle(lat_i, lon_i, lat_j, lon_j) <= phys_nbrhd_degrees is
  ! included in the neighborhood of i. This relationship is symmetric: i is in
  ! the neighborhood of j.
  !
  ! Implementation notes.
  !   nbrhdchunk is 0 if neighborhoods are not active; this is standard
  ! behavior. It is 1 if they are. It is used to augment the range
  ! begchunk:endchunk by one, like this: begchunk:endchunk+nbrhdchunk.
  !   We keep endchunk the same as it was because most begchunk:endchunk
  ! quantities and loops are unchanged. We add an extra lchunk, chunk, and
  ! physics_state.
  !   There are three data structures outside of this module that are modified:
  ! extra chunk, lchunk, and phys_state. All three have arrays whose entries are
  ! 1-1 with each other. The column order of these arrays is: [sorted received
  ! gcols, sorted already-present gcols]. The already-present gcols must be
  ! duplicated because parameterizations update local state; the duplicate is
  ! not updated but rather reflects state at the last comm round. Consider the
  ! alternative of not duplicating these columns: the simulation would not be
  ! BFB-invariant to pe layout. Neighborhood physics state can be updated only
  ! through neighborhood communication rounds.
  !   A physics parameterization accesses the neighborhood data through the
  ! extra phys_state. The extra lchunk and chunk support phys_grid-type queries.
  !   There are two supported communication schedules:
  !   1. Dynamics blocks -> nbrhd columns. This mimics the d_p_coupling comm
  ! round and is at least as efficient as 2. In the code, this is indicated by
  ! owning_blocks = .true.
  !   2. There may be a desire to update nbrhd state during a physics time
  ! step. The second schedule supports this: owned-columns -> nbrhd
  ! columns. This is the only correct way to update nbrhd columns during a
  ! physics time step. In the code, owning_blocks = .false. If this pattern is
  ! ever used, to fully update physics_state, the equivalent of dp_coupling's
  ! derived_state must be called on the extra physics_state chunk.
  !   Performance testing shows 2 is actually the way to go
  ! always. phys_loadbalance's pattern makes the c2c pattern comm with fewer
  ! pes.
  !   In dp_coupling, we call derived_physics on the extra physics_state, but we
  ! omit the diagnostic energy calculations and so don't need an extra
  ! physics_tend. Here the goal is to provide full physics_state data for the
  ! neighborhood.
  !   In physconst, we allocate rairv for the extra chunk because it is used in
  ! derived_physics in the call to geopotential_t. But we ignore the other
  ! physconst arrays because these are used only in calculations in which the
  ! extra chunk is not involved.
  !
  ! AMB 2020/09 Initial

  use spmd_utils, only: iam, masterproc, npes
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile, only: iulog
  use ppgrid, only: pver, begchunk, endchunk, nbrhdchunk
  use m_MergeSorts, only: IndexSet, IndexSort
  use phys_grid_types

  implicit none
  private

  type PhysGridData ! collect phys_grid module data we need
     integer :: clat_p_tot, nlcols, ngcols, ngcols_p, phys_alltoall
     integer, pointer, dimension(:) :: lat_p, lon_p, latlon_to_dyn_gcol_map, clat_p_idx
     real(r8), pointer, dimension(:) :: clat_p, clon_p
  end type PhysGridData

  type SparseTriple
     ! xs(i) maps to ys(yptr(i):yptr(i+1)-1)
     integer, allocatable :: xs(:), yptr(:), ys(:)
  end type SparseTriple

  type IdMap
     integer, allocatable :: id1(:), id2(:)
  end type IdMap

  type Offsets
     integer :: ncol
     ! Offsets for the columns in a block or chunk. Each one can have a
     ! different number of repetitions in the send buffer than the others.
     integer, allocatable :: numlev(:), numrep(:)
     ! os(col(i):col(i+1)-1) are offsets for all repetitions of a column.
     integer, allocatable :: col(:), os(:)
  end type Offsets

  type CommData
     integer, allocatable, dimension(:) :: sndcnts, sdispls, rcvcnts, rdispls, &
          pdispls, dp_coup_proc
     integer :: lopt, prev_record_size, dp_coup_steps
  end type CommData

  type ColumnDesc
     ! The local chunk ID and column of the chunk for this column.
     integer :: lcid, icol
  end type ColumnDesc

  type ColumnCopies
     ! The first slot in the nbhrd chunk to write copies is one past this.
     integer :: num_recv_col
     type (ColumnDesc), allocatable :: cols(:) ! list of copy sources
  end type ColumnCopies

  type ChunkDesc
     ! nbrhd(col(i):col(i)-1) is the neighborhood of column i in this chunk. The
     ! range is an index into the list of ColumnDescs, idx2cd.
     integer, allocatable :: col(:), nbrhd(:)
  end type ChunkDesc

  type CommSchedule
     integer :: max_numrep, max_numlev, snd_nrecs, rcv_nrecs
     integer, allocatable :: snd_num(:), rcv_num(:)
     type (Offsets), allocatable :: snd_offset(:)
     integer, allocatable :: rcv_numlev(:), rcv_offset(:)
  end type CommSchedule

  ! A neighborhood is a list of gcols neighboring a column. Neighborhoods,
  ! plural, is a list of these of lists.
  type ColumnNeighborhoods
     logical :: b2c_on, c2c_on
     integer :: verbose, test, pcnst, nchunks, extra_chunk_ncol
     ! Radian angle defining neighborhood. Defines max between a column center
     ! and column centers within its neighborhood.
     real(r8) :: max_angle
     ! For each gcol in iam's chunks, the list of gcols in its neighborhood,
     ! excluding itself.
     type (SparseTriple) :: chk_nbrhds
     ! Local <-> global block IDs, chunks
     integer, allocatable :: ie2bid(:), l2cid(:)
     type (IdMap) :: bid2ie
     ! Communication data. In b2cs, snd_offset is index using local block ID,
     ! not global as in phys_grid.
     type (CommSchedule) :: b2cs, c2cs
     type (CommData) :: b2cd, c2cd
     ! Neighborhood API data.
     type (ColumnCopies) :: cc
     ! c2n(lcid)%nbrhd(c2n(lcid)%col(icol) : c2n(lcid)%col(icol)-1) is the
     ! neighborhood of column icol in lchunk lcid. The value of this array is
     ! icol values into the extra chunk.
     type (ChunkDesc), allocatable :: c2n(:)
  end type ColumnNeighborhoods

  type (ColumnNeighborhoods), private :: cns

  public :: &
       ! Namelist
       nbrhd_defaultopts, &
       nbrhd_setopts, &
       ! phys_grid initialization
       nbrhd_init, &
       nbrhd_init_extra_chunk, &
       ! dp_coupling communication
       nbrhd_block_to_chunk_sizes, &
       nbrhd_block_to_chunk_send_pters, &
       nbrhd_transpose_block_to_chunk, &
       nbrhd_block_to_chunk_recv_pters, &
       ! chunk-to-chunk communication
       nbrhd_chunk_to_chunk_sizes, &
       nbrhd_chunk_to_chunk_send_pters, &
       nbrhd_transpose_chunk_to_chunk, &
       nbrhd_chunk_to_chunk_recv_pters, &
       ! post-communication copying
       nbrhd_get_num_copies, &
       nbrhd_get_copy_idxs, &
       ! API for parameterizations
       nbrhd_get_nbrhd_size, &
       nbrhd_get_nbrhd, &
       ! Options
       nbrhd_get_option_angle, &
       nbrhd_get_option_pcnst, &
       nbrhd_get_option_test, &
       nbrhd_get_option_block_to_chunk_on, &
       nbrhd_get_option_chunk_to_chunk_on, &
       ! For API testing in phys_grid_nbrhd_util
       nbrhd_get_ie2bid, &
       latlon2xyz, unit_sphere_angle

contains

  subroutine nbrhd_defaultopts(phys_nbrhd_degrees_out, phys_nbrhd_pcnst_out, &
       phys_nbrhd_verbose_out, phys_nbrhd_test_out)
    use constituents, only: pcnst
    use dimensions_mod, only: ne
    
    real(r8), optional, intent(out) :: &
         ! Maximum angle in degrees between a column and a neighbor, measured at
         ! column centers, the (lat,lon) from physics_state.
         phys_nbrhd_degrees_out
    integer , optional, intent(out) :: &
         ! Number of constituents to communicate, 1:phys_nbrhd_pcnst,
         ! phys_nbrhd_pcnst <= pcnst.
         phys_nbrhd_pcnst_out, &
         ! Verbosity level, 0 for no output, >0 for some.
         phys_nbrhd_verbose_out, &
         ! Test level, 0 for none, > 0 for some.
         phys_nbrhd_test_out

    ! Default the neighborhood diameter to 0 degrees, which means no nbrhd.
    if (present(phys_nbrhd_degrees_out)) phys_nbrhd_degrees_out = 0
    if (present(phys_nbrhd_pcnst_out  )) phys_nbrhd_pcnst_out   = pcnst
    if (present(phys_nbrhd_verbose_out)) phys_nbrhd_verbose_out = 0
    if (present(phys_nbrhd_test_out   )) phys_nbrhd_test_out    = 0
  end subroutine nbrhd_defaultopts

  subroutine nbrhd_setopts(phys_nbrhd_degrees_in, phys_nbrhd_pcnst_in, &
       phys_nbrhd_verbose_in, phys_nbrhd_test_in)
    use constituents, only: pcnst
    use shr_const_mod, only : pi => shr_const_pi

    real(r8), optional, intent(in) :: phys_nbrhd_degrees_in
    integer , optional, intent(in) :: phys_nbrhd_pcnst_in, phys_nbrhd_verbose_in, &
         phys_nbrhd_test_in

    cns%max_angle = 0
    cns%pcnst = pcnst
    cns%verbose = 0
    cns%test = 0

    if (present(phys_nbrhd_degrees_in)) then
       cns%max_angle = phys_nbrhd_degrees_in * (pi/180._r8)
       if (cns%max_angle < 0) then
          if (masterproc) then
             write(iulog,*) 'nbrhd_setopts: ERROR: phys_nbrhd_degrees=', &
                  phys_nbrhd_degrees_in, &
                  ' is out of range; must be >= 0; setting to 0'
          end if
          cns%max_angle = 0
       end if
    end if

    if (cns%max_angle == 0) return

    if (cns%max_angle > 0) nbrhdchunk = 1

    if (present(phys_nbrhd_pcnst_in)) then
       cns%pcnst = phys_nbrhd_pcnst_in
       if (cns%pcnst < 1 .or. cns%pcnst > pcnst) then
          if (cns%pcnst < 1) cns%pcnst = pcnst
          if (cns%pcnst > pcnst) cns%pcnst = pcnst
          if (masterproc) then
             write(iulog,*) 'nbrhd_setopts: ERROR: phys_nbrhd_pcnst=', &
                  phys_nbrhd_pcnst_in, &
                  ' is out of range; must be in 1:pcnst; setting to', &
                  cns%pcnst
          end if
       end if
    end if

    if (present(phys_nbrhd_verbose_in)) cns%verbose = max(0, phys_nbrhd_verbose_in)
    if (present(phys_nbrhd_test_in)) cns%test = max(0, phys_nbrhd_test_in)

    if (cns%verbose > 0 .and. masterproc) then
       write(iulog,'(a,es13.4,a,i4,a,i2,a,i2)') 'nbr> nbrhd_setopts: angle', &
            cns%max_angle * (180._r8/pi), ' degrees; pcnst', cns%pcnst, &
            '; verbosity', cns%verbose, '; test', cns%test
    end if
  end subroutine nbrhd_setopts

  subroutine nbrhd_init(clat_p_tot, clat_p_idx, clat_p, clon_p, lat_p, lon_p, &
       latlon_to_dyn_gcol_map, nlcols, ngcols, ngcols_p, nchunks, chunks, chunk_extra, &
       knuhcs, phys_alltoall)
    use constituents, only: pcnst

    integer, intent(in) :: clat_p_tot, nlcols, ngcols, ngcols_p, phys_alltoall, nchunks
    integer, target, dimension(:), intent(in) :: clat_p_idx, lat_p, lon_p, &
         latlon_to_dyn_gcol_map
    real(r8), target, dimension(:), intent(in) :: clat_p, clon_p
    type (chunk), intent(in) :: chunks(:)
    type (chunk), intent(out) :: chunk_extra
    type (knuhc), intent(in) :: knuhcs(:)

    type (PhysGridData) :: gd
    logical :: call_init_chunk, e

    if (cns%test > 0) call run_unit_tests()

    e = assert(cns%pcnst >= 1, 'nbrhd%pcnst must be >= 1 to include qv')

    cns%b2c_on = .false. ! nbrhd_d_p_coupling
    cns%c2c_on = .true.  ! nbrhd_p_p_coupling
    cns%nchunks = nchunks

    nbrhdchunk = 1

    gd%clat_p_tot = clat_p_tot
    gd%nlcols = nlcols; gd%ngcols = ngcols; gd%ngcols_p = ngcols_p
    gd%phys_alltoall = phys_alltoall
    gd%lat_p => lat_p; gd%lon_p => lon_p
    gd%clat_p_idx => clat_p_idx; gd%latlon_to_dyn_gcol_map => latlon_to_dyn_gcol_map
    gd%clat_p => clat_p; gd%clon_p => clon_p

    call find_chunk_nbrhds(cns, gd, chunks, cns%chk_nbrhds)
    if (cns%test > 0) call test_nbrhds(cns, gd)
    if (cns%verbose > 0) call histogram_nbrhds(cns%chk_nbrhds)

    call_init_chunk = .true.
    ! Dynamics blocks -> nbrhd columns for use in dp_coupling.F90.
    if (cns%b2c_on) then
       ! We use local block IDs to keep our persistent arrays small. Get global
       ! <-> local block ID maps.
       call make_ie2bid(cns%ie2bid, cns%bid2ie)
       call nbrhd_inith(cns, gd, chunks, knuhcs, .true., call_init_chunk, &
            chunk_extra, cns%b2cs, cns%b2cd)
       call_init_chunk = .false.
    end if

    ! Owning chunks -> nbrhd columns for use in additional rounds within a
    ! physics time step.
    if (cns%c2c_on) then
       call make_l2cid(cns%nchunks, chunks, cns%l2cid)
       call nbrhd_inith(cns, gd, chunks, knuhcs, .false., call_init_chunk, &
            chunk_extra, cns%c2cs, cns%c2cd)
    end if
  end subroutine nbrhd_init

  subroutine nbrhd_init_extra_chunk(chks, lchks, chk, lchk)
    type (chunk), intent(in) :: chks(:)
    type (lchunk), intent(in) :: lchks(begchunk:endchunk+nbrhdchunk)
    type (chunk), intent(inout) :: chk
    type (lchunk), intent(out) :: lchk

    integer :: ncol_prev, ncol_add, icol, i, cid, lcid, dicol
    logical :: e

    if (cns%test > 0) then
       do lcid = begchunk+1, endchunk
          e = assert(lchks(lcid)%cid > lchks(lcid-1)%cid, 'lcid, cid sorted')
       end do
    end if

    ! Map native local chunk columns to duplicate ones.
    call make_cc(cns, lchks, cns%cc)

    ! Append the duplicates to the nbrhd chunk.
    ncol_prev = cns%extra_chunk_ncol
    e = assert(cns%cc%num_recv_col == ncol_prev, 'num_recv_col')
    ncol_add = size(cns%cc%cols)
    if (cns%verbose > 1) write(iulog,*) 'nbr> extra:', ncol_prev, ncol_add
    cns%extra_chunk_ncol = ncol_prev + ncol_add
    chk%ncols = cns%extra_chunk_ncol
    call array_realloc(chk%gcol, ncol_prev, chk%ncols)
    call array_realloc(chk%lat , ncol_prev, chk%ncols)
    call array_realloc(chk%lon , ncol_prev, chk%ncols)
    do i = 1, ncol_add
       lcid = cns%cc%cols(i)%lcid
       icol = cns%cc%cols(i)%icol
       cid = lchks(lcid)%cid
       dicol = ncol_prev + i
       chk%gcol(dicol) = chks(cid)%gcol(icol)
       chk%lat (dicol) = chks(cid)%lat (icol)
       chk%lon (dicol) = chks(cid)%lon (icol)
    end do

    lchk%ncols = chk%ncols
    lchk%cid = cns%nchunks + 1
    lchk%cost = -1
    allocate(lchk%gcol(lchk%ncols), lchk%area(lchk%ncols), lchk%wght(lchk%ncols))
    lchk%gcol(:) = chk%gcol(:)
    ! area and wght will be set in phys_grid after this call.

    call make_c2n(cns, lchks, cns%c2n)
    if (cns%test > 0) call test_c2n(cns, lchks)
  end subroutine nbrhd_init_extra_chunk

  subroutine nbrhd_block_to_chunk_sizes(block_buf_nrecs, chunk_buf_nrecs, &
       max_numlev, max_numrep, num_recv_col)
    ! max_numlev, max_numrep are max sizes for pter arrays. num_recv_col is the
    ! number of received columns.

    integer, intent(out) :: block_buf_nrecs, chunk_buf_nrecs, max_numlev, &
         max_numrep, num_recv_col

    call comm_sizes(cns, cns%b2cs, block_buf_nrecs, chunk_buf_nrecs, max_numlev, &
         max_numrep, num_recv_col)
  end subroutine nbrhd_block_to_chunk_sizes

  subroutine nbrhd_block_to_chunk_send_pters(ie, icol, rcdsz, numlev, numrep, ptr)
    ! Set up the pointer array for column icol of block having local block ID
    ! ie. rcdsz is the record size. On output, ptr(1:numlev, 1:numrep) is filled
    ! with offsets. Since the map from dynamics blocks to chunks is 1-many,
    ! numrep is >= 1 and not simply 1. This is different than in the basic
    ! phys_grid send_pters routine.

    integer, intent(in) :: ie, icol, rcdsz
    integer, intent(out) :: numlev, numrep
    integer, intent(out) :: ptr(:,:) ! >= max_numlev x >= max_numrep

    call comm_send_pters(cns%b2cs, ie, icol, rcdsz, numlev, numrep, ptr)
  end subroutine nbrhd_block_to_chunk_send_pters

  subroutine nbrhd_transpose_block_to_chunk(rcdsz, blk_buf, chk_buf)
    ! If running on just one pe or SPMD is not defined, then there are no comm
    ! data, so this routine does nothing, and comm data have size 0.

    integer, intent(in) :: rcdsz
    real(r8), intent(in) :: blk_buf(rcdsz*cns%b2cs%snd_nrecs)
    real(r8), intent(out) :: chk_buf(rcdsz*cns%b2cs%rcv_nrecs)

    call comm_transpose(cns, cns%b2cs, cns%b2cd, rcdsz, blk_buf, chk_buf)
  end subroutine nbrhd_transpose_block_to_chunk

  subroutine nbrhd_block_to_chunk_recv_pters(icol, rcdsz, numlev, ptr)
    integer, intent(in) :: icol, rcdsz
    integer, intent(out) :: numlev
    integer, intent(out) :: ptr(:) ! >= max_numlev

    call comm_recv_pters(cns%b2cs, icol, rcdsz, numlev, ptr)
  end subroutine nbrhd_block_to_chunk_recv_pters

  subroutine nbrhd_chunk_to_chunk_sizes(block_buf_nrecs, chunk_buf_nrecs, &
       max_numlev, max_numrep, num_recv_col)
    integer, intent(out) :: block_buf_nrecs, chunk_buf_nrecs, max_numlev, &
         max_numrep, num_recv_col

    call comm_sizes(cns, cns%c2cs, block_buf_nrecs, chunk_buf_nrecs, max_numlev, &
         max_numrep, num_recv_col)
  end subroutine nbrhd_chunk_to_chunk_sizes

  subroutine nbrhd_chunk_to_chunk_send_pters(lcid, icol, rcdsz, numlev, numrep, ptr)
    integer, intent(in) :: lcid, icol, rcdsz
    integer, intent(out) :: numlev, numrep
    integer, intent(out) :: ptr(:,:) ! >= max_numlev x >= max_numrep

    call comm_send_pters(cns%c2cs, lcid-begchunk+1, icol, rcdsz, numlev, numrep, ptr)
  end subroutine nbrhd_chunk_to_chunk_send_pters

  subroutine nbrhd_transpose_chunk_to_chunk(rcdsz, snd_buf, rcv_buf)
    integer, intent(in) :: rcdsz
    real(r8), intent(in) :: snd_buf(rcdsz*cns%c2cs%snd_nrecs)
    real(r8), intent(out) :: rcv_buf(rcdsz*cns%c2cs%rcv_nrecs)

    call comm_transpose(cns, cns%c2cs, cns%c2cd, rcdsz, snd_buf, rcv_buf)
  end subroutine nbrhd_transpose_chunk_to_chunk

  subroutine nbrhd_chunk_to_chunk_recv_pters(icol, rcdsz, numlev, ptr)
    integer, intent(in) :: icol, rcdsz
    integer, intent(out) :: numlev
    integer, intent(out) :: ptr(:) ! >= max_numlev

    call comm_recv_pters(cns%c2cs, icol, rcdsz, numlev, ptr)
  end subroutine nbrhd_chunk_to_chunk_recv_pters

  function nbrhd_get_num_copies() result(n)
    integer :: n

    n = size(cns%cc%cols)
  end function nbrhd_get_num_copies

  subroutine nbrhd_get_copy_idxs(i, lcid, icol, icole)
    integer, intent(in) :: i
    integer, intent(out) :: lcid, icol, icole

    logical :: e

    e = assert(i >= 1 .and. i <= nbrhd_get_num_copies(), 'copy_idxs: i')
    lcid = cns%cc%cols(i)%lcid
    icol = cns%cc%cols(i)%icol
    icole = cns%cc%num_recv_col + i
  end subroutine nbrhd_get_copy_idxs

  function nbrhd_get_nbrhd_size(lcid, icol) result(n)
    integer, intent(in) :: lcid, icol
    integer :: n

    logical :: e

    e = assert(lcid >= begchunk .and. lcid <= endchunk, 'nbrhd_size: lcid')
    e = assert(icol >= 1 .and. icol <= size(cns%c2n(lcid)%col)-1, 'nbrhd_size: icol')
    n = cns%c2n(lcid)%col(icol+1) - cns%c2n(lcid)%col(icol)
  end function nbrhd_get_nbrhd_size

  subroutine nbrhd_get_nbrhd(lcid, icol, icols)
    integer, intent(in) :: lcid, icol
    integer, dimension(:), intent(out) :: icols

    integer :: n

    n = nbrhd_get_nbrhd_size(lcid, icol)
    icols(1:n) = cns%c2n(lcid)%nbrhd(cns%c2n(lcid)%col(icol) : &
                                     cns%c2n(lcid)%col(icol+1)-1)
  end subroutine nbrhd_get_nbrhd

  function nbrhd_get_option_angle() result(angle)
    real(r8) :: angle
    angle = cns%max_angle
  end function nbrhd_get_option_angle

  function nbrhd_get_option_pcnst() result(n)
    integer :: n
    n = cns%pcnst
  end function nbrhd_get_option_pcnst

  function nbrhd_get_option_test() result(test)
    integer :: test
    test = cns%test
  end function nbrhd_get_option_test

  function nbrhd_get_option_block_to_chunk_on() result(on)
    logical :: on
    on = cns%b2c_on
  end function nbrhd_get_option_block_to_chunk_on

  function nbrhd_get_option_chunk_to_chunk_on() result(on)
    logical :: on
    on = cns%c2c_on
  end function nbrhd_get_option_chunk_to_chunk_on

  function nbrhd_get_ie2bid(ie) result(bid)
    integer, intent(in) :: ie
    integer :: bid
    bid = cns%ie2bid(ie)
  end function nbrhd_get_ie2bid

  !> -------------------------------------------------------------------
  !> Private routines.

  subroutine nbrhd_inith(cns, gd, chunks, knuhcs, owning_blocks, call_init_chunk, &
       chunk_extra, cs, cd)
    type (ColumnNeighborhoods), intent(inout) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    logical, intent(in) :: owning_blocks, call_init_chunk
    type (chunk), intent(inout) :: chunk_extra
    type (CommSchedule), intent(out) :: cs
    type (CommData), intent(out) :: cd

    type (SparseTriple) :: rpe2nbrs, spe2nbrs
    logical :: e

    call make_rpe2nbrs(cns, gd, chunks, knuhcs, rpe2nbrs, owning_blocks)
    call make_spe2nbrs(cns, gd, chunks, knuhcs, cns%chk_nbrhds, spe2nbrs, owning_blocks)
    call make_comm_schedule(cns, gd, chunks, knuhcs, rpe2nbrs, spe2nbrs, cs, owning_blocks)
    call init_comm_data(cns, cs, cd, gd%phys_alltoall)
    e = assert(cd%dp_coup_steps >= &
               ! -1 accounts for pe = iam
               min(size(rpe2nbrs%xs), size(spe2nbrs%xs)) - 1, &
               'init: dp_coup_steps')
    if (call_init_chunk) then
       cns%extra_chunk_ncol = size(spe2nbrs%ys)
       call init_chunk(cns, gd, spe2nbrs%ys, chunk_extra)
    else
       e = assert(cns%extra_chunk_ncol == size(spe2nbrs%ys), 'inith: extra_chunk_ncol')
    end if
    if (cns%test > 1) then
       call test_comm_schedule(cns, gd, chunks, knuhcs, rpe2nbrs, spe2nbrs, cd, cs, &
            owning_blocks)
    end if
    call SparseTriple_deallocate(rpe2nbrs)
    call SparseTriple_deallocate(spe2nbrs)
  end subroutine nbrhd_inith

  subroutine find_chunk_nbrhds(cns, gd, chunks, cnbrhds)
    ! For each gcol in an iam-owning chunk, find its list of neighbors as
    ! sorted gcols.

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (SparseTriple), intent(out) :: cnbrhds

    integer, allocatable :: idxs(:), xs(:)
    integer :: cid, ncols, gcol, i, cap, lcolid, cnt, ptr
    real(r8) :: lat, lon, xi, yi, zi, angle
    logical :: e

    if (cns%verbose > 1) write(iulog,*) 'nbr> nlcols', gd%nlcols
    cap = gd%nlcols
    allocate(cnbrhds%xs(gd%nlcols), cnbrhds%yptr(gd%nlcols+1), cnbrhds%ys(cap))
    ! Get sorted iam-owning chunks' gcols.
    lcolid = 1
    do cid = 1, cns%nchunks
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

  subroutine make_rpe2nbrs(cns, gd, chunks, knuhcs, rpe2nbrs, owning_blocks)
    ! Make the map of chunk-owning pe ((r)eceiving (pe)s) to gcols, where the
    ! gcols are neighbors of a column in a chunk, each gcol belongs to a block
    ! or chunk that iam owns, and neighbor is defined in find_gcol_nbrhd. On
    ! output, rpe2nbrs has these entries:
    !     xs: sorted list of chunk-owning pes;
    !     yptr: pointers into ys;
    !     ys: for each pe, the list of iam-owning neighbors, as sorted gcols.
    ! Exclude from ys any gcol that is already available to the pe from the
    ! regular comm pattern.

    use dyn_grid, only: get_block_owner_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    type (SparseTriple), intent(out) :: rpe2nbrs
    logical, intent(in) :: owning_blocks

    integer, parameter :: cap_init = 128
    logical, parameter :: exclude_existing = .true.

    integer :: ptr, cap, gcol, bid, cnt, ucnt, i, j, pe, prev, acap, aptr, ng, max_ng, &
         jprev, nupes, chunk_owner
    integer, allocatable :: nbrhd(:), apes(:), agcols(:)
    integer, allocatable, dimension(:) :: idxs, pes, upes, gcols, ugcols, gidxs
    logical :: e, same

    ! Collect (pe,gcol) pairs where gcol is in one of iam's owned blocks or
    ! chunks and pe is the chunk owner of a column having gcol in its nbrhd.
    cap = cap_init
    acap = cap_init
    allocate(nbrhd(cap), idxs(cap), pes(cap), upes(cap), apes(acap), agcols(acap))
    aptr = 1
    do gcol = 1, gd%ngcols
       if (owning_blocks) then
          call gcol2bid(gcol, bid)
          if (get_block_owner_d(bid) /= iam) cycle
       else
          chunk_owner = chunks(knuhcs(gcol)%chunkid)%owner
          if (chunk_owner /= iam) cycle
       end if
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
    if (cnt == 0) return
    ! Count the number of unique pes and the max number of unique gcols per pe.
    allocate(idxs(cnt))
    call IndexSet(cnt, idxs)
    call IndexSort(cnt, idxs, apes)
    ucnt = 0
    prev = -1
    max_ng = 1
    ng = 0
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
    allocate(rpe2nbrs%xs(nupes), rpe2nbrs%yptr(nupes+1), rpe2nbrs%ys(cap), &
         gidxs(max_ng), gcols(max_ng), ugcols(max_ng))
    rpe2nbrs%yptr(1) = 1
    i = 1
    do ucnt = 1, nupes
       rpe2nbrs%xs(ucnt) = apes(idxs(i))
       ng = 0
       do while (apes(idxs(i)) == rpe2nbrs%xs(ucnt))
          ng = ng + 1
          gcols(ng) = agcols(idxs(i))
          i = i + 1
          if (i > cnt) exit
       end do
       call IndexSet(ng, gidxs)
       call IndexSort(ng, gidxs, gcols)
       rpe2nbrs%yptr(ucnt+1) = rpe2nbrs%yptr(ucnt)
       jprev = -1
       ptr = 0
       do j = 1, ng
          if (gcols(gidxs(j)) == jprev) cycle
          jprev = gcols(gidxs(j))
          ptr = ptr + 1
          ugcols(ptr) = jprev
          rpe2nbrs%yptr(ucnt+1) = rpe2nbrs%yptr(ucnt+1) + 1
       end do
       if (rpe2nbrs%yptr(ucnt+1)-1 > cap) then
          cap = max(2*cap, rpe2nbrs%yptr(ucnt+1)-1)
          call array_realloc(rpe2nbrs%ys, rpe2nbrs%yptr(ucnt)-1, cap)
       end if
       e = assert(rpe2nbrs%yptr(ucnt+1) - rpe2nbrs%yptr(ucnt) == ptr, 'yptr and ptr')
       rpe2nbrs%ys(rpe2nbrs%yptr(ucnt):rpe2nbrs%yptr(ucnt+1)-1) = ugcols(1:ptr)
    end do
    cnt = rpe2nbrs%yptr(nupes+1)-1
    call array_realloc(rpe2nbrs%ys, cnt, cnt) ! compact memory
    deallocate(apes, agcols, idxs, gidxs, gcols, ugcols)
    if (cns%verbose > 1) then
       write(iulog,*) 'nbr> rpe2nbrs #pes',size(rpe2nbrs%xs)
       if (cns%verbose > 2) then
          do i = 1, size(rpe2nbrs%xs)
             write(iulog,*) 'nbr> pe',rpe2nbrs%xs(i),rpe2nbrs%yptr(i+1)-rpe2nbrs%yptr(i)
          end do
       end if
    end if
  end subroutine make_rpe2nbrs

  subroutine make_spe2nbrs(cns, gd, chunks, knuhcs, cnbrhds, spe2nbrs, owning_blocks)
    ! Make the map of owning pe ((s)ending (pe)s) to gcols, where the gcols are
    ! neighbors of a column in a block, each gcol belongs to a chunk that iam
    ! owns, and neighbor is defined in find_gcol_nbrhd. On output, spe2nbrs has
    ! these entries:
    !     xs: sorted list of owning pes;
    !     yptr: pointers into ys;
    !     ys: for each pe, the list of iam-owning sorted gcols.

    use dyn_grid, only: get_block_owner_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    type (SparseTriple), intent(in) :: cnbrhds
    type (SparseTriple), intent(out) :: spe2nbrs
    logical, intent(in) :: owning_blocks

    logical, parameter :: exclude_existing = .true.

    integer, allocatable, dimension(:) :: unbrs, wrk(:)
    integer, allocatable, dimension(:) :: pes, idxs
    integer :: i, j, k, n, gcol, bid, cnt, prev
    logical :: e

    e = assert(size(cnbrhds%yptr)-1 == gd%nlcols, 'nlcols')
    ! Unlike for rpe2nbrs, we are filling a 1-1 map, so we just need the unique
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
    if (cns%verbose > 1) &
         write(iulog,*) 'nbr> spe2nbrs', cnbrhds%yptr(gd%nlcols+1)-1, size(unbrs), n
    ! For each gcol, get the pe of the owning block or chunk.
    allocate(pes(n), idxs(n))
    do i = 1, n
       gcol = unbrs(i)
       if (owning_blocks) then
          call gcol2bid(gcol, bid)
          pes(i) = get_block_owner_d(bid)
       else
          pes(i) = chunks(knuhcs(gcol)%chunkid)%owner
       end if
       e = assert(pes(i) >= 0 .and. pes(i) <= npes-1, 'spe2nbrs pes')
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
    ! Fill spe2nbrs.
    allocate(spe2nbrs%xs(cnt), spe2nbrs%yptr(cnt+1), spe2nbrs%ys(n))
    j = 1
    i = 1
    spe2nbrs%yptr(j) = 1
    do j = 1, cnt
       spe2nbrs%xs(j) = pes(idxs(i))
       k = 0
       do while (pes(idxs(i)) == spe2nbrs%xs(j))
          spe2nbrs%ys(spe2nbrs%yptr(j)+k) = unbrs(idxs(i))
          k = k + 1
          i = i + 1
          if (i > n) exit
       end do
       spe2nbrs%yptr(j+1) = spe2nbrs%yptr(j) + k
    end do
    e = assert(spe2nbrs%yptr(cnt+1) == n+1, 'spe2nbrs%yptr post')
    ! Sort each set of gcols.
    do j = 1, cnt
       n = spe2nbrs%yptr(j+1) - spe2nbrs%yptr(j)
       call IndexSet(n, idxs(1:n))
       unbrs(1:n) = spe2nbrs%ys(spe2nbrs%yptr(j):spe2nbrs%yptr(j+1)-1)
       call IndexSort(n, idxs(1:n), unbrs(1:n))
       do i = 1, n
          spe2nbrs%ys(spe2nbrs%yptr(j)+i-1) = unbrs(idxs(i))
       end do
    end do
    deallocate(unbrs, idxs, pes)
    if (cns%verbose > 1) then
       write(iulog,*) 'nbr> spe2nbrs #pes',size(spe2nbrs%xs)
       if (cns%verbose > 2) then
          do i = 1, size(spe2nbrs%xs)
             write(iulog,*) 'nbr> pe',spe2nbrs%xs(i),spe2nbrs%yptr(i+1)-spe2nbrs%yptr(i)
          end do
       end if
    end if
  end subroutine make_spe2nbrs

  function find_gcol_nbrhd(gd, max_angle, gcol, nbrhd, ptr, cap) result(cnt)
    ! Find all columns having center within max_angle of gcol. Append entries
    ! to nbrhd(ptr:), reallocating as necessary. cap is nbrhd's capacity at
    ! input, and it is updated when reallocation occurs. The gcols list is
    ! sorted.

    type (PhysGridData), intent(in) :: gd
    real(r8), intent(in) :: max_angle
    integer, intent(in) :: gcol, ptr
    integer, allocatable, intent(inout) :: nbrhd(:)
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

  subroutine make_comm_schedule(cns, gd, chunks, knuhcs, rpe2nbrs, spe2nbrs, cs, &
       owning_blocks)
    use dyn_grid, only: get_block_gcol_cnt_d, get_block_owner_d, get_block_lvl_cnt_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    type (SparseTriple), intent(in) :: rpe2nbrs, spe2nbrs
    type (CommSchedule), intent(out) :: cs
    logical, intent(in) :: owning_blocks

    integer, allocatable :: sgcols(:), l2cids(:)
    integer :: i, j, k, lid, cid, gid, nlcl, gcol, blockid, pecnt, glbcnt, &
         pe, numlev, ptr, n, icol
    logical :: e

    if (owning_blocks) then
       nlcl = size(cns%ie2bid)
    else
       nlcl = size(cns%l2cid)
    end if
    allocate(cs%snd_num(0:npes-1), cs%rcv_num(0:npes-1), &
         cs%snd_offset(nlcl))

    ! Get column counts.
    do lid = 1, nlcl
       if (owning_blocks) then
          gid = cns%ie2bid(lid)
          n = get_block_gcol_cnt_d(gid)
       else
          cid = cns%l2cid(lid)
          n = chunks(cid)%ncols
       end if
       cs%snd_offset(lid)%ncol = n
       ! numrep is redundant wrt col, but it's useful
       allocate(cs%snd_offset(lid)%numlev(n), cs%snd_offset(lid)%numrep(n), &
            cs%snd_offset(lid)%col(n+1))
       cs%snd_offset(lid)%numlev(:) = 0
       cs%snd_offset(lid)%numrep(:) = 0
    end do
    ! Get repetition counts. A gcol in a block is in general in the
    ! neighborhoods of multiple chunks' gcols on multiple pes.
    do i = 1, size(rpe2nbrs%xs)
       do j = rpe2nbrs%yptr(i), rpe2nbrs%yptr(i+1)-1
          if (owning_blocks) then
             gcol = rpe2nbrs%ys(j) ! gcol in a chunk on pe
             call gcol2bid(gcol, blockid, icol)
             e = assert(get_block_owner_d(blockid) == iam, &
                        'b2c sched: gcol is owned')
             k = binary_search(nlcl, cns%bid2ie%id1, blockid)
             e = assert(k >= 1, 'sched: blockid is in map')
             lid = cns%bid2ie%id2(k) ! local block ID providing data to the chunk
             e = assert(icol >= 1 .and. icol <= cs%snd_offset(lid)%ncol, &
                        'b2c sched: icol is in range')
          else
             gcol = rpe2nbrs%ys(j)
             e = assert(chunks(knuhcs(gcol)%chunkid)%owner == iam, &
                  'c2c sched: gcol is owned')
             cid = knuhcs(gcol)%chunkid
             lid = binary_search(nlcl, cns%l2cid, cid)
             e = assert(lid >= 1, 'c2c sched: cid is in map')
             icol = knuhcs(gcol)%col
          end if
          cs%snd_offset(lid)%numrep(icol) = cs%snd_offset(lid)%numrep(icol) + 1
       end do
    end do
    ! Allocate offset arrays.
    do lid = 1, nlcl
       allocate(cs%snd_offset(lid)%os(sum(cs%snd_offset(lid)%numrep)))
       cs%snd_offset(lid)%col(1) = 1
       do i = 1, cs%snd_offset(lid)%ncol
          cs%snd_offset(lid)%col(i+1) = cs%snd_offset(lid)%col(i) + &
               cs%snd_offset(lid)%numrep(i)
       end do
       cs%snd_offset(lid)%numrep(:) = 0
    end do
    ! Get offsets and send counts.
    glbcnt = 0
    cs%snd_num(:) = 0
    cs%max_numrep = 0
    cs%max_numlev = 0
    do i = 1, size(rpe2nbrs%xs)
       pecnt = 0
       do j = rpe2nbrs%yptr(i), rpe2nbrs%yptr(i+1)-1
          gcol = rpe2nbrs%ys(j)
          if (owning_blocks) then
             call gcol2bid(gcol, blockid, icol)
             k = binary_search(nlcl, cns%bid2ie%id1, blockid)
             lid = cns%bid2ie%id2(k)
             numlev = get_block_lvl_cnt_d(gid, icol)
          else
             cid = knuhcs(gcol)%chunkid
             lid = binary_search(nlcl, cns%l2cid, cid)
             icol = knuhcs(gcol)%col
             numlev = pver + 1
          end if
          ptr = cs%snd_offset(lid)%col(icol)
          k = cs%snd_offset(lid)%numrep(icol)
          cs%snd_offset(lid)%os(ptr+k) = glbcnt
          cs%snd_offset(lid)%numrep(icol) = k + 1
          cs%max_numrep = max(cs%max_numrep, k + 1)
          cs%snd_offset(lid)%numlev(icol) = numlev
          cs%max_numlev = max(cs%max_numlev, numlev)
          glbcnt = glbcnt + numlev
          pecnt = pecnt + numlev
       end do
       cs%snd_num(rpe2nbrs%xs(i)) = pecnt
    end do
    cs%snd_nrecs = glbcnt

    ! Sort the received gcols so that the order is independent of comm
    ! schedule. Then we can use multiple comm schedules for the same icol
    ! ordering in the extra chunk.
    call sort(spe2nbrs%ys, sgcols)

    n = size(sgcols)
    allocate(cs%rcv_offset(n), cs%rcv_numlev(n))
    glbcnt = 0
    cs%rcv_num(:) = 0
    do i = 1, size(spe2nbrs%xs)
       pe = spe2nbrs%xs(i)
       pecnt = 0
       do j = spe2nbrs%yptr(i), spe2nbrs%yptr(i+1)-1
          gcol = spe2nbrs%ys(j)
          if (owning_blocks) then
             call gcol2bid(gcol, blockid, icol)
             e = assert(get_block_owner_d(blockid) == pe, &
                        'b2c sched: gcol pe association')
             numlev = get_block_lvl_cnt_d(blockid, icol)
          else
             e = assert(chunks(knuhcs(gcol)%chunkid)%owner == pe, &
                        'c2c sched: gcol pe association')
             icol = knuhcs(gcol)%col
             numlev = pver + 1
          end if
          k = binary_search(n, sgcols, gcol)
          cs%rcv_offset(k) = glbcnt
          cs%max_numlev = max(cs%max_numlev, numlev)
          cs%rcv_numlev(k) = numlev
          glbcnt = glbcnt + numlev
          pecnt = pecnt + numlev
       end do
       cs%rcv_num(pe) = pecnt
    end do
    cs%rcv_nrecs = glbcnt

    deallocate(sgcols)
  end subroutine make_comm_schedule

  subroutine init_comm_data(cns, cs, cd, phys_alltoall)
    use spmd_utils, only: pair, ceil2

    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommSchedule), intent(in) :: cs
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
          if (cs%snd_num(pe) > 0 .or. cs%rcv_num(pe) > 0) then
             cd%dp_coup_steps = cd%dp_coup_steps + 1
             if (j == 2) cd%dp_coup_proc(cd%dp_coup_steps) = pe
          end if
       end do
       if (j == 1) allocate(cd%dp_coup_proc(cd%dp_coup_steps))
    end do    

    if (cns%verbose > 1) write(iulog,*) 'nbr> dp_coup_steps', cd%dp_coup_steps
  end subroutine init_comm_data

  subroutine make_comm_data(cns, cs, cd, rcdsz)
#if defined SPMD
    use spmd_utils, only: mpicom
#endif

    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommSchedule), intent(in) :: cs
    type (CommData), intent(inout) :: cd
    integer, intent(in) :: rcdsz

    integer :: pe
    logical :: e
    
    e = assert(rcdsz >= 1, 'b2cd: valid record size')
    if (rcdsz == cd%prev_record_size) return
    cd%prev_record_size = rcdsz
    
    cd%sdispls(0) = 0
    cd%sndcnts(0) = rcdsz*cs%snd_num(0)
    do pe = 1, npes-1
       cd%sdispls(pe) = cd%sdispls(pe-1) + cd%sndcnts(pe-1)
       cd%sndcnts(pe) = rcdsz*cs%snd_num(pe)
    enddo

    cd%rdispls(0) = 0
    cd%rcvcnts(0) = rcdsz*cs%rcv_num(0)
    do pe = 1, npes-1
       cd%rdispls(pe) = cd%rdispls(pe-1) + cd%rcvcnts(pe-1)
       cd%rcvcnts(pe) = rcdsz*cs%rcv_num(pe)
    enddo

#if defined SPMD
    call mpialltoallint(cd%rdispls, 1, cd%pdispls, 1, mpicom)
#endif
  end subroutine make_comm_data

  subroutine make_ie2bid(ie2bid, bid2ie)
    ! id2gid is a list of iam's owned global block IDs in block local ID
    ! order. bid2ie%id1 is the sorted list of global block IDs, and bid2ie%id2
    ! is the list of corresponding local IDs.

    use dyn_grid, only: get_block_bounds_d, get_block_gcol_d, get_block_owner_d, &
         get_block_gcol_cnt_d

    integer, allocatable, intent(out) :: ie2bid(:)
    type (IdMap), intent(out) :: bid2ie

    integer, allocatable :: gcols(:)
    integer :: bf, bl, bid, cnt, pe, nid, blockid, bcid, ie, ngcols, i
    logical :: e

    call get_block_bounds_d(bf, bl)
    cnt = 0
    do bid = bf, bl
       pe = get_block_owner_d(bid)
       if (pe == iam) cnt = cnt + 1
    end do
    allocate(ie2bid(cnt), gcols(128))
    ! This seems a bit convoluted, but I'm not seeing an easier way to get block
    ! IDs in local ID order.
    do bid = bf, bl
       pe = get_block_owner_d(bid)
       if (pe /= iam) cycle
       ngcols = get_block_gcol_cnt_d(bid)
       if (ngcols > size(gcols)) then
          deallocate(gcols)
          allocate(gcols(ngcols))
       end if
       call get_block_gcol_d(bid, ngcols, gcols)
       call gcol2bid(gcols(1), blockid, bcid, ie)
       e = assert(bid == blockid, 'bid == blockid')
       e = assert(ie >= 1 .and. ie <= cnt, 'ie in bounds')
       ie2bid(ie) = bid
    end do
    deallocate(gcols)
    ! Now the opposite direction.
    allocate(bid2ie%id1(cnt), bid2ie%id2(cnt))
    call IndexSet(cnt, bid2ie%id2)
    bid2ie%id1(:) = ie2bid(:)
    call IndexSort(cnt, bid2ie%id2, bid2ie%id1)
    do i = 1, cnt
       bid2ie%id1(i) = ie2bid(bid2ie%id2(i))
    end do
  end subroutine make_ie2bid
  
  subroutine make_l2cid(nchunks, chunks, l2cid)
    integer, intent(in) :: nchunks
    type (chunk), intent(in) :: chunks(:)
    integer, allocatable, intent(out) :: l2cid(:)

    integer :: nlcl, cid, lid

    nlcl = 0
    do cid = 1, nchunks
       if (chunks(cid)%owner /= iam) cycle
       nlcl = nlcl + 1
    end do
    allocate(l2cid(nlcl))
    lid = 0
    do cid = 1, nchunks
       if (chunks(cid)%owner /= iam) cycle
       lid = lid + 1
       l2cid(lid) = cid
    end do
  end subroutine make_l2cid

  subroutine init_chunk(cns, gd, gcols, chk)
    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    integer, intent(in) :: gcols(:)
    type (chunk), intent(out) :: chk

    integer, allocatable :: idxs(:)
    integer :: i

    ! Sort the received gcols so we have an order independent of comm schedule.
    chk%ncols = cns%extra_chunk_ncol
    allocate(idxs(chk%ncols))
    call IndexSet(chk%ncols, idxs)
    call IndexSort(chk%ncols, idxs, gcols)
    ! Fill the chunk.
    chk%dcols = chk%ncols
    chk%owner = iam
    chk%lcid = endchunk + nbrhdchunk
    chk%estcost = -1
    allocate(chk%gcol(chk%ncols), chk%lat(chk%ncols), chk%lon(chk%ncols))
    do i = 1, chk%ncols
       chk%gcol(i) = gcols(idxs(i))
       chk%lat(i) = gd%lat_p(chk%gcol(i))
       chk%lon(i) = gd%lon_p(chk%gcol(i))
    end do
    deallocate(idxs)
  end subroutine init_chunk

  subroutine make_cc(cns, lchks, cc)
    type (ColumnNeighborhoods), intent(in) :: cns
    type (lchunk), intent(in) :: lchks(begchunk:endchunk)
    type (ColumnCopies), intent(out) :: cc

    integer, allocatable, dimension(:) :: ugcols, lcids, icols
    integer :: nlcols, nugcols, lcid, icol, gcol, cnt, cap, i, k
    logical :: e

    ! Unique neighborhood gcols.
    nlcols = size(cns%chk_nbrhds%xs)
    call make_unique(cns%chk_nbrhds%yptr(nlcols+1)-1, cns%chk_nbrhds%ys, ugcols)
    nugcols = size(ugcols)

    cap = 128
    allocate(lcids(cap), icols(cap))
    cnt = 0
    do lcid = begchunk, endchunk
       do icol = 1, lchks(lcid)%ncols
          gcol = lchks(lcid)%gcol(icol)
          k = binary_search(size(ugcols), ugcols, gcol)
          if (k == -1) cycle ! not in a nbrhd
          cnt = cnt + 1
          if (cnt > cap) then
             call array_realloc(lcids, cap, 2*cap)
             call array_realloc(icols, cap, 2*cap)
             cap = 2*cap
          end if
          lcids(cnt) = lcid
          icols(cnt) = icol
       end do
    end do
    deallocate(ugcols)

    cc%num_recv_col = cns%extra_chunk_ncol
    allocate(cc%cols(cnt))
    do i = 1, cnt
       cc%cols(i)%lcid = lcids(i)
       cc%cols(i)%icol = icols(i)
    end do

    deallocate(lcids, icols)
  end subroutine make_cc

  subroutine make_c2n(cns, lchks, c2n)
    type (ColumnNeighborhoods), intent(in) :: cns
    type (lchunk), intent(in) :: lchks(begchunk:endchunk+nbrhdchunk)
    type (ChunkDesc), allocatable, intent(out) :: c2n(:)

    integer, allocatable, dimension(:) :: idxs, sgcols
    integer :: lcid, i, j, k, icol, gcol, cnt, lcolid, n, extra, ncol
    logical :: e

    extra = endchunk+nbrhdchunk

    ! Sorted list of extra gcols for search in the next step.
    n = lchks(extra)%ncols
    allocate(idxs(n), sgcols(n))
    call IndexSet(n, idxs)
    call IndexSort(n, idxs, lchks(extra)%gcol)
    do i = 1, n
       sgcols(i) = lchks(extra)%gcol(idxs(i))
       if (i > 1) e = assert(sgcols(i) > sgcols(i-1), 'c2n: gcols are unique')
    end do

    allocate(c2n(begchunk:endchunk))
    do lcid = begchunk, endchunk
       ncol = lchks(lcid)%ncols
       cnt = 0
       do icol = 1, ncol
          gcol = lchks(lcid)%gcol(icol)
          lcolid = SparseTriple_in_xs(cns%chk_nbrhds, gcol)
          e = assert(lcolid > 0, 'c2n: gcol present')
          cnt = cnt + cns%chk_nbrhds%yptr(lcolid+1) - cns%chk_nbrhds%yptr(lcolid)
       end do
       allocate(c2n(lcid)%col(ncol+1), c2n(lcid)%nbrhd(cnt))
       c2n(lcid)%nbrhd(:) = -1
       c2n(lcid)%col(1) = 1
       i = 1
       do icol = 1, ncol
          gcol = lchks(lcid)%gcol(icol)
          lcolid = SparseTriple_in_xs(cns%chk_nbrhds, gcol)
          c2n(lcid)%col(icol+1) = c2n(lcid)%col(icol) + &
               (cns%chk_nbrhds%yptr(lcolid+1) - cns%chk_nbrhds%yptr(lcolid))
          do j = cns%chk_nbrhds%yptr(lcolid), cns%chk_nbrhds%yptr(lcolid+1)-1
             gcol = cns%chk_nbrhds%ys(j)
             k = binary_search(size(sgcols), sgcols, gcol)
             e = assert(k > 0, 'c2n: nbr gcol')
             c2n(lcid)%nbrhd(i) = idxs(k)
             e = assert(lchks(extra)%gcol(c2n(lcid)%nbrhd(i)) == gcol, &
                        'c2n: gcol association')
             i = i + 1
          end do
       end do
       e = assert(all(c2n(lcid)%nbrhd >= 1), 'c2n: nbrhd filled')
    end do
    deallocate(idxs, sgcols)
  end subroutine make_c2n

  subroutine gcol2bid(gcol, block_id, bcid, ie)
    ! Map gcol_d to global block ID and optionally the column within the block
    ! and the block's local ID. The local ID is expensive to compute.

    use dyn_grid, only: get_gcol_block_cnt_d, get_gcol_block_d

    integer, intent(in) :: gcol
    integer, intent(out) :: block_id
    integer, optional, intent(out) :: bcid, ie

    integer :: block_cnt, blockids(1), bcids(1), ies(1)
    logical :: e

    block_cnt = get_gcol_block_cnt_d(gcol)
    e = assert(block_cnt == 1, 'only block_cnt=1 is supported')
    if (present(ie)) then
       call get_gcol_block_d(gcol, block_cnt, blockids, bcids, ies)
       ie = ies(1)
    else
       call get_gcol_block_d(gcol, block_cnt, blockids, bcids)
    end if
    block_id = blockids(1)
    if (present(bcid)) bcid = bcids(1)
  end subroutine gcol2bid

  subroutine comm_sizes(cns, cs, block_buf_nrecs, chunk_buf_nrecs, &
       max_numlev, max_numrep, num_recv_col)
    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommSchedule), intent(in) :: cs
    integer, intent(out) :: block_buf_nrecs, chunk_buf_nrecs, max_numlev, &
         max_numrep, num_recv_col

    block_buf_nrecs = cs%snd_nrecs
    chunk_buf_nrecs = cs%rcv_nrecs
    max_numlev = cs%max_numlev
    max_numrep = cs%max_numrep
    num_recv_col = cns%cc%num_recv_col
  end subroutine comm_sizes

  subroutine comm_send_pters(cs, lid, icol, rcdsz, numlev, numrep, ptr)
    type (CommSchedule), intent(in) :: cs
    integer, intent(in) :: lid, icol, rcdsz
    integer, intent(out) :: numlev, numrep
    integer, intent(out) :: ptr(:,:) ! >= max_numlev x >= max_numrep

    integer :: i, j, k
    logical :: e

    e = assert(lid >= 1 .and. lid <= size(cs%snd_offset), 'send_pters: lid')
    e = assert(icol >= 1 .and. icol <= cs%snd_offset(lid)%ncol, 'send_pters: icol')
    numlev = cs%snd_offset(lid)%numlev(icol)
    numrep = cs%snd_offset(lid)%numrep(icol)
    ptr(:,:) = -1
    j = cs%snd_offset(lid)%col(icol)
    do i = 1, numrep
       ptr(1,i) = rcdsz*cs%snd_offset(lid)%os(j+i-1) + 1
       do k = 2, numlev
          ptr(k,i) = ptr(1,i) + rcdsz*(k-1)
       end do
    end do
  end subroutine comm_send_pters

  subroutine comm_transpose(cns, cs, cd, rcdsz, snd_buf, rcv_buf)
#if defined SPMD
    use spmd_utils, only: mpicom, altalltoallv
    use mpishorthand, only: mpir8
#endif

    type (ColumnNeighborhoods), intent(in) :: cns
    type (CommSchedule), intent(in) :: cs
    type (CommData), intent(inout) :: cd
    integer, intent(in) :: rcdsz
    real(r8), intent(in) :: snd_buf(rcdsz*cs%snd_nrecs)
    real(r8), intent(out) :: rcv_buf(rcdsz*cs%rcv_nrecs)

#if defined SPMD
    integer, parameter :: msgtag = 6042

    integer :: ssz, rsz, lwindow

    call make_comm_data(cns, cs, cd, rcdsz)
    ssz = rcdsz*cs%snd_nrecs
    rsz = rcdsz*cs%rcv_nrecs
    lwindow = -1
    call altalltoallv(cd%lopt, iam, npes, &
                      cd%dp_coup_steps, cd%dp_coup_proc, &
                      snd_buf, ssz, cd%sndcnts, cd%sdispls, mpir8, &
                      rcv_buf, rsz, cd%rcvcnts, cd%rdispls, mpir8, &
                      msgtag, cd%pdispls, mpir8, lwindow, mpicom)
#endif
  end subroutine comm_transpose

  subroutine comm_recv_pters(cs, icol, rcdsz, numlev, ptr)
    type (CommSchedule), intent(in) :: cs
    integer, intent(in) :: icol, rcdsz
    integer, intent(out) :: numlev
    integer, intent(out) :: ptr(:)

    integer :: k
    logical :: e
    
    e = assert(icol >= 1 .and. icol <= cns%extra_chunk_ncol, 'recv_pters: icol')
    numlev = cs%rcv_numlev(icol)
    ptr(1) = rcdsz*cs%rcv_offset(icol) + 1
    do k = 2, numlev
       ptr(k) = ptr(1) + rcdsz*(k-1)
    end do
  end subroutine comm_recv_pters

  !> -------------------------------------------------------------------
  !> General utilities.

  function test(nerr, cond, message) result(out)
    ! Assertion that is always enabled, for use in unit tests.
    
    integer, intent(inout) :: nerr
    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    if (.not. cond) then
       write(iulog,*) 'nbr> test ', trim(message)
       nerr = nerr + 1
    end if
    out = cond
  end function test

  function assert(cond, message) result(out)
    ! Assertion that can be disabled.
    use cam_abortutils, only: endrun

    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    if (.not. cond) then
       write(iulog,*) 'nbr> assert ', trim(message)
       call endrun('nbr> assert')
    end if
    out = cond
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
    e = assert((k == 1 .or. a(max(1,k-1)) <= val) .and. &
               (k == n .or. val < a(min(n,k))), &
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

    integer, allocatable, intent(inout) :: a(:)
    integer, intent(in) :: n, n_new

    integer, allocatable :: buf(:)
    integer :: i, n_min
    logical :: e

    e = assert(size(a) >= n, 'array_realloc size(a)')
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
    integer, allocatable, intent(out) :: ua(:)

    integer, allocatable :: idxs(:)
    integer :: cnt, prev, i
    logical :: e

    if (n == 0) then
       allocate(ua(0))
       return
    end if
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

  subroutine sort(a, b)
    ! Simple sort wrapper for when idxs is not needed.

    integer, intent(in) :: a(:)
    integer, allocatable, intent(out) :: b(:)

    integer, allocatable :: idxs(:)
    integer :: n, i
    logical :: e

    n = size(a)
    allocate(idxs(n), b(n))
    call IndexSet(n, idxs)
    call IndexSort(n, idxs, a)
    do i = 1, n
       b(i) = a(idxs(i))
    end do
    deallocate(idxs)
  end subroutine sort

  subroutine SparseTriple_deallocate(st)
    type (SparseTriple), intent(out) :: st
    if (allocated(st%xs)) deallocate(st%xs, st%yptr, st%ys)
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

  !> -------------------------------------------------------------------
  !> Internal tests.

  subroutine run_unit_tests()
    ! Unit tests for helper routines.

    use shr_const_mod, only: pi => shr_const_pi

    integer, parameter :: n = 4, b(n) = (/ -2, -1, 3, 7 /), &
         m = 6, c(m) = (/ 1, 3, 2, -1, 2, 3 /)
    real(r8), parameter :: a(n) = (/ -1.0_r8, 1.0_r8, 1.5_r8, 3.0_r8 /), &
         tol = epsilon(1.0_r8)

    integer, allocatable :: uc(:)
    real(r8) :: lat1, lon1, lat2, lon2, x1, y1, z1, x2, y2, z2, angle
    integer :: k, nerr
    logical :: e

    nerr = 0

    k = upper_bound_or_in_range(n, a, -1.0_r8); e = test(nerr, k == 2, 'uboir 1')
    k = upper_bound_or_in_range(n, a, -1.0_r8, 2); e = test(nerr, k == 2, 'uboir 2')
    k = upper_bound_or_in_range(n, a, 3.0_r8, 2); e = test(nerr, k == n, 'uboir 3')
    k = upper_bound_or_in_range(n, a, 3.0_r8, -11); e = test(nerr, k == n, 'uboir 4')
    k = upper_bound_or_in_range(n, a, 1.2_r8); e = test(nerr, k == 3, 'uboir 5')

    k = binary_search(n, b, -22, -5); e = test(nerr, k == -1, 'binsrc 1')
    k = binary_search(n, b, -2); e = test(nerr, k == 1, 'binsrc 2')
    k = binary_search(n, b, 3, 2); e = test(nerr, k == 3, 'binsrc 3')
    k = binary_search(n, b, 7); e = test(nerr, k == 4, 'binsrc 4')
    k = binary_search(n, b, 7, 4); e = test(nerr, k == 4, 'binsrc 5')
    k = binary_search(n, b, 7, 5); e = test(nerr, k == 4, 'binsrc 6')
    k = binary_search(n, b, 0, 15); e = test(nerr, k == -1, 'binsrc 7')

    lat1 = -pi/3
    lon1 = pi/2
    call latlon2xyz(lat1, lon1, x1, y1, z1)
    lat2 = lat1 - 0.1_r8
    lon2 = lon1
    angle = unit_sphere_angle(x1, y1, z1, lat2, lon2)
    e = test(nerr, reldif(0.1_r8,angle) <= 10*tol, 'usa 1')

    call make_unique(m, c, uc)
    e = test(nerr, size(uc) == 4, 'uc length')
    e = test(nerr, uc(1) == -1, 'first(uc)')
    e = test(nerr, uc(4) == 3, 'last(uc)')
    do k = 2, size(uc)
       e = test(nerr, uc(k) > uc(k-1), 'uc sorted')
    end do
    deallocate(uc)

    k = 3
    call incr(k)
    e = test(nerr, k == 4, 'incr')

    if (nerr > 0) write(iulog,*) 'nbr> run_unit_tests FAIL', nerr
  end subroutine run_unit_tests

  subroutine test_nbrhds(cns, gd)
    ! A chunk-owned column may not be in the neighborhood of any other
    ! chunk-owned columns. Use this fact to brute-force check the neighborhood
    ! lists.

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd

    integer, allocatable, dimension(:) :: ugcols
    real(r8) :: x, y, z, angle, min_angle
    integer :: nlcols, nugcols, i, j, k, gcol, jgcol
    logical :: e

    nlcols = size(cns%chk_nbrhds%xs)
    call make_unique(cns%chk_nbrhds%yptr(nlcols+1)-1, cns%chk_nbrhds%ys, ugcols)
    nugcols = size(ugcols)
    do i = 1, nlcols
       gcol = cns%chk_nbrhds%xs(i)
       k = binary_search(size(ugcols), ugcols, gcol)
       if (k >= 1) cycle
       ! This chunk-owned column is not in a neighborhood of any other
       ! chunk-owned column. Check through brute force that this is correct.
       call latlon2xyz(gd%clat_p(gd%lat_p(gcol)), gd%clon_p(gd%lon_p(gcol)), x, y, z)
       min_angle = 100*cns%max_angle
       do j = 1, nlcols
          if (j == i) cycle
          jgcol = cns%chk_nbrhds%xs(j)
          angle = unit_sphere_angle(x, y, z, &
               gd%clat_p(gd%lat_p(jgcol)), gd%clon_p(gd%lon_p(jgcol)))
          min_angle = min(min_angle, angle)
       end do
       e = assert(min_angle > cns%max_angle, 'test_nbrhds: angle')
    end do
  end subroutine test_nbrhds

  subroutine histogram_nbrhds(n)
#if defined SPMD
    use spmd_utils, only: mpicom
    use mpishorthand, only: mpiint
#endif

    type (SparseTriple), intent(in) :: n

    integer, allocatable :: lhist(:), ghist(:)
    integer :: lmaxsz, gmaxsz, i, idx, lmax(1), gmax(1)

    lmaxsz = 0
    do i = 1, size(n%xs)
       lmaxsz = max(lmaxsz, n%yptr(i+1) - n%yptr(i))
    end do
    allocate(lhist(lmaxsz+1))
    lhist(:) = 0
    do i = 1, size(n%xs)
       idx = n%yptr(i+1) - n%yptr(i) + 1
       lhist(idx) = lhist(idx) + 1
    end do
#if defined SPMD
    lmax(1) = lmaxsz
    call mpiallmaxint(lmax, gmax, 1, mpicom)
    gmaxsz = gmax(1)
    call array_realloc(lhist, lmaxsz+1, gmaxsz+1)
    lhist(lmaxsz+2:) = 0
    allocate(ghist(gmaxsz+1))
    call mpisum(lhist, ghist, gmaxsz+1, mpiint, 0, mpicom)
#else
    gmaxsz = lmaxsz
    allocate(ghist(gmaxsz+1))
    ghist(:) = lhist(:)
#endif
    if (masterproc) then
       do i = 1, gmaxsz+1
          if (ghist(i) > 0) write(iulog,'(a,i5,i10)') 'nbr> hist', i-1, ghist(i)
       end do
    end if
    deallocate(lhist, ghist)
  end subroutine histogram_nbrhds

  subroutine test_comm_schedule(cns, gd, chunks, knuhcs, rpe2nbrs, spe2nbrs, cd, cs, &
       owning_blocks)
    use dyn_grid, only: get_block_gcol_cnt_d, get_block_gcol_d, get_horiz_grid_d

    type (ColumnNeighborhoods), intent(in) :: cns
    type (PhysGridData), intent(in) :: gd
    type (chunk), intent(in) :: chunks(:)
    type (knuhc), intent(in) :: knuhcs(:)
    type (SparseTriple), intent(in) :: rpe2nbrs, spe2nbrs
    type (CommSchedule), intent(in) :: cs
    type (CommData), intent(in) :: cd
    logical, intent(in) :: owning_blocks

    real(r8), parameter :: none = -10000
    integer, parameter :: rcdsz = 2

    real(r8), allocatable, dimension(:) :: lats, lons, sbuf, rbuf, lats_d, lons_d
    real(r8) :: lat, lon, x, y, z, angle
    integer, allocatable :: sptr(:,:), rptr(:), sgcols(:)
    integer :: cid, ncols, gcol, i, j, k, lid, bnrecs, cnrecs, icol, max_numlev, &
         max_numrep, num_recv_col, numlev, numrep, bid, ngcols, gcols(16), nerr, &
         jgcol, lcide
    logical :: e

    if (masterproc) write(iulog,*) 'nbr> test_comm_schedule', owning_blocks
    nerr = 0

    allocate(lats(gd%ngcols), lons(gd%ngcols))
    lats(:) = none; lons(:) = none

    ! Fill lats, lons with iam-owning chunks' gcols' data. These are available
    ! from standard phys_grid comm.
    do cid = 1, cns%nchunks
       if (chunks(cid)%owner /= iam) cycle
       ncols = chunks(cid)%ncols
       do i = 1, ncols
          gcol = chunks(cid)%gcol(i)
          lats(gcol) = gd%clat_p(gd%lat_p(gcol))
          lons(gcol) = gd%clon_p(gd%lon_p(gcol))
       end do
    end do

    if (owning_blocks) then
       call nbrhd_block_to_chunk_sizes(bnrecs, cnrecs, max_numlev, max_numrep, num_recv_col)
    else
       call nbrhd_chunk_to_chunk_sizes(bnrecs, cnrecs, max_numlev, max_numrep, num_recv_col)
    end if
    allocate(sbuf(rcdsz*bnrecs), rbuf(rcdsz*cnrecs))
    sbuf(:) = none; rbuf(:) = none

    ! Pack send buffer.
    allocate(sptr(max_numlev,max_numrep))
    do lid = 1, size(cs%snd_offset) ! caller knows this
       if (owning_blocks) then
          bid = cns%ie2bid(lid)
          ngcols = get_block_gcol_cnt_d(bid)
          e = test(nerr, size(gcols) >= ngcols, 'comm: ngcols size')
          call get_block_gcol_d(bid, ngcols, gcols)
       else
          cid = cns%l2cid(lid)
          ngcols = chunks(cid)%ncols
       end if
       e = test(nerr, ngcols == cs%snd_offset(lid)%ncol, 'comm: ngcols = ncol')
       e = test(nerr, cs%snd_offset(lid)%ncol == size(cs%snd_offset(lid)%numlev), &
                'comm: ncol')
       do icol = 1, cs%snd_offset(lid)%ncol ! ditto
          if (owning_blocks) then
             gcol = gcols(icol)
          else
             gcol = chunks(cid)%gcol(icol)
          end if
          lat = gd%clat_p(gd%lat_p(gcol))
          lon = gd%clon_p(gd%lon_p(gcol))
          if (owning_blocks) then
             call nbrhd_block_to_chunk_send_pters(lid, icol, rcdsz, numlev, numrep, sptr)
          else
             call nbrhd_chunk_to_chunk_send_pters(begchunk+lid-1, icol, rcdsz, numlev, &
                  numrep, sptr)
          end if
          e = test(nerr, numlev <= max_numlev .and. numrep <= max_numrep, &
                   'comm: sptr size')
          e = test(nerr, all(sptr(:numlev,:numrep) >= 1), 'comm: sptr >= 1')
          do j = 1, numrep
             do k = 1, numlev
                e = test(nerr, sptr(k,j) >= 1 .and. sptr(k,j) <= size(sbuf), &
                         'comm: sptr sbuf')
                sbuf(sptr(k,j)+0) = lat + (k-1)
                sbuf(sptr(k,j)+1) = lon + (k-1)
             end do
          end do
       end do
    end do
    deallocate(sptr)

    e = test(nerr, .not. any(sbuf == none), 'comm: sbuf has no none values')
    if (owning_blocks) then
       call nbrhd_transpose_block_to_chunk(rcdsz, sbuf, rbuf)
    else
       call nbrhd_transpose_chunk_to_chunk(rcdsz, sbuf, rbuf)
    end if
    e = test(nerr, .not. any(rbuf == none), 'comm: rbuf has no none values')

    ! Unpack recv buffer.
    call sort(spe2nbrs%ys, sgcols)
    allocate(rptr(max_numlev))
    lcide = endchunk+nbrhdchunk
    do icol = 1, size(spe2nbrs%ys) ! from nbrhd_block_to_chunk_sizes
       if (owning_blocks) then
          call nbrhd_block_to_chunk_recv_pters(icol, rcdsz, numlev, rptr)
       else
          call nbrhd_chunk_to_chunk_recv_pters(icol, rcdsz, numlev, rptr)
       end if
       e = test(nerr, icol <= size(spe2nbrs%ys), 'comm: icol range')
       gcol = sgcols(icol) ! also from chunk query
       k = 1
       ! We should never unpack into a slot having other than the none value.
       e = test(nerr, lats(gcol) == none, 'comm: lats(gcol) is none')
       lats(gcol) = rbuf(rptr(k)+0)
       lons(gcol) = rbuf(rptr(k)+1)
       do k = 2, numlev
          e = test(nerr, rbuf(rptr(k)+0) == lats(gcol) + (k-1) .and. &
                         rbuf(rptr(k)+1) == lons(gcol) + (k-1), 'comm: lat,lon')
       end do
    end do
    deallocate(rptr, sgcols)

    deallocate(sbuf, rbuf)

    ! For each gcol in iam's chunks, check that
    ! * its nbrhd has all non-none values;
    ! * the values are correct;
    ! * the angular distance is <= max_angle.
    allocate(lats_d(gd%ngcols), lons_d(gd%ngcols))
    call get_horiz_grid_d(gd%ngcols, clat_d_out=lats_d, clon_d_out=lons_d)
    do cid = 1, cns%nchunks
       if (chunks(cid)%owner /= iam) cycle
       ncols = chunks(cid)%ncols
       do i = 1, ncols
          gcol = chunks(cid)%gcol(i)
          k = SparseTriple_in_xs(cns%chk_nbrhds, gcol)
          e = test(nerr, k >= 1, 'comm: gcol has a nbrhd')
          call latlon2xyz(lats(gcol), lons(gcol), x, y, z)
          do j = cns%chk_nbrhds%yptr(k), cns%chk_nbrhds%yptr(k+1)-1
             jgcol = cns%chk_nbrhds%ys(j)
             e = test(nerr, lats(jgcol) /= none, 'comm: lats(jgcol) has a value')
             e = test(nerr, lats(jgcol) == lats_d(jgcol), 'comm: lat')
             e = test(nerr, lons(jgcol) == lons_d(jgcol), 'comm: lon')
             angle = unit_sphere_angle(x, y, z, lats(jgcol), lons(jgcol))
             e = test(nerr, angle <= cns%max_angle, 'comm: angle')
          end do
          if (.not. e) exit
       end do
       if (.not. e) exit
    end do
    deallocate(lats_d, lons_d)

    deallocate(lats, lons)

    if (nerr > 0) write(iulog,*) 'nbr> test_b2c_comm_schedule FAIL', nerr
  end subroutine test_comm_schedule

  subroutine test_c2n(cns, lchks)
    type (ColumnNeighborhoods), intent(in) :: cns
    type (lchunk), intent(in) :: lchks(begchunk:endchunk+nbrhdchunk)

    integer, allocatable, dimension(:) :: icols
    integer :: nerr, lcid, icol, n, i, k, j1, j2, lcolid, gcol, extra
    logical :: e

    extra = endchunk+nbrhdchunk
    allocate(icols(128))
    nerr = 0
    do lcid = begchunk, endchunk
       do icol = 1, lchks(lcid)%ncols
          lcolid = SparseTriple_in_xs(cns%chk_nbrhds, lchks(lcid)%gcol(icol))
          n = nbrhd_get_nbrhd_size(lcid, icol)
          j1 = cns%chk_nbrhds%yptr(lcolid)
          j2 = cns%chk_nbrhds%yptr(lcolid+1)
          e = test(nerr, n == j2 - j1, 'api: n')
          if (n > size(icols)) then
             deallocate(icols)
             allocate(icols(2*n))
          end if
          call nbrhd_get_nbrhd(lcid, icol, icols)
          do i = 1, n
             gcol = lchks(extra)%gcol(icols(i))
             k = binary_search(n, cns%chk_nbrhds%ys(j1:j2-1), gcol)
             e = test(nerr, k /= -1, 'api: gcol found')
             e = assert(e, 'going down?')
          end do
       end do
    end do
    deallocate(icols)
    if (nerr > 0) write(iulog,*) 'nbr> test_c2n FAIL', nerr
  end subroutine test_c2n

end module phys_grid_nbrhd
