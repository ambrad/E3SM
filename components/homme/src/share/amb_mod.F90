module amb_mod
  use metagraph_mod, only: MetaVertex_t
  use parallel_mod, only: parallel_t, abortmp
  use dimensions_mod, only: npart
  use gridgraph_mod, only: GridVertex_t, GridEdge_t

  implicit none

  type, public :: GridManager_t
     integer :: rank
     integer, allocatable :: sfcfacemesh(:,:), sfc2rank(:), gvid(:)
     type (GridVertex_t), pointer :: gv(:) => null()
     type (GridEdge_t), pointer :: ge(:) => null()
  end type GridManager_t

contains

  subroutine amb_run(par)
    use dimensions_mod, only: nelem, ne

    type (parallel_t), intent(in) :: par
    integer, allocatable :: sfctest(:)
    type (GridManager_t) :: gm
    integer :: ie, i, j, face, id, sfc, nelemd, nelemdi
    logical, parameter :: dbg = .true.

    allocate(gm%sfc2rank(npart+1))
    call amb_genspacepart(nelem, npart, gm%sfc2rank)
    allocate(gm%sfcfacemesh(ne,ne))
    call amb_genspacecurve(ne, gm%sfcfacemesh)

    if (dbg .and. par%masterproc) then
       ! amb_genspacepart
       if (gm%sfc2rank(npart+1) /= nelem) then
          print *, 'AMB> nelem',nelem,'sfc2rank',gm%sfc2rank
       end if
       nelemd = gm%sfc2rank(2) - gm%sfc2rank(1)
       do i = 3, npart+1
          nelemdi = gm%sfc2rank(i) - gm%sfc2rank(i-1)
          if (nelemdi > nelemd) then
             print *, 'AMB> nelem',nelem,'sfc2rank',gm%sfc2rank
             exit
          end if
          nelemd = nelemdi
       end do
       ! u<->s and ->sfc
       allocate(sfctest(nelem))
       sfctest = 0
       do ie = 1, nelem
          call u2si(ie,i,j,face)
          id = s2ui(i,j,face)
          if (.not. (id == ie .and. &
               i >= 1 .and. i <= ne .and. &
               j >= 1 .and. j <= ne .and. &
               face >=1 .and. face <= 6)) then
             print *, 'AMB> u<->s:',ie,id,i,j,face
          end if
          sfc = u2sfc(gm%sfcfacemesh, id)
          if (.not. (sfc >= 0 .and. sfc < nelem)) then
             print *, 'AMB> u2sfc:',id,sfc
          end if
          sfctest(sfc+1) = sfctest(sfc+1) + 1
       end do
       do ie = 1, nelem
          if (sfctest(ie) .ne. 1) then
             print *, 'AMB> sfctest:',ie,sfctest(ie)
          end if
       end do
       deallocate(sfctest)
    end if

    gm%rank = par%rank
    call amb_cube_gv_pass1(gm)
    call amb_cube_gv_pass2(gm)
    deallocate(gm%sfcfacemesh)
    call amb_ge(gm)
    deallocate(gm%sfc2rank)
  end subroutine amb_run

  function s2ui(i, j, face) result (id)
    use dimensions_mod, only: ne

    integer, intent(in) :: i, j, face
    integer :: id

    id = ((face-1)*ne + (j-1))*ne + i
  end function s2ui

  subroutine u2si(id, i, j, face)
    use dimensions_mod, only: ne

    integer, intent(in) :: id
    integer, intent(out) :: i, j, face
    integer :: nesq

    nesq = ne*ne
    face = (id-1)/nesq + 1
    j = mod(id-1, nesq)/ne + 1
    i = mod(id-1, ne) + 1
  end subroutine u2si

  function s2sfc(sfcfacemesh, i, j, face) result(sfc)
    integer, intent(in) :: sfcfacemesh(:,:)
    integer, intent(in) :: i, j, face
    integer :: sfc, offset, ne, nesq

    ne = size(sfcfacemesh,1)
    nesq = ne*ne
    select case (face)
    case (1); sfc =          sfcfacemesh(i     , ne-j+1)
    case (2); sfc =   nesq + sfcfacemesh(i     , ne-j+1)
    case (3); sfc = 5*nesq + sfcfacemesh(i     , j     )
    case (4); sfc = 3*nesq + sfcfacemesh(ne-j+1, i     )
    case (5); sfc = 4*nesq + sfcfacemesh(i     , j     )
    case (6); sfc = 2*nesq + sfcfacemesh(ne-i+1, ne-j+1)
    end select
  end function s2sfc

  function u2sfc(sfcfacemesh, id) result(sfc)
    integer, intent(in) :: sfcfacemesh(:,:)
    integer, intent(in) :: id
    integer :: i, j, face, sfc

    call u2si(id, i, j, face)
    sfc = s2sfc(sfcfacemesh, i, j, face)
  end function u2sfc
  
  subroutine amb_cmp(mv_other)
    type (MetaVertex_t) :: mv_other
  end subroutine amb_cmp

  subroutine amb_genspacecurve(ne, Mesh)
    use spacecurve_mod, only: IsFactorable, genspacecurve

    integer, intent(in) :: ne
    integer, intent(out) :: Mesh(ne,ne)
    integer, allocatable :: Mesh2(:,:), Mesh2_map(:,:,:), sfcij(:,:)
    integer :: i, i2, j, j2, k, ne2, sfc_index

    if(IsFactorable(ne)) then
       call GenspaceCurve(Mesh)
       !      call PrintCurve(Mesh) 
    else
       ! find the smallest ne2 which is a power of 2 and ne2>ne
       ne2=2**ceiling( log(real(ne))/log(2d0) )
       if (ne2<ne) call abortmp('Fatal SFC error')

       allocate(Mesh2(ne2,ne2))
       allocate(Mesh2_map(ne2,ne2,2))
       allocate(sfcij(0:ne2*ne2,2))

       call GenspaceCurve(Mesh2)  ! SFC partition for ne2

       ! associate every element on the ne x ne mesh (Mesh)
       ! with its closest element on the ne2 x ne2 mesh (Mesh2)
       ! Store this as a map from Mesh2 -> Mesh in Mesh2_map.
       ! elements in Mesh2 which are not mapped get assigned a value of 0
       Mesh2_map=0
       do j=1,ne
          do i=1,ne
             ! map this element to an (i2,j2) element
             ! [ (i-.5)/ne , (j-.5)/ne ]  = [ (i2-.5)/ne2 , (j2-.5)/ne2 ]
             i2=nint( ((i-.5)/ne)*ne2 + .5 )
             j2=nint( ((j-.5)/ne)*ne2 + .5 )
             if (i2<1) i2=1
             if (i2>ne2) i2=ne2
             if (j2<1) j2=1
             if (j2>ne2) j2=ne2
             Mesh2_map(i2,j2,1)=i
             Mesh2_map(i2,j2,2)=j
          enddo
       enddo

       ! create a reverse index array for Mesh2
       ! k = Mesh2(i,j) 
       ! (i,j) = (sfcij(k,1),sfci(k,2)) 
       do j=1,ne2
          do i=1,ne2
             k=Mesh2(i,j)
             sfcij(k,1)=i
             sfcij(k,2)=j
          enddo
       enddo

       ! generate a SFC for Mesh with the same ordering as the 
       ! elements in Mesh2 which map to Mesh.
       sfc_index=0
       do k=0,ne2*ne2-1
          i2=sfcij(k,1)
          j2=sfcij(k,2)
          i=Mesh2_map(i2,j2,1)
          j=Mesh2_map(i2,j2,2)
          if (i/=0) then
             ! (i2,j2) element maps to (i,j) element
             Mesh(i,j)=sfc_index
             sfc_index=sfc_index+1
          endif
       enddo
#if 0
       print *,'SFC Mapping to non powers of 2,3 used.  Mesh:'  
       do j=1,ne
          write(*,'(99i3)') (Mesh(i,j),i=1,ne)
       enddo
       call PrintCurve(Mesh2) 
#endif
       deallocate(Mesh2)
       deallocate(Mesh2_map)
       deallocate(sfcij)
    endif
  end subroutine amb_genspacecurve

  subroutine amb_genspacepart(nelem, npart, sfc2rank)
    integer, intent(in) :: nelem, npart
    integer, intent(out) :: sfc2rank(npart+1)
    integer :: nelemd, ipart, extra, s1

    nelemd = nelem/npart
    ! every cpu gets nelemd elements, but the first 'extra' get nelemd+1
    extra = mod(nelem,npart)
    s1 = extra*(nelemd+1)

    ! split curve into two curves:
    ! 1 ... s1  s2 ... nelem
    !
    !  s1 = extra*(nelemd+1)         (count be 0)
    !  s2 = s1+1 
    !
    ! First region gets nelemd+1 elements per Processor
    ! Second region gets nelemd elements per Processor

    sfc2rank(1) = 0
    do ipart = 1, extra
       sfc2rank(ipart+1) = ipart*(nelemd+1)
    end do
    do ipart = extra+1, npart
       sfc2rank(ipart+1) = s1 + (ipart - extra)*nelemd
    end do
  end subroutine amb_genspacepart

  subroutine amb_cube_gv_pass1(gm)
    type (GridManager_t), intent(inout) :: gm
    logical(kind=1), allocatable :: owned_or_used(:)
    integer :: id, ne, nelem, sfc, i, j, k, id_nbr
    logical :: owned, used

    ne = size(gm%sfcfacemesh, 1)
    nelem = 6*ne*ne

    allocate(owned_or_used(nelem))
    owned_or_used = .false.

    do id = 1,nelem
       sfc = u2sfc(gm%sfcfacemesh, id)
       owned = sfc >= gm%sfc2rank(gm%rank+1) .and. sfc < gm%sfc2rank(gm%rank+2)

       if (.not. owned) cycle
       owned_or_used(id) = .true.

       call u2si(id, i, j, k)

       if (j >= 2 .and. i >= 2) then
          ! setup SOUTH, WEST, SW neighbors
          owned_or_used(s2ui(i-1,j,k)) = .true.
          owned_or_used(s2ui(i,j-1,k)) = .true.
          owned_or_used(s2ui(i-1,j-1,k)) = .true.
       end if
    end do

    deallocate(owned_or_used)

    ! owned_or_used(s2ui()) = .true.
  end subroutine amb_cube_gv_pass1
  
  subroutine gv_set(gv, dir, id, face, wgt)
    type (GridVertex_t), intent(inout) :: gv
    integer, intent(in) :: dir, id, face, wgt

    gv%nbrs(dir) = id
    gv%nbrs_face(dir) = face
    gv%nbrs_wgt(dir) = wgt
  end subroutine gv_set

  subroutine amb_cube_gv_pass2(gm)
    use dimensions_mod, only: np
    use control_mod, only : north, south, east, west, neast, seast, swest, nwest

    type (GridManager_t), intent(inout) :: gm
    integer :: igv, id, ne, ngv, sfc, i, j, k, id_nbr
    logical :: owned, used
    type (GridVertex_t), pointer :: gv
    integer, parameter :: EdgeWgtP = np, CornerWgt = 1

    return

    ne = size(gm%sfcfacemesh, 1)
    ngv = size(gm%gv)

    do igv = 1,ngv
       id = gm%gvid(igv)
       gv => gm%gv(igv)

       sfc = u2sfc(gm%sfcfacemesh, id)

       call u2si(id, i, j, k)

       gv%nbrs = -1
       if (j >= 2 .and. i >= 2) then
          ! setup SOUTH, WEST, SW neighbors
          call gv_set(gv, west, s2ui(i-1,j,k), k, EdgeWgtP)
          call gv_set(gv, south, s2ui(i,j-1,k), k, EdgeWgtP)
          call gv_set(gv, swest, s2ui(i-1,j-1,k), k, CornerWgt)
       end if
    end do
  end subroutine amb_cube_gv_pass2

  subroutine amb_ge(gm)
    type (GridManager_t), intent(inout) :: gm
  end subroutine amb_ge
end module amb_mod
