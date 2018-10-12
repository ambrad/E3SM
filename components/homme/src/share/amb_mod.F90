module amb_mod
  use metagraph_mod, only: MetaVertex_t
  use parallel_mod, only: parallel_t, abortmp

  implicit none

  

contains

  subroutine amb_run(par)
    use dimensions_mod, only: nelem, ne, npart

    type (parallel_t), intent(in) :: par
    integer :: sfc2rank(npart+1), mesh(ne,ne)
    logical, parameter :: dbg = .false.

    call amb_genspacecurve(ne, mesh)
    call amb_genspacepart(nelem, npart, sfc2rank)
    if (dbg) then
       print '(a,i4,a)', 'AMB> nelem', nelem, 'sfc2rank'
       print '(i4)', sfc2rank
    end if
  end subroutine amb_run
  
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
       if (ne2<ne) call abortmp('Fatel SFC error')

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
    if (sfc2rank(npart+1) /= nelem) then
       print *, 'nelem',nelem,'sfc2rank',sfc2rank
    end if
  end subroutine amb_genspacepart

end module amb_mod
