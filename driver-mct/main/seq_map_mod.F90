module seq_map_mod

  !---------------------------------------------------------------------
  !
  ! Purpose:
  !
  ! General mapping routines
  ! including self normalizing mapping routine with optional fraction
  !
  ! Author: T. Craig, Jan-28-2011
  !
  !---------------------------------------------------------------------

  use shr_kind_mod      ,only: R8 => SHR_KIND_R8, IN=>SHR_KIND_IN
  use shr_kind_mod      ,only: CL => SHR_KIND_CL, CX => SHR_KIND_CX
  use shr_sys_mod
  use shr_const_mod
  use shr_mct_mod, only: shr_mct_sMatPInitnc, shr_mct_queryConfigFile
  use mct_mod
  use seq_comm_mct
  use component_type_mod
  use seq_map_type_mod
  !amb
  use shr_reprosum_mod  ,only: shr_reprosum_calc
  use shr_infnan_mod    ,only: shr_infnan_isnan, shr_infnan_isinf

  implicit none
  save
  private  ! except
#include <mpif.h>
  
  !--------------------------------------------------------------------------
  ! Public interfaces
  !--------------------------------------------------------------------------

  public :: seq_map_init_rcfile     ! cpl pes
  public :: seq_map_init_rearrolap  ! cpl pes
  public :: seq_map_initvect        ! cpl pes
  public :: seq_map_map             ! cpl pes
  public :: seq_map_mapvect         ! cpl pes
  public :: seq_map_readdata        ! cpl pes

  interface seq_map_avNorm
     module procedure seq_map_avNormArr
     module procedure seq_map_avNormAvF
  end interface seq_map_avNorm

  !--------------------------------------------------------------------------
  ! Public data
  !--------------------------------------------------------------------------

  !--------------------------------------------------------------------------
  ! Private data
  !--------------------------------------------------------------------------

  character(*),parameter :: seq_map_stroff = 'variable_unset'
  character(*),parameter :: seq_map_stron  = 'StrinG_is_ON'
  real(R8),parameter,private :: deg2rad = shr_const_pi/180.0_R8  ! deg to rads

  !=======================================================================
contains
  !=======================================================================

  subroutine seq_map_init_rcfile( mapper, comp_s, comp_d, &
       maprcfile, maprcname, maprctype, samegrid, string, esmf_map)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)        ,intent(inout),pointer :: mapper
    type(component_type) ,intent(inout)         :: comp_s
    type(component_type) ,intent(inout)         :: comp_d
    character(len=*)     ,intent(in)            :: maprcfile
    character(len=*)     ,intent(in)            :: maprcname
    character(len=*)     ,intent(in)            :: maprctype
    logical              ,intent(in)            :: samegrid
    character(len=*)     ,intent(in),optional   :: string
    logical              ,intent(in),optional   :: esmf_map
    !
    ! Local Variables
    !
    type(mct_gsmap), pointer    :: gsmap_s ! temporary pointers
    type(mct_gsmap), pointer    :: gsmap_d ! temporary pointers
    integer(IN)                 :: mpicom
    character(CX)               :: mapfile, nl_mapfile
    character(CL)               :: maptype
    integer(IN)                 :: mapid
    character(len=128)          :: nl_label
    logical                     :: nl_found, conservative
    character(len=*),parameter  :: nl_strategy = 'X'
    character(len=*),parameter  :: subname = "(seq_map_init_rcfile) "
    !-----------------------------------------------------

    if (seq_comm_iamroot(CPLID) .and. present(string)) then
       write(logunit,'(A)') subname//' called for '//trim(string)
    endif

    call seq_comm_setptrs(CPLID, mpicom=mpicom)

    gsmap_s => component_get_gsmap_cx(comp_s)
    gsmap_d => component_get_gsmap_cx(comp_d)

    if (mct_gsmap_Identical(gsmap_s,gsmap_d)) then
       call seq_map_mapmatch(mapid,gsmap_s=gsmap_s,gsmap_d=gsmap_d,strategy="copy")

       if (mapid > 0) then
          call seq_map_mappoint(mapid,mapper)
       else
          call seq_map_mapinit(mapper,mpicom)
          mapper%copy_only = .true.
          mapper%strategy = "copy"
          mapper%gsmap_s => component_get_gsmap_cx(comp_s)
          mapper%gsmap_d => component_get_gsmap_cx(comp_d)
          mapper%nl_on = .false.
       endif

    elseif (samegrid) then
       call seq_map_mapmatch(mapid,gsmap_s=gsmap_s,gsmap_d=gsmap_d,strategy="rearrange")

       if (mapid > 0) then
          call seq_map_mappoint(mapid,mapper)
       else
          ! --- Initialize rearranger
          call seq_map_mapinit(mapper,mpicom)
          mapper%rearrange_only = .true.
          mapper%strategy = "rearrange"
          mapper%gsmap_s => component_get_gsmap_cx(comp_s)
          mapper%gsmap_d => component_get_gsmap_cx(comp_d)
          call seq_map_gsmapcheck(gsmap_s, gsmap_d)
          call mct_rearr_init(gsmap_s, gsmap_d, mpicom, mapper%rearr)
          mapper%nl_on = .false.
       endif

    else

       ! --- Initialize Smatp
       call shr_mct_queryConfigFile(mpicom,maprcfile,maprcname,mapfile,maprctype,maptype)

       nl_label = maprcname(1:len(maprcname)-1)//'_highorder:'
       call shr_mct_queryConfigFile(mpicom, maprcfile, trim(nl_label), nl_mapfile, &
            Label1Found=nl_found)
       if (nl_found) nl_found = nl_mapfile /= "idmap_ignore"
       conservative = .false.
       if (nl_found) conservative = seq_map_should_nonlinear_map_conserve(maprcname)

       call seq_map_mapmatch(mapid,gsMap_s=gsMap_s,gsMap_d=gsMap_d,mapfile=mapfile,strategy=maptype, &
            nl_on=nl_found,nl_mapfile=nl_mapfile,nl_conservative=conservative)

       if (mapid > 0) then
          call seq_map_mappoint(mapid,mapper)
       else
          call seq_map_mapinit(mapper,mpicom)
          mapper%mapfile = trim(mapfile)
          mapper%strategy= trim(maptype)
          mapper%gsmap_s => component_get_gsmap_cx(comp_s)
          mapper%gsmap_d => component_get_gsmap_cx(comp_d)

          call shr_mct_sMatPInitnc(mapper%sMatp, mapper%gsMap_s, mapper%gsMap_d, trim(mapfile),trim(maptype),mpicom)
          if (present(esmf_map)) mapper%esmf_map = esmf_map

          if (mapper%esmf_map) then
             call shr_sys_abort(subname//' ERROR: esmf SMM not supported')
          endif  ! esmf_map          

          ! Optional high-order map
          if (seq_comm_iamroot(CPLID)) print *,'amb> init',trim(nl_label)
          mapper%nl_on = nl_found
          if (nl_found) then
             if (seq_comm_iamroot(CPLID)) print *,'amb> init',trim(nl_mapfile)
             mapper%nl_on = .true.
             mapper%nl_conservative = conservative
             mapper%nl_mapfile = trim(nl_mapfile)
             call shr_mct_sMatPInitnc(mapper%nl_sMatp, mapper%gsMap_s, mapper%gsMap_d, &
                  trim(nl_mapfile), nl_strategy, mpicom)
          end if
       endif  ! mapid >= 0
    endif

    if (seq_comm_iamroot(CPLID)) then
       write(logunit,'(2A,I6,4A)') subname,' mapper counter, strategy, mapfile = ', &
            mapper%counter,' ',trim(mapper%strategy),' ',trim(mapper%mapfile)
       if (mapper%nl_on) then
          write(logunit,'(2A,I6,4A,2L)') subname, &
               ' mapper counter, nl_strategy, nl_conservative, nl_mapfile = ', &
               mapper%counter,' ',nl_strategy,' ',nl_conservative,trim(mapper%nl_mapfile)
       end if
       call shr_sys_flush(logunit)
    endif

    !amb
    mapper%dom_cx_s => comp_s%dom_cx
    mapper%dom_cx_d => comp_d%dom_cx

  end subroutine seq_map_init_rcfile

  !=======================================================================

  subroutine seq_map_init_rearrolap(mapper, comp_s, comp_d, string)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)        ,intent(inout),pointer :: mapper
    type(component_type) ,intent(inout)         :: comp_s
    type(component_type) ,intent(inout)         :: comp_d
    character(len=*)     ,intent(in),optional   :: string
    !
    ! Local Variables
    !
    integer(IN)                :: mapid
    type(mct_gsmap), pointer   :: gsmap_s
    type(mct_gsmap), pointer   :: gsmap_d
    integer(IN)                :: mpicom
    character(len=*),parameter :: subname = "(seq_map_init_rearrolap) "
    !-----------------------------------------------------

    if (seq_comm_iamroot(CPLID) .and. present(string)) then
       write(logunit,'(A)') subname//' called for '//trim(string)
    endif

    call seq_comm_setptrs(CPLID, mpicom=mpicom)

    gsmap_s => component_get_gsmap_cx(comp_s)
    gsmap_d => component_get_gsmap_cx(comp_d)

    if (mct_gsmap_Identical(gsmap_s,gsmap_d)) then
       call seq_map_mapmatch(mapid,gsmap_s=gsmap_s,gsmap_d=gsmap_d,strategy="copy")

       if (mapid > 0) then
          call seq_map_mappoint(mapid,mapper)
       else
          call seq_map_mapinit(mapper,mpicom)
          mapper%copy_only = .true.
          mapper%strategy = "copy"
          mapper%gsmap_s => component_get_gsmap_cx(comp_s)
          mapper%gsmap_d => component_get_gsmap_cx(comp_d)
          mapper%nl_on = .false.
       endif

    else
       call seq_map_mapmatch(mapid,gsmap_s=gsmap_s,gsmap_d=gsmap_d,strategy="rearrange")

       if (mapid > 0) then
          call seq_map_mappoint(mapid,mapper)
       else
          ! --- Initialize rearranger
          call seq_map_mapinit(mapper, mpicom)
          mapper%rearrange_only = .true.
          mapper%strategy = "rearrange"
          mapper%gsmap_s => component_get_gsmap_cx(comp_s)
          mapper%gsmap_d => component_get_gsmap_cx(comp_d)
          mapper%nl_on = .false.
          call seq_map_gsmapcheck(gsmap_s, gsmap_d)
          call mct_rearr_init(gsmap_s, gsmap_d, mpicom, mapper%rearr)
       endif

    endif

    if (seq_comm_iamroot(CPLID)) then
       write(logunit,'(2A,I6,4A)') subname,' mapper counter, strategy, mapfile = ', &
            mapper%counter,' ',trim(mapper%strategy),' ',trim(mapper%mapfile)
       call shr_sys_flush(logunit)
    endif

  end subroutine seq_map_init_rearrolap

  !=======================================================================

  subroutine seq_map_map( mapper, av_s, av_d, fldlist, norm, avwts_s, avwtsfld_s, &
       string, msgtag )

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   ,intent(inout)       :: mapper
    type(mct_aVect) ,intent(in)          :: av_s
    type(mct_aVect) ,intent(inout)       :: av_d
    character(len=*),intent(in),optional :: fldlist
    logical         ,intent(in),optional :: norm
    type(mct_aVect) ,intent(in),optional :: avwts_s
    character(len=*),intent(in),optional :: avwtsfld_s
    character(len=*),intent(in),optional :: string
    integer(IN)     ,intent(in),optional :: msgtag
    !
    ! Local Variables
    !
    logical :: lnorm
    integer(IN),save :: ltag    ! message tag for rearrange
    character(len=*),parameter :: subname = "(seq_map_map) "
    !-----------------------------------------------------

    if (seq_comm_iamroot(CPLID) .and. present(string)) then
       write(logunit,'(A)') subname//' called for '//trim(string)
    endif

    lnorm = .true.
    if (present(norm)) then
       lnorm = norm
    endif

    if (present(msgtag)) then
       ltag = msgtag
    else
       ltag = 2000
    endif

    if (present(avwts_s) .and. .not. present(avwtsfld_s)) then
       write(logunit,*) subname,' ERROR: avwts_s present but avwtsfld_s not'
       call shr_sys_abort(subname//' ERROR: avwts present')
    endif
    if (.not. present(avwts_s) .and. present(avwtsfld_s)) then
       write(logunit,*) subname,' ERROR: avwtsfld_s present but avwts_s not'
       call shr_sys_abort(subname//' ERROR: avwtsfld present')
    endif

    if (mapper%copy_only) then
       !-------------------------------------------
       ! COPY data
       !-------------------------------------------
       if (present(fldlist)) then
          call mct_aVect_copy(aVin=av_s,aVout=av_d,rList=fldlist,vector=mct_usevector)
       else
          call mct_aVect_copy(aVin=av_s,aVout=av_d,vector=mct_usevector)
       endif

    else if (mapper%rearrange_only) then
       !-------------------------------------------
       ! REARRANGE data
       !-------------------------------------------
       if (present(fldlist)) then
          call mct_rearr_rearrange_fldlist(av_s, av_d, mapper%rearr, tag=ltag, VECTOR=mct_usevector, &
               ALLTOALL=mct_usealltoall, fldlist=fldlist)
       else
          call mct_rearr_rearrange(av_s, av_d, mapper%rearr, tag=ltag, VECTOR=mct_usevector, &
               ALLTOALL=mct_usealltoall)
       endif

    else
       !-------------------------------------------
       ! MAP data
       !-------------------------------------------
       if (present(avwts_s)) then
          if (present(fldlist)) then
             call seq_map_avNorm(mapper, av_s, av_d, avwts_s, trim(avwtsfld_s), &
                  rList=fldlist, norm=lnorm)
          else
             call seq_map_avNorm(mapper, av_s, av_d, avwts_s, trim(avwtsfld_s), &
                  norm=lnorm)
          endif
       else
          if (present(fldlist)) then
             call seq_map_avNorm(mapper, av_s, av_d, rList=fldlist, norm=lnorm)
          else
             call seq_map_avNorm(mapper, av_s, av_d, norm=lnorm)
          endif
       endif
    end if

  end subroutine seq_map_map

  !=======================================================================

  subroutine seq_map_initvect(mapper, type, comp_s, comp_d, string)

    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)        ,intent(inout)       :: mapper
    character(len=*)     ,intent(in)          :: type
    type(component_type) ,intent(inout)       :: comp_s
    type(component_type) ,intent(inout)       :: comp_d
    character(len=*)     ,intent(in),optional :: string
    !
    ! Local Variables
    !
    type(mct_gGrid), pointer   :: dom_s
    type(mct_gGrid), pointer   :: dom_d
    integer(IN)                :: klon, klat, lsize, n
    character(len=CL)          :: lstring
    character(len=*),parameter :: subname = "(seq_map_initvect) "
    !-----------------------------------------------------

    lstring = ' '
    if (present(string)) then
       if (seq_comm_iamroot(CPLID)) write(logunit,'(A)') subname//' called for '//trim(string)
       lstring = trim(string)
    endif

    dom_s => component_get_dom_cx(comp_s)
    dom_d => component_get_dom_cx(comp_d)

    if (trim(type(1:6)) == 'cart3d') then
       if (mapper%cart3d_init == trim(seq_map_stron)) return

       !--- compute these up front for vector mapping ---
       lsize = mct_aVect_lsize(dom_s%data)
       allocate(mapper%slon_s(lsize),mapper%clon_s(lsize), &
            mapper%slat_s(lsize),mapper%clat_s(lsize))
       klon = mct_aVect_indexRa(dom_s%data, "lon" )
       klat = mct_aVect_indexRa(dom_s%data, "lat" )
       do n = 1,lsize
          mapper%slon_s(n) = sin(dom_s%data%rAttr(klon,n)*deg2rad)
          mapper%clon_s(n) = cos(dom_s%data%rAttr(klon,n)*deg2rad)
          mapper%slat_s(n) = sin(dom_s%data%rAttr(klat,n)*deg2rad)
          mapper%clat_s(n) = cos(dom_s%data%rAttr(klat,n)*deg2rad)
       enddo

       lsize = mct_aVect_lsize(dom_d%data)
       allocate(mapper%slon_d(lsize),mapper%clon_d(lsize), &
            mapper%slat_d(lsize),mapper%clat_d(lsize))
       klon = mct_aVect_indexRa(dom_d%data, "lon" )
       klat = mct_aVect_indexRa(dom_d%data, "lat" )
       do n = 1,lsize
          mapper%slon_d(n) = sin(dom_d%data%rAttr(klon,n)*deg2rad)
          mapper%clon_d(n) = cos(dom_d%data%rAttr(klon,n)*deg2rad)
          mapper%slat_d(n) = sin(dom_d%data%rAttr(klat,n)*deg2rad)
          mapper%clat_d(n) = cos(dom_d%data%rAttr(klat,n)*deg2rad)
       enddo
       mapper%cart3d_init = trim(seq_map_stron)
    endif

  end subroutine seq_map_initvect

  !=======================================================================

  subroutine seq_map_mapvect( mapper, type, av_s, av_d, fldu, fldv, norm, string )

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   ,intent(inout)       :: mapper
    character(len=*),intent(in)          :: type
    type(mct_aVect) ,intent(in)          :: av_s
    type(mct_aVect) ,intent(inout)       :: av_d
    character(len=*),intent(in)          :: fldu
    character(len=*),intent(in)          :: fldv
    logical         ,intent(in),optional :: norm
    character(len=*),intent(in),optional :: string
    !
    ! Local Variables
    !
    logical :: lnorm
    character(len=CL) :: lstring
    character(len=*),parameter :: subname = "(seq_map_mapvect) "
    !-----------------------------------------------------

    lstring = ' '
    if (present(string)) then
       if (seq_comm_iamroot(CPLID)) write(logunit,'(A)') subname//' called for '//trim(string)
       lstring = trim(string)
    endif

    if (mapper%copy_only .or. mapper%rearrange_only) then
       return
    endif

    lnorm = .true.
    if (present(norm)) then
       lnorm = norm
    endif

    if (trim(type(1:6)) == 'cart3d') then
       if (mapper%cart3d_init /= trim(seq_map_stron)) then
          call shr_sys_abort(trim(subname)//' ERROR: cart3d not initialized '//trim(lstring))
       endif
       call seq_map_cart3d(mapper, type, av_s, av_d, fldu, fldv, norm=lnorm, string=string)
    elseif (trim(type) == 'none') then
       call seq_map_map(mapper, av_s, av_d, fldlist=trim(fldu)//':'//trim(fldv), norm=lnorm)
    else
       write(logunit,*) subname,' ERROR: type unsupported ',trim(type)
       call shr_sys_abort(trim(subname)//' ERROR in type='//trim(type))
    end if

  end subroutine seq_map_mapvect

  !=======================================================================

  subroutine seq_map_cart3d( mapper, type, av_s, av_d, fldu, fldv, norm, string)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   ,intent(inout)       :: mapper
    character(len=*),intent(in)          :: type
    type(mct_aVect) ,intent(in)          :: av_s
    type(mct_aVect) ,intent(inout)       :: av_d
    character(len=*),intent(in)          :: fldu
    character(len=*),intent(in)          :: fldv
    logical         ,intent(in),optional :: norm
    character(len=*),intent(in),optional :: string
    !
    ! Local Variables
    !
    integer           :: lsize
    logical           :: lnorm
    integer           :: ku,kv,kux,kuy,kuz,n
    real(r8)          :: ue,un,ur,ux,uy,uz,speed
    real(r8)          :: urmaxl,urmax,uravgl,uravg,spavgl,spavg
    type(mct_aVect)   :: av3_s, av3_d
    integer(in)       :: mpicom,my_task,ierr,urcnt,urcntl
    character(len=*),parameter :: subname = "(seq_map_cart3d) "

    lnorm = .true.
    if (present(norm)) then
       lnorm=norm
    endif

    mpicom = mapper%mpicom

    ku = mct_aVect_indexRA(av_s, trim(fldu), perrwith='quiet')
    kv = mct_aVect_indexRA(av_s, trim(fldv), perrwith='quiet')

    if (ku /= 0 .and. kv /= 0) then
       lsize = mct_aVect_lsize(av_s)
       call mct_avect_init(av3_s,rList='ux:uy:uz',lsize=lsize)

       lsize = mct_aVect_lsize(av_d)
       call mct_avect_init(av3_d,rList='ux:uy:uz',lsize=lsize)

       kux = mct_aVect_indexRA(av3_s,'ux')
       kuy = mct_aVect_indexRA(av3_s,'uy')
       kuz = mct_aVect_indexRA(av3_s,'uz')
       lsize = mct_aVect_lsize(av_s)
       do n = 1,lsize
          ur = 0.0_r8
          ue = av_s%rAttr(ku,n)
          un = av_s%rAttr(kv,n)
          ux = mapper%clon_s(n)*mapper%clat_s(n)*ur - &
               mapper%clon_s(n)*mapper%slat_s(n)*un - &
               mapper%slon_s(n)*ue
          uy = mapper%slon_s(n)*mapper%clon_s(n)*ur - &
               mapper%slon_s(n)*mapper%slat_s(n)*un + &
               mapper%clon_s(n)*ue
          uz = mapper%slat_s(n)*ur + &
               mapper%clat_s(n)*un
          av3_s%rAttr(kux,n) = ux
          av3_s%rAttr(kuy,n) = uy
          av3_s%rAttr(kuz,n) = uz
       enddo

       call seq_map_map(mapper, av3_s, av3_d, norm=lnorm)

       kux = mct_aVect_indexRA(av3_d,'ux')
       kuy = mct_aVect_indexRA(av3_d,'uy')
       kuz = mct_aVect_indexRA(av3_d,'uz')
       lsize = mct_aVect_lsize(av_d)
       urmaxl = -1.0_r8
       uravgl = 0.0_r8
       urcntl = 0
       spavgl = 0.0_r8
       do n = 1,lsize
          ux = av3_d%rAttr(kux,n)
          uy = av3_d%rAttr(kuy,n)
          uz = av3_d%rAttr(kuz,n)
          ue = -mapper%slon_d(n)          *ux + &
               mapper%clon_d(n)          *uy
          un = -mapper%clon_d(n)*mapper%slat_d(n)*ux - &
               mapper%slon_d(n)*mapper%slat_d(n)*uy + &
               mapper%clat_d(n)*uz
          ur =  mapper%clon_d(n)*mapper%clat_d(n)*ux + &
               mapper%slon_d(n)*mapper%clat_d(n)*uy - &
               mapper%slat_d(n)*uz
          speed = sqrt(ur*ur + ue*ue + un*un)
          if (trim(type) == 'cart3d_diag' .or. trim(type) == 'cart3d_uvw_diag') then
             if (speed /= 0.0_r8) then
                urmaxl = max(urmaxl,abs(ur))
                uravgl = uravgl + abs(ur)
                spavgl = spavgl + speed
                urcntl = urcntl + 1
             endif
          endif
          if (type(1:10) == 'cart3d_uvw') then
             !--- this adds ur to ue and un, while preserving u/v angle and total speed ---
             if (un == 0.0_R8) then
                !--- if ue is also 0.0 then just give speed to ue, this is arbitrary ---
                av_d%rAttr(ku,n) = sign(speed,ue)
                av_d%rAttr(kv,n) = 0.0_r8
             else if (ue == 0.0_R8) then
                av_d%rAttr(ku,n) = 0.0_r8
                av_d%rAttr(kv,n) = sign(speed,un)
             else
                av_d%rAttr(ku,n) = sign(speed/sqrt(1.0_r8 + ((un*un)/(ue*ue))),ue)
                av_d%rAttr(kv,n) = sign(speed/sqrt(1.0_r8 + ((ue*ue)/(un*un))),un)
             endif
          else
             !--- this ignores ur ---
             av_d%rAttr(ku,n) = ue
             av_d%rAttr(kv,n) = un
          endif
       enddo
       if (trim(type) == 'cart3d_diag' .or. trim(type) == 'cart3d_uvw_diag') then
          call mpi_comm_rank(mpicom,my_task,ierr)
          call shr_mpi_max(urmaxl,urmax,mpicom,'urmax')
          call shr_mpi_sum(uravgl,uravg,mpicom,'uravg')
          call shr_mpi_sum(spavgl,spavg,mpicom,'spavg')
          call shr_mpi_sum(urcntl,urcnt,mpicom,'urcnt')
          if (my_task == 0 .and. urcnt > 0) then
             uravg = uravg / urcnt
             spavg = spavg / urcnt
             write(logunit,*) trim(subname),' cart3d uravg,urmax,spavg = ',uravg,urmax,spavg
          endif
       endif

       call mct_avect_clean(av3_s)
       call mct_avect_clean(av3_d)

    endif  ! ku,kv

  end subroutine seq_map_cart3d

  !=======================================================================

  subroutine seq_map_readdata(maprcfile, maprcname, mpicom, ID, &
       ni_s, nj_s, av_s, gsmap_s, avfld_s, filefld_s, &
       ni_d, nj_d, av_d, gsmap_d, avfld_d, filefld_d, string)

    !--- lifted from work by J Edwards, April 2011

    use shr_pio_mod, only : shr_pio_getiosys, shr_pio_getiotype
    use pio, only : pio_openfile, pio_closefile, pio_read_darray, pio_inq_dimid, &
         pio_inq_dimlen, pio_inq_varid, file_desc_t, io_desc_t, iosystem_desc_t, &
         var_desc_t, pio_int, pio_get_var, pio_double, pio_initdecomp, pio_freedecomp
    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    character(len=*),intent(in)             :: maprcfile
    character(len=*),intent(in)             :: maprcname
    integer(IN)     ,intent(in)             :: mpicom
    integer(IN)     ,intent(in)             :: ID
    integer(IN)     ,intent(out)  ,optional :: ni_s
    integer(IN)     ,intent(out)  ,optional :: nj_s
    type(mct_avect) ,intent(inout),optional :: av_s
    type(mct_gsmap) ,intent(in)   ,optional :: gsmap_s
    character(len=*),intent(in)   ,optional :: avfld_s
    character(len=*),intent(in)   ,optional :: filefld_s
    integer(IN)     ,intent(out)  ,optional :: ni_d
    integer(IN)     ,intent(out)  ,optional :: nj_d
    type(mct_avect) ,intent(inout),optional :: av_d
    type(mct_gsmap) ,intent(in)   ,optional :: gsmap_d
    character(len=*),intent(in)   ,optional :: avfld_d
    character(len=*),intent(in)   ,optional :: filefld_d
    character(len=*),intent(in)   ,optional :: string
    !
    ! Local Variables
    !
    type(iosystem_desc_t), pointer :: pio_subsystem
    integer(IN)       :: pio_iotype
    type(file_desc_t) :: File    ! PIO file pointer
    type(io_desc_t)   :: iodesc  ! PIO parallel io descriptor
    integer(IN)       :: rcode   ! pio routine return code
    type(var_desc_t)  :: vid     ! pio variable  ID
    integer(IN)       :: did     ! pio dimension ID
    integer(IN)       :: na      ! size of source domain
    integer(IN)       :: nb      ! size of destination domain
    integer(IN)       :: i       ! index
    integer(IN)       :: mytask  ! my task
    integer(IN), pointer :: dof(:)    ! DOF pointers for parallel read
    character(len=256):: fileName
    character(len=64) :: lfld_s, lfld_d, lfile_s, lfile_d
    character(*),parameter :: areaAV_field = 'aream'
    character(*),parameter :: areafile_s   = 'area_a'
    character(*),parameter :: areafile_d   = 'area_b'
    character(len=*),parameter :: subname  = "(seq_map_readdata) "
    !-----------------------------------------------------

    if (seq_comm_iamroot(CPLID) .and. present(string)) then
       write(logunit,'(A)') subname//' called for '//trim(string)
       call shr_sys_flush(logunit)
    endif

    call MPI_COMM_RANK(mpicom,mytask,rcode)

    lfld_s = trim(areaAV_field)
    if (present(avfld_s)) then
       lfld_s = trim(avfld_s)
    endif

    lfld_d = trim(areaAV_field)
    if (present(avfld_d)) then
       lfld_s = trim(avfld_d)
    endif

    lfile_s = trim(areafile_s)
    if (present(filefld_s)) then
       lfile_s = trim(filefld_s)
    endif

    lfile_d = trim(areafile_d)
    if (present(filefld_d)) then
       lfile_d = trim(filefld_d)
    endif

    call I90_allLoadF(trim(maprcfile),0,mpicom,rcode)
    if(rcode /= 0) then
       write(logunit,*)"Cant find maprcfile file ",trim(maprcfile)
       call shr_sys_abort(trim(subname)//"i90_allLoadF File Not Found")
    endif

    call i90_label(trim(maprcname),rcode)
    if(rcode /= 0) then
       write(logunit,*)"Cant find label ",maprcname
       call shr_sys_abort(trim(subname)//"i90_label Not Found")
    endif

    call i90_gtoken(filename,rcode)
    if(rcode /= 0) then
       write(logunit,*)"Error reading token ",filename
       call shr_sys_abort(trim(subname)//"i90_gtoken Error on filename read")
    endif

    pio_subsystem => shr_pio_getiosys(ID)
    pio_iotype = shr_pio_getiotype(ID)

    rcode = pio_openfile(pio_subsystem, File, pio_iotype, filename)

    if (present(ni_s)) then
       rcode = pio_inq_dimid (File, 'ni_a', did)  ! number of lons in input grid
       rcode = pio_inq_dimlen(File, did  , ni_s)
    end if
    if(present(nj_s)) then
       rcode = pio_inq_dimid (File, 'nj_a', did)  ! number of lats in input grid
       rcode = pio_inq_dimlen(File, did  , nj_s)
    end if
    if(present(ni_d)) then
       rcode = pio_inq_dimid (File, 'ni_b', did)  ! number of lons in output grid
       rcode = pio_inq_dimlen(File, did  , ni_d)
    end if
    if(present(nj_d)) then
       rcode = pio_inq_dimid (File, 'nj_b', did)  ! number of lats in output grid
       rcode = pio_inq_dimlen(File, did  , nj_d)
    endif

    !--- read and load area_a ---
    if (present(av_s)) then
       if (.not.present(gsmap_s)) then
          call shr_sys_abort(trim(subname)//' ERROR av_s must have gsmap_s')
       endif
       rcode = pio_inq_dimid (File, 'n_a', did)  ! size of  input vector
       rcode = pio_inq_dimlen(File, did  , na)
       i = mct_avect_indexra(av_s, trim(lfld_s))
       call mct_gsmap_OrderedPoints(gsMap_s, mytask, dof)
       call pio_initdecomp(pio_subsystem, pio_double, (/na/), dof, iodesc)
       deallocate(dof)
       rcode = pio_inq_varid(File,trim(lfile_s),vid)
       call pio_read_darray(File, vid, iodesc, av_s%rattr(i,:), rcode)
       call pio_freedecomp(File,iodesc)
    end if

    !--- read and load area_b ---
    if (present(av_d)) then
       if (.not.present(gsmap_d)) then
          call shr_sys_abort(trim(subname)//' ERROR av_d must have gsmap_d')
       endif
       rcode = pio_inq_dimid (File, 'n_b', did)  ! size of output vector
       rcode = pio_inq_dimlen(File, did  , nb)
       i = mct_avect_indexra(av_d, trim(lfld_d))
       call mct_gsmap_OrderedPoints(gsMap_d, mytask, dof)
       call pio_initdecomp(pio_subsystem, pio_double, (/nb/), dof, iodesc)
       deallocate(dof)
       rcode = pio_inq_varid(File,trim(lfile_d),vid)
       call pio_read_darray(File, vid, iodesc, av_d%rattr(i,:), rcode)
       call pio_freedecomp(File,iodesc)
    endif


    call pio_closefile(File)

  end subroutine seq_map_readdata

  !=======================================================================

  subroutine seq_map_avNormAvF(mapper, av_i, av_o, avf_i, avfifld, rList, norm)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   , intent(inout)       :: mapper  ! mapper
    type(mct_aVect) , intent(in)          :: av_i    ! input
    type(mct_aVect) , intent(inout)       :: av_o    ! output
    type(mct_aVect) , intent(in)          :: avf_i   ! extra src "weight"
    character(len=*), intent(in)          :: avfifld ! field name in avf_i
    character(len=*), intent(in),optional :: rList   ! fields list
    logical         , intent(in),optional :: norm    ! normalize at end
    !
    integer(IN) :: lsize_i, lsize_f, kf, j
    real(r8),allocatable :: frac_i(:)
    logical :: lnorm
    character(*),parameter :: subName = '(seq_map_avNormAvF) '
    !-----------------------------------------------------

    lnorm = .true.
    if (present(norm)) then
       lnorm = norm
    endif

    lsize_i = mct_aVect_lsize(av_i)
    lsize_f = mct_aVect_lsize(avf_i)

    if (lsize_i /= lsize_f) then
       write(logunit,*) subname,' ERROR: lsize_i ne lsize_f ',lsize_i,lsize_f
       call shr_sys_abort(subname//' ERROR size_i ne lsize_f')
    endif

    !--- extract frac_i field from avf_i to pass to seq_map_avNormArr ---
    allocate(frac_i(lsize_i))
    do j = 1,lsize_i
       kf = mct_aVect_indexRA(avf_i,trim(avfifld))
       frac_i(j) = avf_i%rAttr(kf,j)
    enddo

    if (present(rList)) then
       call seq_map_avNormArr(mapper, av_i, av_o, frac_i, rList=rList, norm=lnorm)
    else
       call seq_map_avNormArr(mapper, av_i, av_o, frac_i, norm=lnorm)
    endif

    deallocate(frac_i)

  end subroutine seq_map_avNormAvF

  !=======================================================================

  subroutine seq_map_avNormArr(mapper, av_i, av_o, norm_i, rList, norm)

    implicit none
    !-----------------------------------------------------
    !
    ! Arguments
    !
    type(seq_map)   , intent(inout) :: mapper! mapper
    type(mct_aVect) , intent(in)    :: av_i  ! input
    type(mct_aVect) , intent(inout) :: av_o  ! output
    real(r8)        , intent(in), optional :: norm_i(:)  ! source "weight"
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
    character(*),parameter :: subName = '(seq_map_avNormArr) '
    character(len=*),parameter :: ffld = 'norm8wt'  ! want something unique
    !amb
    character(len=*), parameter :: afldname  = 'aream'
    character(len=128) :: msg
    logical :: amroot, verbose, infnanfilt
    integer(IN) :: mpicom, ierr, iam, k, natt, nsum, nfld, kArea, lidata(2), gidata(2), i, n
    integer(IN), dimension(:), allocatable :: idxs_need_safety, mask_safety
    real(r8) :: tmp, area, lo, hi
    real(r8), allocatable, dimension(:) :: lmins, gmins, lmaxs, gmaxs, glbl_masses, gwts, gwts_safety
    real(r8), allocatable, dimension(:,:) :: dof_masses, caas_wgt, oglims, lcl_lo, lcl_hi
    !-----------------------------------------------------

    infnanfilt = .false.
    verbose = .true.
    call seq_comm_setptrs(CPLID, mpicom=mpicom)
    call mpi_comm_rank(mpicom, iam, ierr)
    amroot = iam == 0

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
       if (mapper%nl_on) then
          call mct_aVect_init(nl_avp_o, rList=trim( rList)//trim(appnd), lsize=lsize_o)
       end if
    else
       lrList = mct_aVect_exportRList2c(av_i)
       call mct_aVect_init(avp_i, rList=trim(lrList)//trim(appnd), lsize=lsize_i)
       lrList = mct_aVect_exportRList2c(av_o)
       call mct_aVect_init(avp_o, rList=trim(lrList)//trim(appnd), lsize=lsize_o)
       if (mapper%nl_on) then
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

    if (mapper%nl_on) then
       if (verbose .and. amroot) then
          print *,'amb> ', trim(mapper%nl_mapfile), ' ', &
               trim(mapper%strategy), mapper%nl_conservative, lnorm, present(norm_i)
       end if
       natt = size(avp_i%rAttr, 1)
       allocate(lcl_lo(natt,lsize_o), lcl_hi(natt,lsize_o))
       call sMat_avMult_and_calc_bounds(avp_i, mapper%nl_sMatp, nl_avp_o, &
            lcl_lo, lcl_hi, infnanfilt)
       ! Compute global bounds.
       allocate(lmins(natt), gmins(natt), lmaxs(natt), gmaxs(natt))
       lmins(:) =  1.e30_r8
       lmaxs(:) = -1.e30_r8
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
             print '(a,i2,a,i2,es23.15,es23.15)', &
                  'amb> src-bnds ', k, '/', natt, gmins(k), gmaxs(k)
          end do
       end if
       if (infnanfilt) then
          do k = 1,natt
             if (shr_infnan_isnan(gmins(k)) .or. shr_infnan_isinf(gmins(k))) gmins(k) = 0
             if (shr_infnan_isnan(gmaxs(k)) .or. shr_infnan_isinf(gmaxs(k))) gmaxs(k) = 0
          end do
       end if
       if (verbose) then
          allocate(oglims(natt,2))
          lmins(:) =  1.e30_r8
          lmaxs(:) = -1.e30_r8
          do j = 1,lsize_o
             do k = 1,natt
                tmp = nl_avp_o%rAttr(k,j)
                if (infnanfilt) then
                   if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) cycle
                end if
                lmins(k) = min(lmins(k), tmp)
                lmaxs(k) = max(lmaxs(k), tmp)
             end do
          end do
          if (infnanfilt) then
             do k = 1,natt
                if (lmins(k) > lmaxs(k)) then
                   lmins(k) = 0
                   lmaxs(k) = 0
                end if
             end do
          end if
          call mpi_allreduce(lmins, oglims(:,1), natt, MPI_DOUBLE_PRECISION, MPI_MIN, mpicom, ierr)
          call mpi_allreduce(lmaxs, oglims(:,2), natt, MPI_DOUBLE_PRECISION, MPI_MAX, mpicom, ierr)
          if (amroot) then
             do k = 1,natt
                print '(a,i2,a,i2,es23.15,es23.15)', &
                     'amb> pre-bnds ', k, '/', natt, oglims(k,1), oglims(k,2)
             end do
          end if
          deallocate(oglims)
       end if
       deallocate(lmins, lmaxs)
       ! Mask high-order field against low-order. Occasionally an exact 0 in the
       ! low-order field will map the high-order field unnecessarily, but that's
       ! OK: it's a local reduction in order to one, not a wrong value.
       do j = 1,lsize_o
          do k = 1,natt
             if (avp_o%rAttr(k,j) == 0) nl_avp_o%rAttr(k,j) = 0
          end do
       end do
       ! Compute global mass in low-order and high-order fields.
       kArea = mct_aVect_indexRA(mapper%dom_cx_d%data, afldname)
       nsum = lsize_o
       nfld = 2*natt
       allocate(dof_masses(nsum,nfld), glbl_masses(nfld)) ! low- and high-order
       if (mct_aVect_lSize(mapper%dom_cx_d%data) /= lsize_o) then
          print *,'amb> sizes do not match',lsize_o,mct_aVect_lSize(mapper%dom_cx_d%data)
          call shr_sys_abort(subname//' ERROR: amb> sizes do not match')
       end if
       do j = 1,lsize_o
          area = mapper%dom_cx_d%data%rAttr(kArea,j)
          dof_masses(j,1:natt) = avp_o%rAttr(:,j)*area
          dof_masses(j,natt+1:nfld) = nl_avp_o%rAttr(:,j)*area
       end do
       if (infnanfilt) then
          do k = 1,nfld
             do j = 1,lsize_o
                if (shr_infnan_isnan(dof_masses(j,k)) .or. shr_infnan_isinf(dof_masses(j,k))) &
                     dof_masses(j,k) = 0
             end do
          end do
       end if
       call shr_reprosum_calc(dof_masses, glbl_masses, nsum, nsum, nfld)
       deallocate(dof_masses)
       ! Clip against source-derived bounds.
       nsum = lsize_o
       nfld = 3*natt
       allocate(caas_wgt(nsum,nfld)) ! dm, cap low, cap high
       caas_wgt(:,:) = 0
       do j = 1,lsize_o
          area = mapper%dom_cx_d%data%rAttr(kArea,j)
          do k = 1,natt
             tmp = nl_avp_o%rAttr(k,j)
             if (infnanfilt) then
                if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                   tmp = 0
                   nl_avp_o%rAttr(k,j) = tmp
                end if
             end if
             lo = lcl_lo(k,j)
             hi = lcl_hi(k,j)
             if (tmp < lo) then
                caas_wgt(j,k) = (tmp - lo)*area
                nl_avp_o%rAttr(k,j) = lo
             else if (tmp > hi) then
                caas_wgt(j,k) = (tmp - hi)*area
                nl_avp_o%rAttr(k,j) = hi
             end if
             tmp = nl_avp_o%rAttr(k,j)
             caas_wgt(j,  natt+k) = (tmp - lo)*area
             caas_wgt(j,2*natt+k) = (hi - tmp)*area
          end do
       end do
       allocate(gwts(nfld))
       call shr_reprosum_calc(caas_wgt, gwts, nsum, nsum, nfld)
       ! Combine clipping and global mass error into a single dm value.
       gwts(1:natt) = gwts(1:natt) + (glbl_masses(1:natt) - glbl_masses(natt+1:2*natt))
       ! Check whether we need to relax to the safety problem.
       allocate(idxs_need_safety(natt), mask_safety(natt))
       mask_safety(:) = 0
       n = 0
       do k = 1,natt
          if (((gwts(k) < 0 .and. -gwts(k) > gwts(  natt+k))  .or.  &
               (gwts(k) > 0 .and.  gwts(k) > gwts(2*natt+k))) .and. &
               ! A common case where the above filter triggers unnecessarily is
               ! when gmins(k) == gmaxs(k) and there is a machine-precision-
               ! level global mass difference in the low- and high-order maps.
               ! Skip this index in this case.
               gmins(k) < gmaxs(k)) then
             idxs_need_safety(n+1) = k
             mask_safety(k) = 1
             n = n + 1
             if (verbose .and. amroot) then
                print '(a,i2,a,i2,3es25.15)','amb>   safety', &
                     k, '/', natt, gwts(k), gwts(natt+k), gwts(2*natt+k)
             end if
          end if
       end do
       ! Solve safety problem where needed.
       if (n > 0) then
          do j = 1,lsize_o
             area = mapper%dom_cx_d%data%rAttr(kArea,j)
             do i = 1,n
                k = idxs_need_safety(i)
                tmp = nl_avp_o%rAttr(k,j)
                if (infnanfilt) then
                   if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) then
                      tmp = 0
                      nl_avp_o%rAttr(k,j) = tmp
                   end if
                end if
                lo = gmins(k)
                hi = gmaxs(k)
                if (tmp < lo) then
                   caas_wgt(j,i) = (tmp - lo)*area
                   nl_avp_o%rAttr(k,j) = lo
                else if (tmp > hi) then
                   caas_wgt(j,i) = (tmp - hi)*area
                   nl_avp_o%rAttr(k,j) = hi
                end if
                tmp = nl_avp_o%rAttr(k,j)
                caas_wgt(j,  n+i) = (tmp - lo)*area
                caas_wgt(j,2*n+i) = (hi - tmp)*area
             end do
          end do
          nfld = 3*n
          allocate(gwts_safety(nfld))
          call shr_reprosum_calc(caas_wgt, gwts_safety, nsum, nsum, nfld)
          do i = 1,n
             k = idxs_need_safety(i)
             gwts(k) = gwts_safety(i) + (glbl_masses(k) - glbl_masses(natt+k))
          end do
          deallocate(gwts_safety, idxs_need_safety)
       end if
       deallocate(caas_wgt)
       if (verbose .and. amroot) then
          do k = 1,natt
             if (gwts(k) /= 0 .or. glbl_masses(k) /= 0) then
                print '(a,i2,a,i2,es23.15,es23.15,es23.15,es10.2)', &
                     'amb>  caas-dm ', k, '/', natt, &
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
       ! Adjust high-order solution and set avp_o.
       do k = 1,natt
          if (mask_safety(k) == 1) then
             lo = gmins(k)
             hi = gmaxs(k)
             if (gwts(k) > 0) then
                tmp = gwts(2*natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      avp_o%rAttr(k,j) = nl_avp_o%rAttr(k,j) + &
                           ((hi - nl_avp_o%rAttr(k,j))/tmp)*gwts(k)
                   end do
                end if
             else if (gwts(k) < 0) then
                tmp = gwts(natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      avp_o%rAttr(k,j) = nl_avp_o%rAttr(k,j) + &
                           ((nl_avp_o%rAttr(k,j) - lo)/tmp)*gwts(k)
                   end do
                end if
             end if
          else
             if (gwts(k) > 0) then
                tmp = gwts(2*natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      hi = lcl_hi(k,j)
                      avp_o%rAttr(k,j) = nl_avp_o%rAttr(k,j) + &
                           ((hi - nl_avp_o%rAttr(k,j))/tmp)*gwts(k)
                   end do
                end if
             else if (gwts(k) < 0) then
                tmp = gwts(natt+k)
                if (tmp /= 0) then
                   do j = 1,lsize_o
                      lo = lcl_lo(k,j)
                      avp_o%rAttr(k,j) = nl_avp_o%rAttr(k,j) + &
                           ((nl_avp_o%rAttr(k,j) - lo)/tmp)*gwts(k)
                   end do
                end if
             end if
          end if
       end do
       deallocate(gwts, lcl_lo, lcl_hi, mask_safety)
       call mct_aVect_clean(nl_avp_o)
       ! Clip for numerics, just against the global extrema.
       do j = 1,lsize_o
          do k = 1,natt
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
          if (infnanfilt) then
             do k = 1,natt
                do j = 1,lsize_o
                   if (shr_infnan_isnan(dof_masses(j,k)) .or. shr_infnan_isinf(dof_masses(j,k))) &
                        dof_masses(j,k) = 0
                end do
             end do
          end if
          call shr_reprosum_calc(dof_masses, gwts, nsum, nsum, natt)
          deallocate(dof_masses)
          if (amroot) then
             do k = 1,natt
                if (gwts(k) /= 0 .or. glbl_masses(k) /= 0) then
                   tmp = (gwts(k) - glbl_masses(k))/abs(glbl_masses(k))
                   if (tmp < 1e-15) then
                      msg = ''
                   else if (tmp < 1e-13) then
                      msg = ' OK'
                   else
                      msg = ' ALARM'
                   end if
                   print '(a,i2,a,i2,es23.15,es23.15,es10.2,a)', &
                        'amb> fin-mass ', k, '/', natt, glbl_masses(k), gwts(k), tmp, trim(msg)
                end if
             end do
          end if
          deallocate(gwts)
          ! check bounds
          allocate(lmins(natt), lmaxs(natt), oglims(natt,2))
          lmins(:) =  1.e30_r8
          lmaxs(:) = -1.e30_r8
          do j = 1,lsize_o
             do k = 1,natt
                tmp = avp_o%rAttr(k,j)
                if (infnanfilt) then
                   if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) cycle
                end if
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
                print '(a,i2,a,i2,es23.15,es23.15,a)', &
                     'amb> fin-bnds ', k, '/', natt, oglims(k,1), oglims(k,2), trim(msg)
             end do
          end if
          deallocate(lmins, lmaxs, oglims)
       end if
       deallocate(gmins, gmaxs, glbl_masses)
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

    !--- copy back into av_o and we are done ---

    call mct_aVect_copy(aVin=avp_o, aVout=av_o, VECTOR=mct_usevector)

    call mct_aVect_clean(avp_i)
    call mct_aVect_clean(avp_o)

  end subroutine seq_map_avNormArr

  subroutine sMat_avMult_and_calc_bounds(xAV, sMatPlus, yAV, lo, hi, infnanfilt)
    ! Compute
    !     x' = rearrange(x)
    !     y' = A*x'
    !     l, u = bounds(A, x').
    ! l, u can the ben used to compute
    !     y = clip(y', l, u).
    ! For each entry i, bounds(A, x) returns min/maxval(x such that A(i,:) is a
    ! structural non-0). That is, l(i), u(i) are bounds derived from the
    ! discrete domain of dependence of y(i).
    !   During initialization, strategy 'X' ('Xonly') was specified. Thus each
    ! y(i) has full access to its discrete domain of dependence.

    type (mct_aVect), intent(in)    :: xAV
    type (mct_sMatp), intent(inout) :: sMatPlus
    type (mct_aVect), intent(out)   :: yAV
    real(r8), dimension(:,:), intent(out) :: lo, hi
    logical, intent(in) :: infnanfilt

    type (mct_aVect) :: xPrimeAV
    integer :: ierr, ne, natt, irow, icol, iwgt, i, j, row, col, ysize
    real(r8) :: wgt, tmp

    ! y = 0
    call mct_aVect_init(xPrimeAV, xAV, sMatPlus%XPrimeLength)
    call mct_aVect_zero(xPrimeAV)
    ! x' = rearrange(x)
    call mct_rearr_rearrange(xAV, xPrimeAV, sMatPlus%XToXPrime, &
         tag=sMatPlus%Tag, vector=mct_usevector, &
         alltoall=.true., handshake=.true.)
    ! y' = A*x'
    call mct_sMat_avMult(xPrimeAV, sMatPlus%Matrix, yAV, vector=mct_usevector)
    ! l, u = bounds(A, x')
    ysize = mct_aVect_lsize(yAV)
    natt = size(yAV%rAttr, 1)
    lo(:,:) =  1.e30_r8
    hi(:,:) = -1.e30_r8
    ne = mct_sMat_lsize(sMatPlus%Matrix)
    irow = mct_sMat_indexIA(sMatPlus%Matrix,'lrow')
    icol = mct_sMat_indexIA(sMatPlus%Matrix,'lcol')
    iwgt = mct_sMat_indexRA(sMatPlus%Matrix,'weight')
    do i = 1, ne
       row = sMatPlus%Matrix%data%iAttr(irow,i)
       col = sMatPlus%Matrix%data%iAttr(icol,i)
       wgt = sMatPlus%Matrix%data%rAttr(iwgt,i)
       if (wgt == 0) cycle
       do j = 1, natt
          tmp = xPrimeAV%rAttr(j,col)
          if (infnanfilt) then
             if (shr_infnan_isnan(tmp) .or. shr_infnan_isinf(tmp)) cycle
          end if
          lo(j,row) = min(lo(j,row), tmp)
          hi(j,row) = max(hi(j,row), tmp)
       end do
    end do
    if (infnanfilt) then
       do i = 1, ysize
          do j = 1, natt
             if (lo(j,i) > hi(j,i)) then
                lo(j,i) = 0
                hi(j,i) = 0
             end if
          end do
       end do
    end if
    call mct_aVect_clean(xPrimeAV, ierr)
  end subroutine sMat_avMult_and_calc_bounds

end module seq_map_mod
