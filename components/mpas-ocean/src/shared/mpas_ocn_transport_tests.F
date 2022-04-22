! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_transport_tests
!
!> \brief MPAS ocean transport test suite module
!> \author Andrew Bradley and Peter Bosler (Sandia National Laboratories)
!> \date   02/10/21
!> \details
!>  This module contains routines for the 2D transport tests defined in
!>
!> P. H. Lauritzen, W. C. Skamarock, M. J. Prather, and M. A. Taylor, 2012,
!> "A standard test case suite for two-dimensional linear transport on the sphere,"
!> Geosci. Model Dev. 5:887--901.
!>
!> and the 3D transport tests defined in
!>
!> J. Kent, P. A. Ullrich, and C. Jablonowski, 2014, "Dynamical core model
!> intercomparison project: Tracer transport test cases," Q. J. Roy. Met. Soc.
!> 140:681, 1279--1293.
!>
!>
!> ** This file is in 'shared' because it is used by both forward mode and init mode **
!>
!
!-----------------------------------------------------------------------
module ocn_transport_tests

  use mpas_kind_types
  use mpas_io_units
  use mpas_derived_types
  use mpas_pool_routines
  use mpas_constants
  use mpas_stream_manager
  use mpas_dmpar
  use mpi !amb debug

#ifndef EXCLUDE_INIT_MODE
  use ocn_config, only : config_init_configuration, &
                         config_ocean_run_mode, &
                         config_transport_tests_flow_id, &
                         config_transport_tests_salinity, &
                         config_transport_tests_temperature, &
                         config_transport_tests_vert_levels
#else
  use ocn_config, only:  config_ocean_run_mode, &
                         config_transport_tests_flow_id, &
                         config_transport_tests_salinity, &
                         config_transport_tests_temperature, &
                         config_transport_tests_vert_levels
#endif

  implicit none
  private

  public :: ocn_transport_test_init_setup, &
            ocn_transport_test_init_validate, &
            ocn_transport_test_velocity_edge, &
            ocn_transport_test_velocity_vert

  !--------------------------------------------------------------------
  !
  ! Private module variables
  !
  !--------------------------------------------------------------------
  real(kind=RKIND), parameter :: &
       pi = 3.141592653589793d0, two_pi = 2.0_RKIND*pi, &
       one_day = 86400.0_RKIND, twelve_days = 12.0_RKIND*one_day, &
       ztop = 12000.0_RKIND, T0 = 300.0_RKIND, Rd = 287.04_RKIND, &
       grav = 9.80616_RKIND, p0 = 100000.0_RKIND, H = Rd*T0/grav

  character(len=32), parameter :: tc1str = "rotation"
  character(len=32), parameter :: tc2str = "nondivergent2D"
  character(len=32), parameter :: tc3str = "divergent2D"
  character(len=32), parameter :: tc4str = "correlatedTracers2D"
  character(len=32), parameter :: tc5str = "nondivergent3D"

#ifdef MPAS_DEBUG
  logical, parameter :: flush_log = .true.
#endif

  !amb debug stuff
  logical :: first = .true.
  integer :: mpirank, comm_global, verbosity
  logical :: amroot
  public :: amb_amroot, amb_get_rank, amb_get_verbosity

!***********************************************************************
  contains

  !--------------------------------------------------------------------
  !
  ! Public module functions
  !
  !--------------------------------------------------------------------

!***********************************************************************
!
!  routine ocn_transport_test_init_setup
!
!> \brief Initializes transport test suite
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>  Initializes constants and choices for transport test suite.
!
!-----------------------------------------------------------------------
  subroutine ocn_transport_test_init_setup(domain, err)
    type (domain_type), intent(inout) :: domain
    integer, intent(out) :: err

    type (block_type), pointer :: block_ptr
    type (mpas_pool_type), pointer :: meshPool, verticalMeshPool, statePool, &
         diagnosticsPool, forcingPool, tracersPool
    integer, pointer :: nVertLevels, nCellsSolve, nEdgesSolve, nEdge, &
         nVerticesSolve, index_temperature, index_salinity, index_tracer1
    integer, dimension(:), pointer :: minLevelCell, maxLevelCell
    real (kind=RKIND), pointer :: sphere_radius
    real (kind=RKIND), dimension(:), pointer :: refBottomDepth, refZMid, &
         lonCell, latEdge, latCell, latVertex, bottomDepth, fCell, fEdge, &
         fVertex, xCell, yCell
    real (kind=RKIND), dimension(:,:), pointer :: layerThickness, &
         restingThickness, normalVelocity
    real (kind=RKIND), dimension(:,:,:), pointer :: activeTracers,&
                                                    debugTracers
    character (len=StrKIND), pointer :: config_init_configuration
    integer :: iCell, iEdge, iVertex, k, i
    real (kind=RKIND) :: vs, shape
    logical :: is_3D

    err = 0

#ifdef MPAS_DEBUG
    call mpas_log_write("entered subroutine ocn_transport_tests_init_setup", MPAS_LOG_OUT, flushNow=flush_log)
    call mpas_pool_get_config(domain % configs, 'config_init_configuration', config_init_configuration)
    call mpas_log_write("config data : "//trim(config_init_configuration), MPAS_LOG_OUT, intArgs=[config_transport_tests_flow_id], realArgs=[config_transport_tests_temperature, config_transport_tests_salinity], flushNow=flush_log)
#endif

    block_ptr => domain % blocklist
    do while(associated(block_ptr))
       call mpas_pool_get_subpool(block_ptr % structs, 'mesh',meshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'verticalMesh',verticalMeshPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'state',statePool)
       call mpas_pool_get_subpool(block_ptr % structs, 'diagnostics',diagnosticsPool)
       call mpas_pool_get_subpool(block_ptr % structs, 'forcing',forcingPool)

       call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)

       call mpas_pool_get_dimension(meshPool, 'nVertLevels',nVertLevels)
       call mpas_pool_get_dimension(meshPool, 'nCellsSolve',nCellsSolve)
       call mpas_pool_get_dimension(meshPool, 'nEdge', nEdge)
       call mpas_pool_get_dimension(meshPool, 'nEdgesSolve',nEdgesSolve)
       call mpas_pool_get_dimension(meshPool, 'nVerticesSolve',nVerticesSolve)
       call mpas_pool_get_config(meshPool, 'sphere_radius', sphere_radius)

       call mpas_pool_get_dimension(tracersPool, 'index_temperature',index_temperature)
       call mpas_pool_get_dimension(tracersPool, 'index_salinity',index_salinity)
       call mpas_pool_get_dimension(tracersPool, 'index_tracer1',index_tracer1)

       call mpas_pool_get_array(meshPool, 'minLevelCell', minLevelCell)
       call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
       call mpas_pool_get_array(meshPool, 'bottomDepth', bottomDepth)
       call mpas_pool_get_array(meshPool, 'latCell', latCell)
       call mpas_pool_get_array(meshPool, 'latEdge', latEdge)
       call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
       call mpas_pool_get_array(meshPool, 'lonCell', lonCell)
       call mpas_pool_get_array(meshPool, 'refBottomDepth', refBottomDepth)
       call mpas_pool_get_array(meshPool, 'fCell', fCell)
       call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
       call mpas_pool_get_array(meshPool, 'fVertex', fVertex)
       call mpas_pool_get_array(meshPool, 'xCell', xCell)
       call mpas_pool_get_array(meshPool, 'yCell', yCell)


       call mpas_pool_get_array(verticalMeshPool, 'refZMid', refZMid)
       call mpas_pool_get_array(verticalMeshPool, 'restingThickness',restingThickness)

       call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocity, 1)
       call mpas_pool_get_array(tracersPool, 'activeTracers',activeTracers, 1)
       call mpas_pool_get_array(tracersPool, 'debugTracers',debugTracers, 1)
       call mpas_pool_get_array(statePool, 'layerThickness',layerThickness, 1)

       is_3D = is_test_3D(config_transport_tests_flow_id)

       ! These don't matter, but set them anyway.
       do k = 1, nVertLevels
          refBottomDepth(k) = (real(k,RKIND)/nVertLevels)*ztop
          refZMid(k) = -((k - 0.5_RKIND)/nVertLevels)*ztop
       end do

       do iCell = 1, nCellsSolve
          if (associated(activeTracers)) then
             activeTracers(index_temperature, :, iCell) = 15.0d0
             activeTracers(index_salinity, :, iCell) = 35.0d0
          endif

          vs = 1.0_RKIND
          if (associated(debugTracers)) then
             if (config_transport_tests_flow_id /= 4) then
                do k = 1, nVertLevels
                   debugTracers(1,k,iCell) = vs*xyztrig(lonCell(iCell), latCell(iCell))
                   debugTracers(2,k,iCell) = vs*cosine_bells(lonCell(iCell), latCell(iCell))
                   debugTracers(3,k,iCell) = vs*slotted_cylinders(lonCell(iCell), latCell(iCell))
                end do
             else
                ! Correlated Tracers test case
                do k = 1, nVertLevels
                   debugTracers(1,k,iCell) = vs*xyztrig(lonCell(iCell), latCell(iCell))
                   debugTracers(2,k,iCell) = vs*cosine_bells(lonCell(iCell), latCell(iCell))
                   debugTracers(3,k,iCell) = correlation_fn(debugTracers(2, k, iCell))
                enddo
             endif
             if (is_3D) then
                do k = 1, nVertLevels
                   shape = (1 - cos(2*pi*(k - 0.5)/nVertLevels))/2
                   do i = 1, 3
                      debugTracers(i,k,iCell) = shape*debugTracers(i,k,iCell)
                   end do
                end do
             end if
          endif

          do k = 1, nVertLevels
             layerThickness(k, iCell) = ztop/nVertLevels
             restingThickness(k, iCell) = layerThickness(k, iCell)
          end do

          fCell(iCell) = 0.0_RKIND ! 0 Coriolis
          bottomDepth(iCell) = ztop
          minLevelCell(iCell) = 1
          maxLevelCell(iCell) = nVertLevels
       end do

       do iEdge = 1, nEdgesSolve
          fEdge(iEdge) = 0.0_RKIND
       end do

       do iVertex=1, nVerticesSolve
          fVertex(iVertex) = 0.0_RKIND
       end do

       do iEdge = 1,nEdgesSolve
          call ocn_transport_test_velocity_edge(meshPool, 0.0_RKIND, 0.0_RKIND, &
               normalVelocity)
       enddo
       ! We don't need to initialize vertAleTransportTop because it's diagnostic.

       block_ptr => block_ptr % next
    end do
#ifdef MPAS_DEBUG
    call mpas_log_write("exiting subroutine ocn_transport_test_init_setup", MPAS_LOG_OUT, .true. , flushNow=flush_log)
#endif
  end subroutine ocn_transport_test_init_setup

!***********************************************************************
!
!  routine ocn_transport_test_init_validate
!
!> \brief Sets number of vertical levels.
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>
!
!-----------------------------------------------------------------------
  subroutine ocn_transport_test_init_validate(cfgpl, pkgpl, iocontext, err)
    type (mpas_pool_type), intent(inout) :: cfgpl, pkgpl
    type (mpas_io_context_type), intent(inout) :: iocontext
    integer, intent(out) :: err

    integer, pointer :: config_vert_levels, config_transport_tests_vert_levels

    err = 0

    call mpas_pool_get_config(cfgpl, 'config_vert_levels', config_vert_levels)
    call mpas_pool_get_config(cfgpl, 'config_transport_tests_vert_levels', &
         config_transport_tests_vert_levels)
    config_vert_levels = config_transport_tests_vert_levels
  end subroutine ocn_transport_test_init_validate

!***********************************************************************
!
!  routine ocn_transport_test_velocity_edge
!
!> \brief Sets edge normal velocities for prescribed flow fields
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details Sets edge normal velocities for prescribed flow fields.
!>
!
!-----------------------------------------------------------------------
  subroutine ocn_transport_test_velocity_edge(meshPool, days_since_start, dt, normalVelocity)
    type (mpas_pool_type), intent(in) :: meshPool
    real(RKIND), intent(in) :: days_since_start, dt
    real(RKIND), intent(out) :: normalVelocity(:,:)

    real(RKIND), dimension(:), pointer :: latEdge, lonEdge, latVertex, lonVertex
    integer, dimension(:,:), pointer :: verticesOnEdge
    real(RKIND), pointer :: sphere_radius
    real(RKIND) :: time_sec, nml(2), z
    integer, pointer :: nEdges
    integer :: i, k, nlev
    logical :: is_3D

    time_sec = days_since_start*one_day + dt
    call mpas_pool_get_config(meshPool, 'sphere_radius', sphere_radius)

    call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
    call mpas_pool_get_array(meshPool, 'latEdge', latEdge)
    call mpas_pool_get_array(meshPool, 'lonEdge', lonEdge)
    call mpas_pool_get_array(meshPool, 'latVertex', latVertex)
    call mpas_pool_get_array(meshPool, 'lonVertex', lonVertex)
    call mpas_pool_get_array(meshPool, 'verticesOnEdge', verticesOnEdge)

    is_3D = is_test_3D(config_transport_tests_flow_id)
    nlev = size(normalVelocity,1)
    z = 0.0_RKIND
    do i = 1, nEdges
       call calc_edge_normal( &
            lonEdge(i), latEdge(i), &
            lonVertex(verticesOnEdge(1,i)), latVertex(verticesOnEdge(1,i)), &
            lonVertex(verticesOnEdge(2,i)), latVertex(verticesOnEdge(2,i)), &
            nml)
       if (is_3D) then
          do k = 1, nlev
             z = ((k - 0.5)/nlev)*ztop
             normalVelocity(k,i) = calc_vnml( &
                  config_transport_tests_flow_id, time_sec, sphere_radius, &
                  z, lonEdge(i), latEdge(i), nml)
          end do
       else
          normalVelocity(:,i) = calc_vnml( &
               config_transport_tests_flow_id, time_sec, sphere_radius, &
               z, lonEdge(i), latEdge(i), nml)
       end if
    end do
  end subroutine ocn_transport_test_velocity_edge

!***********************************************************************
!
!  routine ocn_transport_test_velocity_vert
!
!> \brief Sets edge normal velocities for prescribed flow fields
!> \author Andrew Bradley
!> \date   4/21/2022
!> \details Sets vertical velocity at vertical interface cell centers.
!>
!
!-----------------------------------------------------------------------
  subroutine ocn_transport_test_velocity_vert(meshPool, days_since_start, dt, w)
    type (mpas_pool_type), intent(in) :: meshPool
    real(RKIND), intent(in) :: days_since_start, dt
    real(RKIND), intent(out) :: w(:,:)

    real(RKIND), dimension(:), pointer :: latCell, lonCell
    real(RKIND), pointer :: sphere_radius
    real(RKIND) :: time_sec, nml(2), z
    integer, pointer :: nCells
    integer :: i, k, nlev
    logical :: is_3D

    time_sec = days_since_start*one_day + dt
    call mpas_pool_get_config(meshPool, 'sphere_radius', sphere_radius)

    call mpas_pool_get_dimension(meshPool, 'nCells', nCells)
    call mpas_pool_get_array(meshPool, 'latCell', latCell)
    call mpas_pool_get_array(meshPool, 'lonCell', lonCell)

    is_3D = is_test_3D(config_transport_tests_flow_id)
    nlev = size(w,1) - 1
    z = 0.0_RKIND
    if (is_3D) then
       do i = 1, nCells
          do k = 1, nlev+1
             z = (real(k-1, RKIND)/nlev)*ztop
             ! Negative because vertical direction sign convention is the
             ! opposite of the 3D test's.
             w(k,i) = -calc_vvert( &
                  config_transport_tests_flow_id, time_sec, sphere_radius, &
                  z, lonCell(i), latCell(i))
          end do
       end do
    else
       w = 0
    end if
  end subroutine ocn_transport_test_velocity_vert

!--------------------------------------------------------------------
!
! Private module functions
!
!--------------------------------------------------------------------


  pure function flow_id_name(fid) result(tcname)
    integer, intent(in) :: fid
    character(len=32) :: tcname
    select case (fid)
      case (1) ; tcname = tc1str
      case (2) ; tcname = tc2str
      case (3) ; tcname = tc3str
      case (4) ; tcname = tc4str
      case (5) ; tcname = tc5str
    end select
  end function flow_id_name

  pure function is_test_3D(fid) result(is_3D)
    integer, intent(in) :: fid
    logical :: is_3D
    is_3D = fid == 5
  end function is_test_3D

!***********************************************************************
!
!  routine calc_vnml
!
!> \brief calculates edge normal velocity for prescribed flow
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>  This function computes the edge-normal velocity associated with
!> a prescribed flow advection test.
!
!-----------------------------------------------------------------------
  function calc_vnml(fid, t, radius, z, lon, lat, nml) result(vnml)
    integer, intent(in) :: fid
    real(RKIND), intent(in) :: t, radius, z, lon, lat, &
      nml(2) !< edge normal vector
    real(RKIND) :: vnml, u, v, w
    u = 0; v = 0
    select case(fid)
       case(1);   call flow_rotation      (   lon, lat, u, v)
       case(2,4); call flow_nondivergent  (t, lon, lat, u, v)
       case(3);   call flow_divergent     (t, lon, lat, u, v)
       case(5);   call flow_nondivergent3d(t, lon, lat, z, u, v, w)
    end select
    vnml = radius*(nml(1)*u + nml(2)*v)
  end function calc_vnml

!***********************************************************************
!
!  routine calc_vvert
!
!> \brief calculates edge normal velocity for prescribed flow
!> \author Andrew Bradley
!> \date   4/21/2022
!> \details
!>  This function computes the vertical velocity.
!
!-----------------------------------------------------------------------
  function calc_vvert(fid, t, radius, z, lon, lat) result(w)
    integer, intent(in) :: fid
    real(RKIND), intent(in) :: t, radius, z, lon, lat
    real(RKIND) :: u, v, w
    w = 0
    select case(fid)
    case(5); call flow_nondivergent3d(t, lon, lat, z, u, v, w)
    end select
  end function calc_vvert

!--------------------------------------------------------------------
!
! Prescribed velocities
!
!***********************************************************************
!
!  routine flow_rotation
!
!> \brief solid-body rotation velocity
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>  This function computes the velocity [u(lon,lat,t), v(lon,lat,t)]
!> for a solid-body rotation flow field.
!
!-----------------------------------------------------------------------
  pure subroutine flow_rotation(lon, lat, u, v)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND), intent(out) :: u, v
    real(RKIND) :: Omega(3), r(3), vel(3), east(3), north(3)
    Omega = (/0.2_RKIND, 0.7_RKIND, 1.0_RKIND/) !< Constant rotation vector
    Omega = (two_pi/twelve_days)*(Omega/norm(Omega))
    call lonlat2xyz(lon, lat, r(1), r(2), r(3))
    call cross(Omega, r, vel)
    call calc_local_east_north(r, east, north)
    u = sum(vel*east)
    v = sum(vel*north)
  end subroutine flow_rotation

!***********************************************************************
!
!  routine flow_nondivergent
!
!> \brief non-divergent deformational flow velocity
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>  This function computes the velocity [u(lon,lat,t), v(lon,lat,t)]
!> for a non-divergent flow field defined in Lauritzen et al. 2012 section 3
!> equations (16) -- (20)
!
!-----------------------------------------------------------------------
  pure subroutine flow_nondivergent(t, lon, lat, u, v)
    real(RKIND), intent(in) :: t, lon, lat
    real(RKIND), intent(out) :: u, v
    real(RKIND) :: lon_p, coslat, cost
    lon_p = lon - two_pi*t/twelve_days
    coslat = cos(lat)
    cost = cos(pi*t/twelve_days)
    u = (1/twelve_days)*(10*(sin(lon_p)**2)*sin(2*lat)*cost + two_pi*coslat)
    v = (10/twelve_days)*sin(2*lon_p)*coslat*cost
  end subroutine flow_nondivergent

!***********************************************************************
!
!  routine flow_divergent
!
!> \brief divergent deformational flow velocity
!> \author Andrew Bradley
!> \date   12/2/2020
!> \details
!>  This function computes the velocity [u(lon,lat,t), v(lon,lat,t)]
!> for a divergent flow field defined in Lauritzen et al. 2012 section 3
!> equations (21) -- (22)
!
!-----------------------------------------------------------------------
  pure subroutine flow_divergent(t, lon, lat, u, v)
    real(RKIND), intent(in) :: t, lon, lat
    real(RKIND), intent(out) :: u, v
    real(RKIND) :: lon_p, coslat, cost
    lon_p = lon - two_pi*t/twelve_days
    coslat = cos(lat)
    cost = cos(pi*t/twelve_days)
    u = (1/twelve_days)*(-5*(sin(lon_p/2)**2)*sin(2*lat)*(coslat**2)*cost + two_pi*coslat)
    v = (2.5_RKIND/twelve_days)*sin(lon_p)*(coslat**3)*cost
  end subroutine flow_divergent

!***********************************************************************
!
!  routine flow_nondivergent3D
!
!> \brief non-divergent deformational flow velocity: 3D variant
!> \author Andrew Bradley
!> \date   3/21/2022
!> \details
!>  This function computes the velocity [u(lon,lat,t), v(lon,lat,t),
!> w(lon,lat,t)] for a non-divergent flow field defined in DCMIP 2012 Test 1-1,
!> with two exceptions. First, the factor 2 appearing in cos(2 pi t/T) is
!> removed, as it leads to nonperiodic trajectories. Second, the shape
!> parameter b is 1 rather than 0.1 to make the test easier and thus useful at
!> coarser resolutions.
!
!-----------------------------------------------------------------------
  pure subroutine flow_nondivergent3D(t, lon, lat, z, u, v, w)
    real(RKIND), intent(in) :: t, lon, lat, z
    real(RKIND), intent(out) :: u, v, w
    real(RKIND), parameter :: b = 1.0_RKIND, omega0 = (2*23000.0_RKIND*pi)/twelve_days
    real(RKIND) :: lon_p, coslat, cost, sinlonp, ptop, p, s, s_p, ud, rho
    lon_p = lon - two_pi*t/twelve_days
    coslat = cos(lat)
    cost = cos(pi*t/twelve_days)
    sinlonp = sin(lon_p)
    ptop = calc_z2p(ztop)
    p = calc_z2p(z)
    s = (1 + exp((ptop-p0)/(b*ptop)) - exp((p-p0)/(b*ptop)) - exp((ptop-p)/(b*ptop)))
    s_p = (-exp((p-p0)/(b*ptop)) + exp((ptop-p)/(b*ptop)))/(b*ptop)
    ud = omega0*cos(lon_p)*coslat**2*cost*s_p
    u = (1/twelve_days)*(10*(sin(lon_p)**2)*sin(2*lat)*cost + two_pi*coslat) + ud
    v = (10/twelve_days)*sin(2*lon_p)*coslat*cost
    rho = p/(Rd*T0)
    w = (-omega0/(grav*get_rho(p)))*sinlonp*coslat*cost*s
  end subroutine flow_nondivergent3D

  ! 3D test helpers.
  pure function get_rho(p) result (rho)
    real(RKIND), intent(in) :: p
    real(RKIND) :: rho
    rho = p/(Rd*T0)
  end function get_rho

  pure function calc_z2p(z) result (p)
    real(RKIND), intent(in) :: z
    real(RKIND) :: p
    p = p0*exp(-z/H)
  end function calc_z2p

!
! end prescribed velocities
!
!--------------------------------------------------------------------

!--------------------------------------------------------------------
! Tracer initial conditions
!

  ! Lauritzen et al. 2012 does not have a global C-infty tracer; we add this one.
  pure function xyztrig(lon, lat) result (f)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND) :: f, x, y, z
    call lonlat2xyz(lon, lat, x, y, z)
    f = 0.5_RKIND*(1 + sin(pi*x)*sin(pi*y)*sin(pi*z))
  end function xyztrig

  ! Lauritzen et al. 2012 eqn. (11)
  pure function cosine_bells(lon, lat) result (cbs)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND), parameter :: r = 0.5_RKIND, b = 0.1_RKIND, c = 0.9_RKIND, &
         lon1 = 5*(pi/6), lat1 = 0, lon2 = -5*(pi/6), lat2 = 0
    real(RKIND) :: cbs, x, y, z, x1, y1, z1, x2, y2, z2, r1, r2
    call lonlat2xyz(lon, lat, x, y, z)
    call lonlat2xyz(lon1, lat1, x1, y1, z1)
    r1 = great_circle_dist_xyz(x, y, z, x1, y1, z1)
    cbs = 0
    if (r1 < r) then
       cbs = cosine_bell(r1, r)
    else
       call lonlat2xyz(lon2, lat2, x2, y2, z2)
       r2 = great_circle_dist_xyz(x, y, z, x2, y2, z2)
       if (r2 < r) cbs = cosine_bell(r2, r)
    end if
    cbs = b + c*cbs
  end function cosine_bells

  ! Lauritzen et al. 2012 eqn. (10)
  pure function cosine_bell(ri, r) result(cb)
    real(RKIND), intent(in) :: ri, r
    real(RKIND) :: cb
    real(RKIND), parameter :: hmax = 1._RKIND
    cb = 0.5_RKIND*hmax*(1 + cos(pi*ri/r))
  end function cosine_bell

  ! Lauritzen et al. 2012 eqns. (14) and (15)
  pure function correlation_fn(q1) result(q2)
    real(RKIND), intent(in) :: q1
    real(RKIND) :: q2
    real(RKIND), parameter :: apsi = -0.8_RKIND, bpsi = 0.9_RKIND
    q2 = apsi * q1 * q1 + bpsi
  end function correlation_fn

  ! Lauritzen et al. 2012 eqn. (12)
  pure function slotted_cylinders(lon, lat) result (scs)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND), parameter :: b = 0.1_RKIND, c = 1._RKIND, R = 1._RKIND, Rh = R/2, &
         lon_thr = Rh/(6*R), lat_thr = 5*(Rh/(12*R)), &
         lon1 = 5*(pi/6), lat1 = 0, lon2 = -5*(pi/6), lat2 = 0
    real(RKIND) :: scs, x, y, z, x1, y1, z1, x2, y2, z2, r1, r2, lon0
    call lonlat2xyz(lon, lat, x, y, z)
    lon0 = lon
    if (lon0 > pi) lon0 = lon0 - two_pi
    call lonlat2xyz(lon1, lat1, x1, y1, z1)
    r1 = great_circle_dist_xyz(x, y, z, x1, y1, z1)
    scs = b
    if (r1 <= Rh) then
       if ( (abs(lon0 - lon1) >= lon_thr) .or. &
            (abs(lon0 - lon1) <  lon_thr .and. lat - lat1 < -lat_thr)) then
          scs = c
       end if
    else
       call lonlat2xyz(lon2, lat2, x2, y2, z2)
       r2 = great_circle_dist_xyz(x, y, z, x2, y2, z2)
       if (r2 <= Rh .and. &
           ((abs(lon0 - lon2) >= lon_thr) .or. &
            (abs(lon0 - lon2) <  lon_thr .and. lat - lat2 >  lat_thr))) then
          scs = c
       end if
    end if
  end function slotted_cylinders

  ! Lauritzen et al. 2012 eqn. (9)
  pure function gaussian_hills(lon, lat) result (ghs)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND), parameter :: &
         lon1 = 5*(pi/6), lat1 = 0, lon2 = -5*(pi/6), lat2 = 0
    real(RKIND) :: ghs, x, y, z, x1, y1, z1, x2, y2, z2
    call lonlat2xyz(lon, lat, x, y, z)
    call lonlat2xyz(lon1, lat1, x1, y1, z1)
    call lonlat2xyz(lon2, lat2, x2, y2, z2)
    ghs = gaussian_hill(x, y, z, x1, y1, z1) + gaussian_hill(x, y, z, x2, y2, z2)
  end function gaussian_hills

  ! Lauritzen et al. 2012 eqn. (7)
  pure function gaussian_hill(x, y, z, xi, yi, zi) result(gh)
    real(RKIND), intent(in) :: x, y, z, xi, yi, zi
    real(RKIND) :: gh
    real(RKIND), parameter :: hmax = 0.95_RKIND, b = 5.0_RKIND
    gh = hmax*exp(-b*((x - xi)**2 + (y - yi)**2 + (z - zi)**2))
  end function gaussian_hill

!
! end tracer initial conditions

!--------------------------------------------------------------------
! Utilities
!

  subroutine calc_edge_normal(lonc, latc, lon1, lat1, lon2, lat2, nml_en)
    real(RKIND), intent(in) :: lonc, latc, lon1, lat1, lon2, lat2
    real(RKIND), intent(out) :: nml_en(2)
    real(RKIND) :: ec(3), v1(3), v2(3), d(3), nml(3), east(3), north(3)
    call lonlat2xyz(lonc, latc, ec(1), ec(2), ec(3))
    call lonlat2xyz(lon1, lat1, v1(1), v1(2), v1(3))
    call lonlat2xyz(lon2, lat2, v2(1), v2(2), v2(3))
    d = v2 - v1
    call cross(d, ec, nml)
    nml = nml/norm(nml)
    call calc_local_east_north(ec, east, north)
    nml_en(1) = sum(nml*east)
    nml_en(2) = sum(nml*north)
    nml_en = nml_en/sqrt(sum(nml_en*nml_en))
  end subroutine calc_edge_normal


  pure function norm(x) result(nx)
    real(RKIND), intent(in) :: x(3)
    real(RKIND) :: nx
    nx = sqrt(sum(x*x))
  end function norm

  pure subroutine cross(a, b, c)
    real(RKIND), intent(in) :: a(3), b(3)
    real(RKIND), intent(out) :: c(3)
    c(1) = a(2)*b(3) - a(3)*b(2);
    c(2) = a(3)*b(1) - a(1)*b(3);
    c(3) = a(1)*b(2) - a(2)*b(1);
  end subroutine cross

  pure subroutine calc_local_east_north(r, east, north)
    real(RKIND), intent(in) :: r(3)
    real(RKIND), intent(out) :: east(3), north(3)
    real(RKIND), parameter :: axis(3) = (/0,0,1/)
    call cross(axis, r, east)
    east = east/norm(east)
    call cross(r, east, north)
    north = north/norm(north)
  end subroutine calc_local_east_north

  pure subroutine lonlat2xyz(lon, lat, x, y, z)
    real(RKIND), intent(in) :: lon, lat
    real(RKIND), intent(out) :: x, y, z
    real(RKIND) :: sinl, cosl
    sinl = sin(lat)
    cosl = cos(lat)
    x = cos(lon)*cosl
    y = sin(lon)*cosl
    z = sinl
  end subroutine lonlat2xyz

  pure subroutine xyz2lonlat(x, y, z, lon, lat)
    real(RKIND), intent(in) :: x, y, z
    real(RKIND), intent(out) :: lon, lat
    real(RKIND) :: r
    lon = atan2(y, x);
    r = sqrt(x*x + y*y + z*z)
    lat = asin(z/r);
  end subroutine xyz2lonlat

  ! TODO: Does this fn need radius, or is it always just used for arc length?
  pure function great_circle_dist_xyz(xA, yA, zA, xB, yB, zB) result(dist)
    real(RKIND), intent(in) :: xA, yA, zA, xB, yB, zB
    real(RKIND) :: dist, cp1, cp2, cp3, cpnorm, dotprod
    cp1 = yA*zB - yB*zA
    cp2 = xB*zA - xA*zB
    cp3 = xA*yB - xB*yA
    cpnorm = sqrt(cp1*cp1 + cp2*cp2 + cp3*cp3)
    dotprod = xA*xB + yA*yB + zA*zB
    dist = atan2(cpnorm, dotprod)
  end function great_circle_dist_xyz

  pure function great_circle_dist_ll(lon1, lat1, lon2, lat2, radius) result(dist)
    real(RKIND), intent(in) :: lon1, lat1, lon2, lat2, radius
    real(RKIND) :: dist, xA, yA, zA, xB, yB, zB
    call lonlat2xyz(lon1, lat1, xA, yA, zA)
    call lonlat2xyz(lon2, lat2, xB, yB, zB)
    dist = radius*great_circle_dist_xyz(xA, yA, zA, xB, yB, zB)
  end function great_circle_dist_ll

!
! end utilities
!--------------------------------------------------------------------

  !amb debug stuff
  subroutine amb_init(comm)
    integer, intent(in), optional :: comm

    integer :: err
    character (len=255) :: env_str
    integer :: tmp

    if (.not. first) return
    first = .false.

    if (present(comm)) then
       comm_global = comm
    else
       comm_global = MPI_COMM_WORLD
    end if

    call mpi_comm_rank(comm_global, mpirank, err)
    amroot = mpirank == 0

    verbosity = 10
    call get_environment_variable("amb_verbosity", env_str, status=err)
    if (err /= 1) then
       read(env_str, '(i)', iostat=err) tmp
       if (tmp) verbosity = tmp
    end if
    if (amroot .and. verbosity > 0) then
       print *,'amb>_init test id', config_transport_tests_flow_id
    end if
  end subroutine amb_init
  function amb_amroot() result(ar)
    logical :: ar
    call amb_init()
    ar = amroot
  end function amb_amroot
  function amb_get_rank() result(r)
    integer :: r
    call amb_init()
    r = mpirank
  end function amb_get_rank
  function amb_get_verbosity() result(r)
    integer :: r
    call amb_init()
    r = verbosity
  end function amb_get_verbosity
end module ocn_transport_tests
