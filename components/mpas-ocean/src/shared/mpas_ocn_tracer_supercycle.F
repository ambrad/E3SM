module ocn_tracer_supercycle
  use ocn_config
  use mpi

  implicit none
  private
  save

  ! settings
  integer :: supercycle
  logical :: sc_use
  ! initialization
  logical :: inited = .false.
  ! state
  integer :: timestep, split_explicit_step, n_ts_iter

  public :: &
       ocn_tracer_supercycle_use, &
       ocn_tracer_supercycle_set_timestep, &
       ocn_tracer_supercycle_set_split_explicit_step, &
       ocn_tracer_supercycle_get_timestep, &
       ocn_tracer_supercycle_get_supercycle, &
       ocn_tracer_supercycle_is_tracer_step, &
       ocn_tracer_supercycle_set_layerThickness0, &
       ocn_tracer_supercycle_accum, &
       ocn_tracer_supercycle_fill, &
       ocn_tracer_supercycle_update

contains

  function ocn_tracer_supercycle_use() result(u)
    logical :: u
    u = sc_use
  end function ocn_tracer_supercycle_use

  subroutine ocn_tracer_supercycle_set_timestep()
    call ocn_tracer_supercycle_init()
    !todo mod this with 'supercycle'
    timestep = timestep + 1
  end subroutine ocn_tracer_supercycle_set_timestep

  subroutine ocn_tracer_supercycle_set_split_explicit_step( &
       split_explicit_step_in, n_ts_iter_in)
    integer, intent(in) :: split_explicit_step_in, n_ts_iter_in
    split_explicit_step = split_explicit_step_in
    n_ts_iter = n_ts_iter_in
  end subroutine ocn_tracer_supercycle_set_split_explicit_step

  function ocn_tracer_supercycle_get_timestep() result(ts)
    integer :: ts
    ts = timestep
  end function ocn_tracer_supercycle_get_timestep

  function ocn_tracer_supercycle_get_supercycle() result(sc)
    integer :: sc
    sc = supercycle
  end function ocn_tracer_supercycle_get_supercycle

  function ocn_tracer_supercycle_is_tracer_step() result(b)
    logical :: b
    b = modulo(timestep, supercycle) == 0
  end function ocn_tracer_supercycle_is_tracer_step

  subroutine ocn_tracer_supercycle_init()
    integer :: err, mpirank
    logical :: amroot

    if (inited) return
    inited = .true.

    timestep = 0

    call mpi_comm_rank(MPI_COMM_WORLD, mpirank, err)
    amroot = mpirank == 0

    sc_use = .false.
    supercycle = config_passive_tracers_supercycle_timestep
    sc_use = supercycle > 1 .and. &
         (    trim(config_time_integrator) == 'split_explicit'   &
         .or. trim(config_time_integrator) == 'unsplit_explicit')
    
    if (amroot) then
       print '(a,l,a,i2,a,l)', 'amb> sc use', sc_use, &
            ' sc', supercycle, ' subcyc', config_ecosysTracers_subcycle_param
    end if
  end subroutine ocn_tracer_supercycle_init

  subroutine ocn_tracer_supercycle_set_layerThickness0(tendPool, nCells, maxLevelCell, &
       layerThickness)
    type (mpas_pool_type), intent(inout) :: tendPool
    integer, intent(in) :: nCells, maxLevelCell(:)
    real (kind=RKIND), intent(in) :: layerThickness(:,:)

    real (kind=RKIND), pointer :: layerThickness0(:,:)
    integer :: ic, k

    if (split_explicit_step /= n_ts_iter) return

    if (modulo(timestep-1, supercycle) == 0) then
       call mpas_pool_get_array(tendPool, 'scLayerThickness0', layerThickness0)
       !$omp parallel
       !$omp parallel do private(k)
       do ic = 1, nCells
          do k = 1, maxLevelCell(ic)
             layerThickness0(k,ic) = layerThickness(k,ic)
          end do
       end do
       !$omp end do
       !$omp end parallel
    end if
  end subroutine ocn_tracer_supercycle_set_layerThickness0

  subroutine ocn_tracer_supercycle_accum(tendPool, nCells, maxLevelCell, nEdges, nVertLevels, &
       layerThickness, layerThicknessEdge, zMid, normalThicknessFlux, vertAleTransportTop)
    type (mpas_pool_type), intent(inout) :: tendPool
    integer, intent(in) :: nCells, maxLevelCell(:), nEdges, nVertLevels
    real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness, layerThicknessEdge, zMid, normalThicknessFlux, vertAleTransportTop
    real (kind=RKIND), dimension(:,:), pointer :: test, t1

    real (kind=RKIND), dimension(:,:), pointer :: accumLayerThickness, &
         accumLayerThicknessEdge, accumZMid, accumNormalThicknessFlux, &
         accumVertAleTransportTop
    integer :: ic, ie, k

    if (split_explicit_step /= n_ts_iter) return

    call mpas_pool_get_array(tendPool, 'scAccumLayerThickness', accumLayerThickness)
    call mpas_pool_get_array(tendPool, 'scAccumLayerThicknessEdge', accumLayerThicknessEdge)
    call mpas_pool_get_array(tendPool, 'scAccumZMid', accumZMid)
    call mpas_pool_get_array(tendPool, 'scAccumNormalThicknessFlux', accumNormalThicknessFlux)
    call mpas_pool_get_array(tendPool, 'scAccumVertAleTransportTop', accumVertAleTransportTop)

    if (modulo(timestep-1, supercycle) == 0) then
       !$omp parallel
       !$omp do private(k)
       do ic = 1, nCells
          do k = 1, maxLevelCell(ic)
             accumLayerThickness(k,ic) = 0
             accumZMid(k,ic) = 0
          end do
          do k = 1, nVertLevels+1
             accumVertAleTransportTop(k,ic) = 0
          end do
       end do
       !$omp end do
       !$omp do private(k)
       do ie = 1, nEdges
          do k = 1, nVertLevels
             accumLayerThicknessEdge(k,ie) = 0
             accumNormalThicknessFlux(k,ie) = 0
          end do
       end do
       !$omp end do
       !$omp end parallel
    end if

    !$omp parallel
    !$omp do private(k)
    do ic = 1, nCells
       do k = 1, maxLevelCell(ic)
          accumLayerThickness(k,ic) = accumLayerThickness(k,ic) + layerThickness(k,ic)
          accumZMid(k,ic) = accumZMid(k,ic) + zMid(k,ic)
       end do
       do k = 1, nVertLevels+1
          accumVertAleTransportTop(k,ic) = accumVertAleTransportTop(k,ic) + &
               vertAleTransportTop(k,ic)
       end do
    end do
    !$omp end do
    !$omp do private(k)
    do ie = 1, nEdges
       do k = 1, nVertLevels
          accumLayerThicknessEdge(k,ie) = accumLayerThicknessEdge(k,ie) + &
               layerThicknessEdge(k,ie)
          accumNormalThicknessFlux(k,ie) = accumNormalThicknessFlux(k,ie) + &
               normalThicknessFlux(k,ie)
       end do
    end do
    !$omp end do
    !$omp end parallel
    
    if (ocn_tracer_supercycle_is_tracer_step()) then
       !$omp parallel
       !$omp do private(k)
       do ic = 1, nCells
          do k = 1, maxLevelCell(ic)
             accumLayerThickness(k,ic) = accumLayerThickness(k,ic)/supercycle
             accumZMid(k,ic) = accumZMid(k,ic)/supercycle
          end do
          do k = 1, nVertLevels+1
             accumVertAleTransportTop(k,ic) = accumVertAleTransportTop(k,ic)/supercycle
          end do
       end do
       !$omp end do
       !$omp do private(k)
       do ie = 1, nEdges
          do k = 1, nVertLevels
             accumLayerThicknessEdge(k,ie) = accumLayerThicknessEdge(k,ie)/supercycle
             accumNormalThicknessFlux(k,ie) = accumNormalThicknessFlux(k,ie)/supercycle
          end do
       end do
       !$omp end do
       !$omp end parallel
    end if
  end subroutine ocn_tracer_supercycle_accum

  subroutine ocn_tracer_supercycle_fill(tendPool, nCells, maxLevelCell, nEdges, nVertLevels, &
       layerThickness, layerThicknessEdge, zMid, normalThicknessFlux, vertAleTransportTop)
    type (mpas_pool_type), intent(inout) :: tendPool
    integer, intent(in) :: nCells, maxLevelCell(:), nEdges, nVertLevels
    real (kind=RKIND), intent(out) :: normalThicknessFlux(:,:)
    real (kind=RKIND), intent(out), dimension(:,:), pointer :: layerThickness, &
         layerThicknessEdge, zMid, vertAleTransportTop

    real (kind=RKIND), dimension(:,:), pointer :: accumNormalThicknessFlux
    integer :: ie, k

    call mpas_pool_get_array(tendPool, 'scAccumNormalThicknessFlux', accumNormalThicknessFlux)
    call mpas_pool_get_array(tendPool, 'scAccumLayerThickness', layerThickness)
    call mpas_pool_get_array(tendPool, 'scAccumLayerThicknessEdge', layerThicknessEdge)
    call mpas_pool_get_array(tendPool, 'scAccumZMid', zMid)
    call mpas_pool_get_array(tendPool, 'scAccumVertAleTransportTop', vertAleTransportTop)
    !$omp end do
    !$omp do private(k)
    do ie = 1, nEdges
       do k = 1, nVertLevels
          normalThicknessFlux(k,ie) = accumNormalThicknessFlux(k,ie)
       end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine ocn_tracer_supercycle_fill

  subroutine ocn_tracer_supercycle_update(tendPool, nCells, maxLevelCell, &
       dt, tcur, ttend, lyrnew, tnew)
    type (mpas_pool_type), intent(inout) :: tendPool
    integer, intent(in) :: nCells, maxLevelCell(:)
    real (kind=RKIND), intent(in) :: dt, tcur(:,:,:), ttend(:,:,:), lyrnew(:,:)
    real (kind=RKIND), intent(out) :: tnew(:,:,:)

    real (kind=RKIND), pointer :: layerThickness0(:,:)
    integer :: ic, k

    if (modulo(timestep, supercycle) /= 0) then
       !$omp parallel
       !$omp do private(k)
       do ic = 1, nCells
          do k = 1, maxLevelCell(ic)
             tnew(:,k,ic) = tcur(:,k,ic)
          end do
       end do
       !$omp end do
       !$omp end parallel
       return
    end if
    call mpas_pool_get_array(tendPool, 'scLayerThickness0', layerThickness0)
    !$omp parallel
    !$omp do private(k)
    do ic = 1, nCells
       do k = 1, maxLevelCell(ic)
          tnew(:,k,ic) = (tcur(:,k,ic)*layerThickness0(k,ic) + &
                          supercycle*dt*ttend(:,k,ic)) / &
                         lyrnew(k,ic)
       end do
    end do
    !$omp end do
    !$omp end parallel
    ! Reset time step counter to start of tracer time step.
    timestep = 0
  end subroutine ocn_tracer_supercycle_update

end module ocn_tracer_supercycle
