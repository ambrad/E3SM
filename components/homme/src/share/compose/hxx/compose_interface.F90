module compose_interface
  use kinds, only: real_kind
  use iso_c_binding, only: c_int
  use dimensions_mod, only: nlev, nlevp, np, nelemd, ne, qsize
  use geometry_interface_mod, only: par, elem
  implicit none

contains

  subroutine init_compose_f90(ne, hyai, hybi, hyam, hybm, ps0) bind(c)
    use hybvcoord_mod, only: set_layer_locations
    use thetal_test_interface, only: init_f90
    use edge_mod_base, only: initEdgeBuffer, edge_g
    use control_mod, only: transport_alg, semi_lagrange_cdr_alg, semi_lagrange_cdr_check, &
         semi_lagrange_hv_q, cubed_sphere_map
    use geometry_interface_mod, only: GridVertex
    use bndry_mod, only: sort_neighbor_buffer_mapping
    use reduction_mod, only: initreductionbuffer, red_sum, red_min, red_max
    use parallel_mod, only: global_shared_buf, nrepro_vars
    use compose_mod, only: compose_init, cedr_set_ie2gci, compose_set_null_bufs
    use sl_advection, only: sl_init1

    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    integer (kind=c_int), value, intent(in) :: ne
    real (kind=real_kind), value, intent(in) :: ps0

    integer :: ie
    real (kind=real_kind) :: mp(np,np), dvv(np,np)

    transport_alg = 12
    semi_lagrange_cdr_alg = 42
    semi_lagrange_cdr_check = semi_lagrange_cdr_alg /= 42
    semi_lagrange_hv_q = 0
    cubed_sphere_map = 2
    qsize = 4

    call init_f90(ne, hyai, hybi, hyam, hybm, dvv, mp, ps0)

    call initEdgeBuffer(par, edge_g, elem, 6*nlev+1)

    call initReductionBuffer(red_sum,5)
    call initReductionBuffer(red_min,1)
    call initReductionBuffer(red_max,1)
    allocate(global_shared_buf(nelemd, nrepro_vars))

    call sort_neighbor_buffer_mapping(par, elem, 1, nelemd)
    call compose_init(par, elem, GridVertex, init_kokkos=.false.)
    call compose_set_null_bufs()
    do ie = 1, nelemd
       call cedr_set_ie2gci(ie, elem(ie)%vertex%number)
    end do
    call sl_init1(par, elem)
  end subroutine init_compose_f90

  subroutine cleanup_compose_f90() bind(c)
    use compose_mod, only: compose_finalize
    use thetal_test_interface, only: cleanup_f90

    call compose_finalize(finalize_kokkos=.false.)
    call cleanup_f90()
  end subroutine cleanup_compose_f90

  subroutine run_compose_standalone_test_f90() bind(c)
    use thetal_test_interface, only: deriv, hvcoord
    use compose_test_mod, only: compose_test
    use domain_mod, only: domain1d_t
    use control_mod, only: transport_alg, statefreq
    use time_mod, only: nmax
    use thread_mod, only: hthreads, vthreads

    type (domain1d_t), pointer :: dom_mt(:)

    hthreads = 1
    vthreads = 1
    allocate(dom_mt(0:0))
    dom_mt(0)%start = 1
    dom_mt(0)%end = nelemd
    transport_alg = 19
    nmax = 6*ne
    statefreq = 2*ne
    call compose_test(par, hvcoord, dom_mt, elem)
    transport_alg = 12
    deallocate(dom_mt)
  end subroutine run_compose_standalone_test_f90
  
end module compose_interface
