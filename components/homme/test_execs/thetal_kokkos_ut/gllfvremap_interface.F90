module physgrid_interface
  use kinds, only: real_kind
  use iso_c_binding, only: c_bool, c_int, c_double, c_ptr, c_loc
  use dimensions_mod, only: nlev, nlevp, np, nelemd, ne, qsize, qsize_d
  use geometry_interface_mod, only: par, elem
  implicit none
  
contains

  subroutine init_gllfvremap_f90(ne, hyai, hybi, hyam, hybm, ps0, dvv, mp, qsize_in, &
       is_sphere) bind(c)
    use hybvcoord_mod, only: set_layer_locations
    use thetal_test_interface, only: init_f90
    use theta_f2c_mod, only: init_elements_c
    use edge_mod_base, only: initEdgeBuffer, edge_g, initEdgeSBuffer
    use prim_advection_base, only: edgeAdvQminmax
    use geometry_interface_mod, only: GridVertex
    use bndry_mod, only: sort_neighbor_buffer_mapping
    use reduction_mod, only: initreductionbuffer, red_sum, red_min, red_max
    use parallel_mod, only: global_shared_buf, nrepro_vars
    use compose_mod, only: compose_init, cedr_set_ie2gci, compose_set_null_bufs
    use sl_advection, only: sl_init1

    real (real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    integer (c_int), value, intent(in) :: ne, qsize_in
    real (real_kind), value, intent(in) :: ps0
    real (real_kind), intent(out) :: dvv(np,np), mp(np,np)
    logical (c_bool), value, intent(in) :: is_sphere

    integer :: ie, edgesz

    if (.not. is_sphere) print *, "NOT IMPL'ED YET"

    qsize = qsize_in

    call init_f90(ne, hyai, hybi, hyam, hybm, dvv, mp, ps0)
    call init_elements_c(nelemd)

    edgesz = max((qsize+3)*nlev+2,6*nlev+1)
    call initEdgeBuffer(par, edge_g, elem, edgesz)
    call initEdgeSBuffer(par, edgeAdvQminmax, elem, qsize*nlev*2)

    call initReductionBuffer(red_sum,5)
    call initReductionBuffer(red_min,1)
    call initReductionBuffer(red_max,1)
    allocate(global_shared_buf(nelemd, nrepro_vars))
  end subroutine init_gllfvremap_f90

  subroutine run_gfr_test(nerr) bind(c)
    use thetal_test_interface, only: deriv, hvcoord
    use domain_mod, only: domain1d_t
    use hybrid_mod, only: hybrid_t, hybrid_create
    use gllfvremap_mod, only: gfr_test

    integer (c_int), intent(out) :: nerr

    integer :: ithr
    type (hybrid_t) :: hybrid
    type (domain1d_t) :: dom_mt(0:0)

    dom_mt(0)%start = 1
    dom_mt(0)%end = nelemd
    hybrid = hybrid_create(par, 0, 1)
    
    nerr = gfr_test(hybrid, dom_mt, hvcoord, deriv, elem)
  end subroutine run_gfr_test

  subroutine run_gfr_check_api(nerr) bind(c)
    use thetal_test_interface, only: hvcoord
    use hybrid_mod, only: hybrid_t, hybrid_create
    use gllfvremap_util_mod, only: gfr_check_api

    integer (c_int), intent(out) :: nerr

    integer :: ithr
    type (hybrid_t) :: hybrid

    hybrid = hybrid_create(par, 0, 1)    
    nerr = gfr_check_api(hybrid, 1, nelemd, hvcoord, elem)
  end subroutine run_gfr_check_api

  subroutine limiter1_clip_and_sum_f90(n, spheremp, qmin, qmax, dp, q) bind(c)
    use gllfvremap_mod, only: limiter1_clip_and_sum

    integer (c_int), value, intent(in) :: n
    real (c_double), intent(in) :: spheremp(n*n), dp(n*n)
    real (c_double), intent(inout) :: qmin, qmax, q(n*n)

    call limiter1_clip_and_sum(n, spheremp, qmin, qmax, dp, q)
  end subroutine limiter1_clip_and_sum_f90

  subroutine calc_dp_fv_f90(nf, ps, dp_fv) bind(c)
    use thetal_test_interface, only: hvcoord
    use gllfvremap_mod, only: calc_dp_fv

    integer (c_int), value, intent(in) :: nf
    real (c_double), intent(in) :: ps(nf*nf)
    real (c_double), intent(out) :: dp_fv(nf*nf,nlev)

    call calc_dp_fv(nf, hvcoord, ps, dp_fv)
  end subroutine calc_dp_fv_f90
end module physgrid_interface
