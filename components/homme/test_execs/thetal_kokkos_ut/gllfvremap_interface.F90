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
    use edge_mod_base, only: initEdgeBuffer, edge_g
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

    integer :: ie

    if (is_sphere) print *, "NOT IMPL'ED YET"

    qsize = qsize_in

    call init_f90(ne, hyai, hybi, hyam, hybm, dvv, mp, ps0)
    call init_elements_c(nelemd)

    call initEdgeBuffer(par, edge_g, elem, 6*nlev+1)

    call initReductionBuffer(red_sum,5)
    call initReductionBuffer(red_min,1)
    call initReductionBuffer(red_max,1)
    allocate(global_shared_buf(nelemd, nrepro_vars))
  end subroutine init_gllfvremap_f90
  
end module physgrid_interface
