module compose_interface
  use kinds, only: real_kind
  use iso_c_binding, only: c_int
  use dimensions_mod, only: nlev, nlevp, np, nelemd
  use geometry_interface_mod, only: par, elem
  implicit none

contains

  subroutine init_compose_f90(ne, hyai, hybi, hyam, hybm, ps0) bind(c)
    use hybvcoord_mod, only: set_layer_locations
    use thetal_test_interface, only: init_f90
    use edge_mod_base, only: initEdgeBuffer, edge_g
    use control_mod, only: transport_alg
    use geometry_interface_mod, only: GridVertex
    use bndry_mod, only: sort_neighbor_buffer_mapping
    use compose_mod, only: compose_init, cedr_set_ie2gci
    use sl_advection, only: sl_init1

    real (kind=real_kind), intent(in) :: hyai(nlevp), hybi(nlevp), hyam(nlev), hybm(nlev)
    integer (kind=c_int), value, intent(in) :: ne
    real (kind=real_kind), value, intent(in) :: ps0

    integer :: ie
    real (kind=real_kind) :: mp(np,np), dvv(np,np)

    transport_alg = 12
    call init_f90(ne, hyai, hybi, hyam, hybm, dvv, mp, ps0)
    call initEdgeBuffer(par, edge_g, elem, 6*nlev+1)
    call sort_neighbor_buffer_mapping(par, elem, 1, nelemd)
    call compose_init(par, elem, GridVertex, init_kokkos=.false.)
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

end module compose_interface
