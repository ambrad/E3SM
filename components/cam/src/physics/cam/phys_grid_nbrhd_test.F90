module phys_grid_nbrhd_test
  ! Test phys_grid_nbrhd's API. This test has to be in a separate
  ! module to resolve dependencies properly.

  use phys_grid_nbrhd
  use phys_grid
  use dyn_grid

  implicit none
  private

  public :: nbrhd_test_api

contains

  subroutine nbrhd_test_api()
  end subroutine nbrhd_test_api

end module phys_grid_nbrhd_test
