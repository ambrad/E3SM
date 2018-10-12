module amb_mod
  use metagraph_mod, only: MetaVertex_t
  use parallel_mod, only: parallel_t

  implicit none

contains

  subroutine amb_run(par)
    type (parallel_t), intent(in) :: par

    print *, "AMB> hi"
  end subroutine amb_run

  subroutine amb_cmp(mv_other)
    type (MetaVertex_t) :: mv_other
  end subroutine amb_cmp

end module amb_mod
