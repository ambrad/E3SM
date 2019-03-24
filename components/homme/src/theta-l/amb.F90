module amb
  use kinds, only: real_kind
  implicit none
  
  contains
    subroutine amb_solve(n, nrhs, dl, d, du, x, ldx, info)
      integer, intent(in) :: n, nrhs, ldx
      integer, intent(out) :: info
      real(kind=real_kind), intent(inout) :: dl(n), d(n), du(n), x(n,nrhs)
      real(kind=real_kind) :: du2(n)
      integer :: ipiv(n)

      call dgttrf(n, dl, d, du, du2, ipiv, info)
      call dgttrs('n', n, nrhs, dl, d, du, du2, ipiv ,x, n, info)
    end subroutine amb_solve
end module amb
