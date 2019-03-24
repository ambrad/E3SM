module amb
  use kinds, only: real_kind
  implicit none
  
  contains
    subroutine amb_solve(output, n, nrhs, dl, d, du, x, ldx, info)
      logical, intent(in) :: output
      integer, intent(in) :: n, nrhs, ldx
      integer, intent(out) :: info
      real(kind=real_kind), intent(inout) :: dl(n-1), d(n), du(n-1), x(n,nrhs)
      real(kind=real_kind) :: du2(n)
      integer :: ipiv(n), i
      real(kind=real_kind) :: off, on, moff, mon

      if (.true.) then
         moff = 1
         mon = 1000
         do i = 2,n-1
            if (i == 1) then
               off = abs(du(1))
               on = abs(d(1))
            else if (i == n) then
               off = abs(dl(n-1))
               on = abs(d(n))
            else
               off = abs(dl(i-1)) + abs(du(i))
               on = abs(d(i))
            end if
            if (on/off < mon/moff) then
               mon = on
               moff = off
            end if
         end do
         if (mon - moff < 0.5) then
            print *,'amb>',mon,moff
         end if
      end if
      call dgttrf(n, dl, d, du, du2, ipiv, info)
      call dgttrs('n', n, nrhs, dl, d, du, du2, ipiv ,x, n, info)
    end subroutine amb_solve
end module amb
