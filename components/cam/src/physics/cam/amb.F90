module amb !todo rename phys_grid_nbrhd_utils
  use spmd_utils, only: iam, masterproc
  use shr_kind_mod, only: r8 => shr_kind_r8, r4 => shr_kind_r4

  implicit none

  type SparseTriple
     ! xs(i) maps to ys(yptr(i):yptr(i+1)-1)
     integer, pointer :: xs(:), yptr(:), ys(:)
  end type SparseTriple

contains
  function test(cond, message) result(out)
    ! Assertion that is always enabled, for use in unit tests.

    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    if (.not. cond) print *, 'amb> assert ', trim(message)
    out = cond
  end function test

  function assert(cond, message) result(out)
    ! Assertion that can be disabled.

    logical, intent(in) :: cond
    character(len=*), intent(in) :: message
    logical :: out

    out = test(cond, message)
  end function assert

  function reldif(a, b) result(r)
    real(r8), intent(in) :: a, b
    real(r8) :: r

    r = abs(b - a)
    if (a == 0) return
    r = r/abs(a)
  end function reldif

  subroutine latlon2xyz(lat,lon,x,y,z)
    real(r8), intent(in) :: lat,lon
    real(r8), intent(out) :: x,y,z

    real(r8) :: sinl, cosl

    sinl = sin(lat)
    cosl = cos(lat);
    x = cos(lon)*cosl
    y = sin(lon)*cosl
    z = sinl
  end subroutine latlon2xyz

  function unit_sphere_angle(x1,y1,z1,lat,lon) result(angle)
    ! Angle between (x1,y1,z1) and (lat,lon).

    real(r8), intent(in) :: x1,y1,z1,lat,lon

    real(r8) :: x2,y2,z2,angle

    call latlon2xyz(lat,lon,x2,y2,z2)
    ! atan2(|v1 x v2|, v1 . v2)
    angle = atan2(sqrt((y1*z2 - y2*z1)**2 + (x2*z1 - x1*z2)**2 + (x1*y2 - x2*y1)**2), &
                  x1*x2 + y1*y2 + z1*z2)
  end function unit_sphere_angle

  function upper_bound_or_in_range(n, a, val, k_in) result (k)
    ! Find k such that
    !   if k > 1 then a(k-1) <= val
    !   if k < n then           val < a(k)
    ! where a(1:n) has unique elements and is ascending. k_in is an
    ! optional hint.

    integer, intent(in) :: n
    integer, intent(in), optional :: k_in
    real(r8), intent(in) :: a(n), val
    
    integer :: lo, hi, k
    logical :: e

    k = 1
    if (present(k_in) .and. k_in >= 1 .and. k_in <= n) k = k_in
    if (val < a(k)) then
       lo = 1
       hi = k
    else
       lo = k
       hi = n
    end if
    do while (hi > lo + 1)
       k = (lo + hi)/2
       e = assert(k > lo .and. k < hi, 'upper_bound_or_in_range k')
       if (val < a(k)) then
          hi = k
       else
          lo = k
       end if
    end do
    k = hi
    e = assert((k == 1 .or. a(k-1) <= val) .and. (k == n .or. val < a(k)), &
               'upper_bound_or_in_range post')
  end function upper_bound_or_in_range

  function binary_search(n, a, val, k_in) result (k)
    ! Find position of val in a(1:n), or return -1 if val is not in
    ! a. k_in is an optional hint.

    integer, intent(in) :: n, a(n), val
    integer, intent(in), optional :: k_in

    integer :: lo, hi, k
    logical :: e

    k = 1
    if (present(k_in) .and. k_in >= 1 .and. k_in <= n) k = k_in
    if (val < a(k)) then
       lo = 1
       hi = k
    else
       lo = k
       hi = n
    end if
    do while (hi > lo + 1)
       k = (lo + hi)/2
       if (val < a(k)) then
          hi = k
       else
          lo = k
          if (a(k) == val) exit
       end if
    end do
    if (a(lo) == val) then
       k = lo
    else if (a(hi) == val) then
       k = hi
    else
       k = -1
    end if
    e = assert(k == -1 .or. (k >= 1 .and. k <= n), 'binary_search post')
  end function binary_search

  subroutine array_realloc(a, n, n_new)
    ! Reallocate a to size n_new, preserving the first min(n,n_new)
    ! values.

    integer, pointer, intent(inout) :: a(:)
    integer, intent(in) :: n, n_new

    integer, allocatable :: buf(:)
    integer :: i, n_min
    logical :: e

    e = assert(n_new >= 1 .and. size(a) >= n, 'array_realloc size(a)')
    if (n == 0) then
       deallocate(a)
       allocate(a(n_new))
       return
    end if
    n_min = min(n, n_new)
    allocate(buf(n_min))
    buf(1:n_min) = a(1:n_min)
    deallocate(a)
    allocate(a(n_new))
    a(1:n_min) = buf(1:n_min)
    deallocate(buf)
  end subroutine array_realloc

  subroutine make_unique(n, a, ua)
    ! On exit, ua is the sorted list of unique entries in a(1:n)

    use m_MergeSorts, only: IndexSet, IndexSort

    integer, intent(in) :: n, a(n)
    integer, pointer, intent(out) :: ua(:)

    integer, allocatable :: idxs(:)
    integer :: cnt, prev, i
    logical :: e

    e = assert(n > 0, 'make_unique: n > 0')
    allocate(idxs(n))
    call IndexSet(n, idxs)
    call IndexSort(n, idxs, a)
    ! Count unique entries.
    cnt = 1
    prev = a(idxs(1))
    do i = 2, n
       if (a(idxs(i)) == prev) cycle
       cnt = cnt + 1
       prev = a(idxs(i))
    end do
    ! Fill unique list.
    allocate(ua(cnt))
    cnt = 1
    prev = a(idxs(1))
    ua(cnt) = prev
    do i = 2, n
       if (a(idxs(i)) == prev) cycle
       cnt = cnt + 1
       prev = a(idxs(i))
       ua(cnt) = prev
    end do
    deallocate(idxs)
  end subroutine make_unique

  subroutine SparseTriple_nullify(st)
    type (SparseTriple), intent(out) :: st
    st%xs => null()
    st%yptr => null()
    st%ys => null()
  end subroutine SparseTriple_nullify

  subroutine SparseTriple_deallocate(st)
    type (SparseTriple), intent(out) :: st
    deallocate(st%xs, st%yptr, st%ys)
    call SparseTriple_nullify(st)
  end subroutine SparseTriple_deallocate

  function SparseTriple_in_xs(st, x) result(k)
    ! Find the position of x in st%xs(:) or -1 if not present.
    type (SparseTriple), intent(in) :: st
    integer, intent(in) :: x
    integer :: k
    k = binary_search(size(st%xs), st%xs, x, 1)
  end function SparseTriple_in_xs

  subroutine run_unit_tests()
    ! Unit tests for nbrhd_* helper routines.

    use shr_const_mod, only: pi => shr_const_pi

    integer, parameter :: n = 4, b(n) = (/ -2, -1, 3, 7 /), &
         m = 6, c(m) = (/ 1, 3, 2, -1, 2, 3 /)
    real(r8), parameter :: a(n) = (/ -1.0_r8, 1.0_r8, 1.5_r8, 3.0_r8 /), &
         tol = epsilon(1.0_r8)

    integer, pointer :: uc(:)
    real(r8) :: lat1, lon1, lat2, lon2, x1, y1, z1, x2, y2, z2, angle
    integer :: k
    logical :: e

    k = upper_bound_or_in_range(n, a, -1.0_r8); e = test(k == 2, 'uboir 1')
    k = upper_bound_or_in_range(n, a, -1.0_r8, 2); e = test(k == 2, 'uboir 2')
    k = upper_bound_or_in_range(n, a, 3.0_r8, 2); e = test(k == n, 'uboir 3')
    k = upper_bound_or_in_range(n, a, 3.0_r8, -11); e = test(k == n, 'uboir 4')
    k = upper_bound_or_in_range(n, a, 1.2_r8); e = test(k == 3, 'uboir 5')

    k = binary_search(n, b, -22, -5); e = test(k == -1, 'binsrc 1')
    k = binary_search(n, b, -2); e = test(k == 1, 'binsrc 2')
    k = binary_search(n, b, 3, 2); e = test(k == 3, 'binsrc 3')
    k = binary_search(n, b, 7); e = test(k == 4, 'binsrc 4')
    k = binary_search(n, b, 7, 4); e = test(k == 4, 'binsrc 5')
    k = binary_search(n, b, 7, 5); e = test(k == 4, 'binsrc 6')
    k = binary_search(n, b, 0, 15); e = test(k == -1, 'binsrc 7')

    lat1 = -pi/3
    lon1 = pi/2
    call latlon2xyz(lat1, lon1, x1, y1, z1)
    lat2 = lat1 - 0.1_r8
    lon2 = lon1
    angle = unit_sphere_angle(x1, y1, z1, lat2, lon2)
    e = test(reldif(0.1_r8,angle) <= 10*tol, 'usa 1')

    call make_unique(m, c, uc)
    e = test(size(uc) == 4, 'uc length')
    e = test(uc(1) == -1, 'first(uc)')
    e = test(uc(4) == 3, 'last(uc)')
    do k = 2, size(uc)
       e = test(uc(k) > uc(k-1), 'uc sorted')
    end do
    deallocate(uc)
  end subroutine run_unit_tests
end module amb
