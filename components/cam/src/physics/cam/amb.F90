module amb
  use spmd_utils, only: iam, masterproc
  use shr_kind_mod, only: r8 => shr_kind_r8, r4 => shr_kind_r4

  implicit none

contains
  function assert(cond, message) result(out)
    logical, intent(in) :: cond
    character(len=*), intent(in) :: message

    logical :: out

    if (.not. cond) print *,'amb> assert ',trim(message)
    out = cond
  end function assert

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
    real(r8), intent(in) :: x1,y1,z1,lat,lon

    real(r8) :: x2,y2,z2,angle

    call latlon2xyz(lat,lon,x2,y2,z2)
    ! atan2(|v1 x v2|, v1 . v2)
    angle = atan2(sqrt((y1*z2 - y2*z1)**2 + (x2*z1 - x1*z2)**2 + (x1*y2 - x2*y1)**2), &
                  x1*x2 + y1*y2 + z1*z2)
  end function unit_sphere_angle

  function lower_bound(n, a, val) result (idx)
    integer, intent(in) :: n
    real(r8), intent(in) :: a(n), val

    integer :: idx

    idx = 1
  end function lower_bound

  function upper_bound(n, a, val) result (idx)
    integer, intent(in) :: n
    real(r8), intent(in) :: a(n), val

    integer :: idx

    idx = 2
  end function upper_bound

  subroutine run_unit_tests()
  end subroutine run_unit_tests
end module amb
