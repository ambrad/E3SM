module dcmip2012_test1_conv

  ! DCMIP 2012 tests 1-1,2,3, with modification for good convergence
  ! testing. See dcmip2012_test1_2_3.F90 for the original code.

  use parallel_mod,       only: abortmp
  ! Use physical constants consistent with HOMME
  use physical_constants, only: a => rearth0, Rd => Rgas, g, cp, pi => dd_pi, p0

  implicit none
  private

  integer, parameter :: rt = 8

  real(rt), parameter :: &
       tau     = 12.d0 * 86400.d0       ! period of motion 12 days

  public :: test1_conv_advection

contains

  subroutine get_nondiv2d_uv(time, lon, lat, u, v)
    ! Classic 2D nondivergent flow field.

    real(rt), intent(in ) :: time, lon, lat
    real(rt), intent(out) :: u, v

    real(rt), parameter :: &
         u0      = (2.d0*pi*a)/tau,    &  ! 2 pi a / 12 days
         k0      = (10.d0*a)/tau          ! velocity magnitude

    real(rt) :: lonp

    ! translational longitude
    lonp = lon - 2.d0*pi*time/tau
    ! zonal velocity
    u = k0*sin(lonp)*sin(lonp)*sin(2.d0*lat)*cos(pi*time/tau) + u0*cos(lat)
    ! meridional velocity
    v = k0*sin(2.d0*lonp)*cos(lat)*cos(pi*time/tau)
  end subroutine get_nondiv2d_uv

  subroutine get_nondiv3d_uv(bs, pbot, ptop, zbot, ztop, ztaper, time, lon, lat, p, z, u, v, w)
    real(rt), intent(in ) :: bs, pbot, ptop, zbot, ztop, ztaper, time, lon, lat, p, z
    real(rt), intent(out) :: u, v, w

    real(rt), parameter :: omega0  = (2*23000.d0*pi)/tau, T0 = 300.d0

    real(rt) :: s, s_p, lonp, ud, c, arg
    
    ! This is essentially the test 1-1 flow. The key difference in this flow
    ! removes the factor of 2 in ud and w cos-time factors. The 2 in the
    ! original code makes trajectories not return to their initial points.

    ! Shape function in p.
    if (p >= pbot .or. p <= ptop) then
       s = 0
       s_p = 0
    else
       c = 0.3d0
       arg = pi*(p - ptop)/(pbot - ptop)
       s = c*sin(arg)**3
       s_p = (3*c*pi/(pbot - ptop))*sin(arg)**2*cos(arg)
    end if
    ! Translational longitude.
    lonp = lon - 2.d0*pi*time/tau
    ! Nondivergent 2D flow.
    call get_nondiv2d_uv(time, lon, lat, u, v)
    ! Taper the 2D nondiv (u,v) flow in the z direction. This does not induce
    ! any w, and the 2D field remains nondivergent at each z.
    u = u*ztaper
    v = v*ztaper
    ! Divergent flow.
    ud = (omega0*a)*cos(lonp)*(cos(lat)**2.0)*cos(pi*time/tau)*s_p
    u = u + ud
    w = -((Rd*T0)/(g*p))*omega0*sin(lonp)*cos(lat)*cos(pi*time/tau)*s
  end subroutine get_nondiv3d_uv

  function get_2d_cinf_tracer(lon, lat) result(q)
    real(rt), intent(in) :: lon, lat

    real(rt) :: q

    real(rt) :: x, y, zeta

    x = cos(lat)*cos(lon)
    y = cos(lat)*sin(lon)
    zeta = sin(lat)
    q = 1.5d0*(1 + sin(pi*x)*sin(pi*y)*sin(pi*zeta))
  end function get_2d_cinf_tracer

  subroutine ll2xyz(lon, lat, x, y, z)
    ! Unit sphere.

    real(rt), intent(in) :: lon, lat
    real(rt), intent(out) :: x, y, z

    real(rt) :: sinl, cosl

    sinl = sin(lat)
    cosl = cos(lat)
    x = cos(lon)*cosl
    y = sin(lon)*cosl
    z = sinl
  end subroutine ll2xyz

  function great_circle_dist(lon1, lat1, lon2, lat2) result(d)
    ! Unit sphere.
    
    real(rt), intent(in) :: lon1, lat1, lon2, lat2
    real(rt) :: d
    
    real(rt) xA, yA, zA, xB, yB, zB, cp1, cp2, cp3, cpnorm, dotprod
    
    call ll2xyz(lon1, lat1, xA, yA, zA)
    call ll2xyz(lon2, lat2, xB, yB, zB)
    cp1 = yA*zB - yB*zA
    cp2 = xB*zA - xA*zB
    cp3 = xA*yB - xB*yA
    cpnorm = sqrt(cp1*cp1 + cp2*cp2 + cp3*cp3)
    dotprod = xA*xB + yA*yB + zA*zB
    d = atan2(cpnorm, dotprod)
  end function great_circle_dist

  function q_gh(x, y, z, xi, yi, zi) result(q)
    real(rt), intent(in) :: x, y, z, xi, yi, zi
    real(rt) :: q

    real(rt), parameter :: h_max = 0.95d0, b = 5.d0
    real(rt) :: r2
    
    r2 = (x - xi)**2 + (y - yi)**2 + (z - zi)**2
    q = exp(-b*r2)
  end function q_gh

  function q_cb(r, ri) result(q)
    real(rt), intent(in) :: r, ri
    real(rt) :: q

    real(rt), parameter :: h_max = 1.d0
    
    q = 0.5d0*h_max*(1 + cos(pi*ri/r))
  end function q_cb

  function get_2d_gaussian_hills(lon, lat) result(q)
    real(rt), intent(in) :: lon, lat
    real(rt) :: q

    real(rt), parameter :: lon1 = 5.d0*(pi/6.d0), lat1 = 0, lon2 = -lon1, lat2 = 0
    real(rt) :: x1, y1, z1, x2, y2, z2, x, y, z

    call ll2xyz(lon1, lat1, x1, y1, z1)
    call ll2xyz(lon2, lat2, x2, y2, z2)
    call ll2xyz(lon, lat, x, y, z)
    q = q_gh(x, y, z, x1, y1, z1) + q_gh(x, y, z, x2, y2, z2)
  end function get_2d_gaussian_hills

  function get_2d_cosine_bells(lon, lat) result(q)
    real(rt), intent(in) :: lon, lat
    real(rt) :: q

    real(rt), parameter :: lon1 = 5.d0*(pi/6.d0), lat1 = 0, lon2 = -lon1, lat2 = 0, &
         &                 r = 0.5d0, b = 0.1d0, c = 0.9d0
    real(rt) :: h, ri

    h = 0
    ri = great_circle_dist(lon, lat, lon1, lat1)
    if (ri < r) then
       h = q_cb(r, ri)
    else
       ri = great_circle_dist(lon, lat, lon2, lat2)
       if (ri < r) h = q_cb(r, ri)
    end if
    q = b + c*h
  end function get_2d_cosine_bells

  function get_2d_correlated_cosine_bells(lon, lat) result(q)
    real(rt), intent(in) :: lon, lat
    real(rt) :: q

    real(rt), parameter :: a = -0.8d0, b = 0.9d0

    q = get_2d_cosine_bells(lon, lat)
    q = a*q + b
  end function get_2d_correlated_cosine_bells

  subroutine test1_conv_advection_deformation( &
       time,lon,lat,p,z,zcoords,u,v,w,t,phis,ps,rho,q1,q2,q3,q4)
    !-----------------------------------------------------------------------
    !     input/output params parameters at given location
    !-----------------------------------------------------------------------
    
    real(rt), intent(in)     :: time       ! simulation time (s)
    real(rt), intent(in)     :: lon        ! Longitude (radians)
    real(rt), intent(in)     :: lat        ! Latitude (radians)
    real(rt), intent(in)     :: z          ! Height (m)
    real(rt), intent(inout)  :: p          ! Pressure  (Pa)
    integer , intent(in)     :: zcoords    ! 0 or 1 see below
    real(rt), intent(out)    :: u          ! Zonal wind (m s^-1)
    real(rt), intent(out)    :: v          ! Meridional wind (m s^-1)
    real(rt), intent(out)    :: w          ! Vertical Velocity (m s^-1)
    real(rt), intent(out)    :: T          ! Temperature (K)
    real(rt), intent(out)    :: phis       ! Surface Geopotential (m^2 s^-2)
    real(rt), intent(out)    :: ps         ! Surface Pressure (Pa)
    real(rt), intent(out)    :: rho        ! density (kg m^-3)
    real(rt), intent(out)    :: q1         ! Tracer q1 (kg/kg)
    real(rt), intent(out)    :: q2         ! Tracer q2 (kg/kg)
    real(rt), intent(out)    :: q3         ! Tracer q3 (kg/kg)
    real(rt), intent(out)    :: q4         ! Tracer q4 (kg/kg)

    ! if zcoords = 1, then we use z and output p
    ! if zcoords = 0, then we use p 

    !-----------------------------------------------------------------------
    !     test case parameters
    !----------------------------------------------------------------------- 
    real(rt), parameter ::           &
         omega0  = (2*23000.d0*pi)/tau,  &  ! velocity magnitude
         T0      = 300.d0,             &  ! temperature
         H       = Rd * T0 / g,        &  ! scale height
         RR      = 1.d0/2.d0,          &  ! horizontal half width divided by 'a'
         ZZ      = 1000.d0,            &  ! vertical half width
         z0      = 5000.d0,            &  ! center point in z
         lambda0 = 5.d0*pi/6.d0,       &  ! center point in longitudes
         lambda1 = 7.d0*pi/6.d0,       &  ! center point in longitudes
         phi0    = 0.d0,               &  ! center point in latitudes
         phi1    = 0.d0, &
         ztop    = 12000.d0

    real(rt) :: height                                                     ! The height of the model levels
    real(rt) :: ptop                                                       ! model top in p
    real(rt) :: sin_tmp, cos_tmp, sin_tmp2, cos_tmp2                       ! Calculate great circle distances
    real(rt) :: d1, d2, r, r2, d3, d4                                      ! For tracer calculations
    real(rt) :: s, bs, s_p                                                 ! Shape function, and parameter
    real(rt) :: lonp                                                       ! Translational longitude, depends on time
    real(rt) :: ud                                                         ! Divergent part of u
    real(rt) :: x,y,zeta,tmp

    !---------------------------------------------------------------------
    !    HEIGHT AND PRESSURE
    !---------------------------------------------------------------------
    
    ! height and pressure are aligned (p = p0 exp(-z/H))
    if (zcoords .eq. 1) then
       height = z
       p = p0 * exp(-z/H)
    else
       height = H * log(p0/p)
    endif

    ! model top in p
    ptop = p0*exp(-ztop/H)

    !---------------------------------------------------------------------
    !    THE VELOCITIES ARE TIME DEPENDENT AND THEREFORE MUST BE UPDATED
    !    IN THE DYNAMICAL CORE
    !---------------------------------------------------------------------

    ! shape function
    bs = 1.0d0
    s = 1.0 + exp((ptop-p0)/(bs*ptop)) - exp((p-p0)/(bs*ptop)) - exp((ptop-p)/(bs*ptop))
    s_p = (-exp((p-p0)/(bs*ptop)) + exp((ptop-p)/(bs*ptop)))/(bs*ptop)

    ! translational longitude
    lonp = lon - 2.d0*pi*time/tau

    ! The key difference in this test relative to the original is removing the
    ! factor of 2 in ud and w cos-time factors. The 2 in the original code makes
    ! trajectories not return to their initial points.

    call get_nondiv2d_uv(time, lon, lat, u, v)

    ! divergent part of zonal velocity
    ud = (omega0*a) * cos(lonp) * (cos(lat)**2.0) * cos(pi*time/tau) * s_p
    u = u + ud

    ! vertical velocity - can be changed to vertical pressure velocity by
    ! omega = -(g*p)/(Rd*T0)*w
    w = -((Rd*T0)/(g*p))*omega0*sin(lonp)*cos(lat)*cos(pi*time/tau)*s

    !-----------------------------------------------------------------------
    !    TEMPERATURE IS CONSTANT 300 K
    !-----------------------------------------------------------------------
    t = T0

    !-----------------------------------------------------------------------
    !    PHIS (surface geopotential) 
    !-----------------------------------------------------------------------
    phis = 0.d0

    !-----------------------------------------------------------------------
    !    PS (surface pressure)
    !-----------------------------------------------------------------------
    ps = p0

    !-----------------------------------------------------------------------
    !    RHO (density)
    !-----------------------------------------------------------------------
    rho = p/(Rd*t)

    !-----------------------------------------------------------------------
    !     initialize Q, set to zero 
    !-----------------------------------------------------------------------
    !  q = 0.d0

    !-----------------------------------------------------------------------
    !     initialize tracers
    !-----------------------------------------------------------------------

    x = cos(lat)*cos(lon)
    y = cos(lat)*sin(lon)
    zeta = sin(lat)
    ! tracer 1 - a C^inf tracer field for order of accuracy analysis
    q1 = 0.5d0*(1 + sin(pi*x)*sin(pi*y)*sin(pi*zeta)*sin(pi*(p-ptop)/(p0-ptop)))

    ! tracer 2 - correlated with 1
    q2 = 0.9d0 - 0.8d0*q1**2

    ! tracer 3 - slotted ellipse

    sin_tmp = sin(lat) * sin(phi0)
    cos_tmp = cos(lat) * cos(phi0)
    sin_tmp2 = sin(lat) * sin(phi1)
    cos_tmp2 = cos(lat) * cos(phi1)

    ! great circle distance without 'a'
    r  = ACOS (sin_tmp + cos_tmp*cos(lon-lambda0)) 
    r2 = ACOS (sin_tmp2 + cos_tmp2*cos(lon-lambda1)) 
    d1 = min( 1.d0, (r/RR)**2 + ((height-z0)/ZZ)**2 )
    d2 = min( 1.d0, (r2/RR)**2 + ((height-z0)/ZZ)**2 )

    ! make the ellipse
    if (d1 .le. RR) then
       q3 = 1.d0
    elseif (d2 .le. RR) then
       q3 = 1.d0
    else
       q3 = 0.1d0
    endif

    ! put in the slot
    if (height .gt. z0 .and. abs(lat) .lt. 0.125d0) then
       q3 = 0.1d0
    endif

    ! tracer 4: q4 is chosen so that, in combination with the other three tracer
    !           fields with weight (3/10), the sum is equal to one
    q4 = 1.d0 - 0.3d0*(q1+q2+q3)

  end subroutine test1_conv_advection_deformation

  subroutine test1_conv_advection_orography( &
       test_minor,time,lon,lat,p,z,zcoords,cfv,hybrid_eta,hya,hyb,u,v,w,t,phis,ps,rho,q1,q2,q3,q4)

    character(len=1), intent(in) :: test_minor ! a, b, or c
    real(rt), intent(in)  :: time            ! simulation time (s)
    real(rt), intent(in)  :: lon             ! Longitude (radians)
    real(rt), intent(in)  :: lat             ! Latitude (radians)
    real(rt), intent(in)  :: hya             ! A coefficient for hybrid-eta coordinate
    real(rt), intent(in)  :: hyb             ! B coefficient for hybrid-eta coordinate

    logical, intent(in)  :: hybrid_eta       ! flag to indicate whether the hybrid sigma-p (eta) coordinate is used

    real(rt), intent(out)  :: p              ! Pressure  (Pa)
    real(rt), intent(out)  :: z              ! Height (m)

    integer , intent(in)     :: zcoords      ! 0 or 1 see below
    integer , intent(in)     :: cfv          ! 0, 1 or 2 see below
    real(rt), intent(out)    :: u            ! Zonal wind (m s^-1)
    real(rt), intent(out)    :: v            ! Meridional wind (m s^-1)
    real(rt), intent(out)    :: w            ! Vertical Velocity (m s^-1)
    real(rt), intent(out)    :: t            ! Temperature (K)
    real(rt), intent(out)    :: phis         ! Surface Geopotential (m^2 s^-2)
    real(rt), intent(out)    :: ps           ! Surface Pressure (Pa)
    real(rt), intent(out)    :: rho          ! density (kg m^-3)
    real(rt), intent(out)    :: q1           ! Tracer q1 (kg/kg)
    real(rt), intent(out)    :: q2           ! Tracer q2 (kg/kg)
    real(rt), intent(out)    :: q3           ! Tracer q3 (kg/kg)
    real(rt), intent(out)    :: q4           ! Tracer q4 (kg/kg)

    real(rt), parameter :: &
         u0      = 2.d0*pi*a/tau,       &  ! Velocity Magnitude (m/s)
         T0      = 300.d0,              &  ! temperature (K)
         H       = Rd * T0 / g,         &  ! scale height (m)
         alpha   = pi/6.d0,             &  ! rotation angle (radians), 30 degrees
         lambdam = 3.d0*pi/2.d0,        &  ! mountain longitude center point (radians)
         phim    = 0.d0,                &  ! mountain latitude center point (radians)
         h0      = 2000.d0,             &  ! peak height of the mountain range (m)
         Rm      = 3.d0*pi/4.d0,        &  ! mountain radius (radians)
         ztop    = 12000.d0,            &  ! model top (m)
         ztop_t  = 2000.d0,             &  ! transition layer
         zbot_q  = ztop_t + 500.d0,     &  ! bottom of tracers; below, all q = 0
         lon_offset = 0.5d0*pi,         &  ! longitudinal translation of std 2d test flow and qs
         ! For Hadley-like flow. Multiply w and tracer vertical extent by (ztop
         ! - ztop_t)/ztop to compensate for smaller domain.
         tau_h   = 1.d0 * 86400.d0,     &  ! period of motion 1 day (in s)
         z1_h    = ztop_t + 1000.d0,    &  ! position of lower tracer bound (m)
         z2_h    = z1_h + 6000.d0,      &  ! position of upper tracer bound (m)
         z0_h    = 0.5d0*(z1_h+z2_h),   &  ! midpoint (m)
         u0_h    = 250.d0,              &  ! Zonal velocity magnitude (m/s)
         ! w0_h is the main parameter to modify to make the test easier (smaller
         ! w0_h) or harder (larger).
         w0_h    = 0.05d0,              &  ! Vertical velocity magnitude (m/s)
         ! For 3D deformational flow.
         bs_a    = 1.0d0                   ! shape function smoothness

    real(rt) :: height             ! Model level heights (m)
    real(rt) :: r                  ! Great circle distance (radians)
    real(rt) :: rz                 ! height differences
    real(rt) :: zs                 ! Surface elevation (m)
    real(rt) :: zetam, ztaper, rho0, z_q_shape, ptop, ptop_t, c0, fl, fl_lat, gz, gz_z, fz, fz_z

    if (cfv /= 0)         call abortmp('test1_conv_advection_orography does not support cfv != 0')
    if (.not. hybrid_eta) call abortmp('test1_conv_advection_orography does not support !hybrid_eta')
    if (zcoords /= 0)     call abortmp('test1_conv_advection_orography does not support zcoords != 0')

    ! mountain oscillation half-width (radians)
    zetam = pi/16.d0
    if (test_minor == 'd') zetam = pi/8.d0
    
    r = acos(sin(phim)*sin(lat) + cos(phim)*cos(lat)*cos(lon - lambdam))
    if (r .lt. Rm) then
       zs = (h0/2.d0)*(1.d0+cos(pi*r/Rm))*cos(pi*r/zetam)**2.d0
    else
       zs = 0.d0
    endif
    zs = -zs ! holes instead of mountains
    phis = g*zs
    ps = p0 * exp(-zs/H)

    p = hya*p0 + hyb*ps
    height = H * log(p0/p)
    z = height

    T = T0

    rho = p/(Rd*T)
    rho0 = p0/(Rd*T)

    if (z <= 0) then
       ztaper = 0
    elseif (z >= ztop_t) then
       ztaper = 1
    else
       ztaper = (1 + cos(pi*(1 + z/ztop_t)))/2
    end if

    w = 0.d0

    select case(test_minor)
    case('a')
       ! Solid body rotation
       ! Zonal Velocity
       u = u0*(cos(lat)*cos(alpha)+sin(lat)*cos(lon)*sin(alpha))
       ! Meridional Velocity
       v = -u0*(sin(lon)*sin(alpha))
    case('b')
       ! 2D nondiv flow in each layer.
       call get_nondiv2d_uv(time, lon + lon_offset, lat, u, v)
       u = u*ztaper
       v = v*ztaper
    case('c')
       ! Moving vortices.
    case('d')
       ! 3D nondiv flow.
       ptop_t = p0*exp(-ztop_t/H)
       ptop = p0*exp(-ztop/H)
       call get_nondiv3d_uv(bs_a, ptop_t, ptop, ztop_t, ztop, ztaper, &
            &               time, lon + lon_offset, lat, p, z, u, v, w)
    case('e')
       u = u0_h*cos(lat)*cos(pi*time/tau_h)*ztaper
       fl = cos(lat)**2
       fl_lat = -2*cos(lat)*sin(lat)
       if (z <= 0) then
          fz = 0
          fz_z = 0
       else
          gz = pi*z/ztop
          gz_z = pi/ztop
          fz = -sin(gz)**3
          fz_z = -3*sin(gz)**2*cos(gz)*gz_z
       end if
       c0 = w0_h*(rho0/rho)*cos(pi*time/tau_h)
       w =    c0*(cos(lat)*fl_lat - 2*sin(lat)*fl)*fz
       v = -a*c0*(cos(lat)*fl                    )*fz_z
    end select

    if (time > 0) then
       q1 = 0; q2 = 0; q3 = 0; q4 = 0
       return
    end if

    z_q_shape = 0.5d0*(1 - cos(2*pi*(z - zbot_q)/(ztop - zbot_q)))
    if (z < zbot_q .or. z > ztop) z_q_shape = 0.d0

    select case(test_minor)
    case('e')
       if (height < z2_h .and. height > z1_h) then
          q1 = 0.5d0 * (1.d0 + cos(2.d0*pi*(z-z0_h)/(z2_h-z1_h)))
       else
          q1 = 0.d0
       end if
       q2 = q1 * get_2d_cinf_tracer(lon, lat)
       q3 = q1 * get_2d_gaussian_hills(lon - lon_offset, lat)
       q4 = q1 * get_2d_cosine_bells(lon - lon_offset, lat)

    case default
       q1 = z_q_shape * get_2d_cinf_tracer(lon, lat)
       q2 = z_q_shape * get_2d_gaussian_hills(lon - lon_offset, lat)
       q3 = z_q_shape * get_2d_cosine_bells(lon - lon_offset, lat)
       q4 = z_q_shape * get_2d_correlated_cosine_bells(lon - lon_offset, lat)
    end select
  end subroutine test1_conv_advection_orography

  subroutine test1_conv_advection(test_case,time,lon,lat,hya,hyb,p,z,u,v,w,use_w,t,phis,ps,rho,q)
    character(len=*), intent(in) :: test_case  ! dcmip2012_test1_{1,2,3a,3b,3c,3d,3e}_conv
    real(rt), intent(in)     :: time       ! simulation time (s)
    real(rt), intent(in)     :: lon, lat   ! Longitude, latitude (radians)
    real(rt), intent(in)     :: hya, hyb   ! Hybrid a, b coefficients
    real(rt), intent(inout)  :: z          ! Height (m)
    real(rt), intent(inout)  :: p          ! Pressure  (Pa)
    real(rt), intent(out)    :: u          ! Zonal wind (m s^-1)
    real(rt), intent(out)    :: v          ! Meridional wind (m s^-1)
    real(rt), intent(out)    :: w          ! Vertical Velocity (m s^-1)
    logical , intent(out)    :: use_w      ! Should caller use w or instead div(u,v)?
    real(rt), intent(out)    :: T          ! Temperature (K)
    real(rt), intent(out)    :: phis       ! Surface Geopotential (m^2 s^-2)
    real(rt), intent(out)    :: ps         ! Surface Pressure (Pa)
    real(rt), intent(out)    :: rho        ! density (kg m^-3)
    real(rt), intent(out)    :: q(5)       ! Tracer q1 (kg/kg)

    integer, parameter :: cfv = 0, zcoords = 0
    logical, parameter :: use_eta = .true.

    character(len=1) :: test_major, test_minor

    test_major = test_case(17:17)
    if (test_major == '3') test_minor = test_case(18:18)

    use_w = .true.
    select case(test_major)
    case('1')
       call test1_conv_advection_deformation( &
            time,lon,lat,p,z,zcoords,u,v,w,t,phis,ps,rho,q(1),q(2),q(3),q(4))
    case ('2')
    case('3')
       !use_w = .false.
       call test1_conv_advection_orography( &
            test_minor,time,lon,lat,p,z,zcoords,cfv,use_eta,hya,hyb,u,v,w,t,phis,ps,rho, &
            q(1),q(2),q(3),q(4))
    end select
  end subroutine test1_conv_advection

end module dcmip2012_test1_conv
