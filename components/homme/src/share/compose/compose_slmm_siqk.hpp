#ifndef INCLUDE_COMPOSE_SLMM_SIQK_HPP
#define INCLUDE_COMPOSE_SLMM_SIQK_HPP

#include <mpi.h>

//> begin SIQK
// To make this initial integration less messy, inline SIQK in this translation
// unit. Once we've put together the Compose library, I'll remove this code.

#ifndef INCLUDE_SIQK_DEFS_HPP
#define INCLUDE_SIQK_DEFS_HPP

#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include <Kokkos_Core.hpp>

#ifdef SIQK_TIME
# include <unistd.h>
# include <sys/time.h>
# include <sys/resource.h>
#endif

// Always want this for GPU.
#define SIQK_NONRECURSIVE

namespace siqk {
namespace ko = Kokkos;

#define pr(m) do {                                          \
    int _pid_ = 0;                                          \
    MPI_Comm_rank(MPI_COMM_WORLD, &_pid_);                  \
    std::stringstream _ss_;                                 \
    _ss_.precision(15);                                     \
    _ss_ << "slmm: pid " << _pid_ << " " << m << std::endl; \
    std::cerr << _ss_.str();                                \
  } while (0)
#define prc(m) pr(#m << " | " << (m))
#define puf(m)"(" << #m << " " << (m) << ")"
#define pu(m) << " " << puf(m)
template<typename T>
static void prarr (const std::string& name, const T* const v, const size_t n) {
  std::stringstream ss;
  ss << name << ": ";
  for (size_t i = 0; i < n; ++i) ss << " " << v[i];
  pr(ss.str());
}
#define mprarr(m) siqk::prarr(#m, m.data(), m.size())

#define SIQK_THROW_IF(condition, message) do {                          \
    if (condition) {                                                    \
      std::stringstream _ss_;                                           \
      _ss_ << __FILE__ << ":" << __LINE__ << ": The condition:\n" << #condition \
        "\nled to the exception\n" << message << "\n";                  \
        throw std::logic_error(_ss_.str());                             \
    }                                                                   \
  } while (0)

#define SIQK_STDERR_IF(condition, message) do {                   \
    try { SIQK_THROW_IF(condition, message); }                    \
    catch (const std::logic_error& e) { std::cerr << e.what(); }  \
  } while (0)

KOKKOS_INLINE_FUNCTION static void error (const char* const msg)
{ ko::abort(msg); }

KOKKOS_INLINE_FUNCTION static void message (const char* const msg)
{ printf("%s\n", msg); }

typedef int Int;
typedef double Real;

#ifdef KOKKOS_HAVE_CUDA
typedef ko::LayoutLeft Layout;
#else
typedef ko::LayoutRight Layout;
#endif

// SIQK's array types.
typedef ko::View<Real*[3], Layout> Vec3s;
typedef ko::View<const Real*[3], Layout> ConstVec3s;
typedef ko::View<Real*[6], Layout> Vec6s;
typedef ko::View<const Real*[6], Layout> ConstVec6s;
typedef ko::View<Real*[3], ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawVec3s;
typedef ko::View<const Real*[3], ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawConstVec3s;
typedef ko::View<Real*, ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawArray;
typedef ko::View<const Real*, ko::LayoutRight, ko::MemoryTraits<ko::Unmanaged> > RawConstArray;
typedef ko::View<Int**, Layout> Idxs;
typedef ko::View<const Int**, Layout> ConstIdxs;
typedef ko::View<Int*[8], Layout> Nodes;
typedef ko::View<const Int*[8], Layout> ConstNodes;

// Decorator for a View. UnmanagedView<ViewType> gives the same view as
// ViewType, except the memory is unmanaged.
template <typename ViewT>
using UnmanagedView = ko::View<
  typename ViewT::data_type, typename ViewT::array_layout,
  typename ViewT::device_type, ko::MemoryTraits<ko::Unmanaged> >;

// Get the host or device version of the array.
template <typename VT, typename ES> struct InExeSpace {
  typedef VT type;
};
template <typename VT> struct InExeSpace<VT, ko::HostSpace> {
  typedef typename VT::HostMirror type;
};

#ifdef KOKKOS_HAVE_CUDA
// A 1D slice of an array.
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::value_type*, ko::LayoutStride, typename VT::device_type,
         ko::MemoryTraits<ko::Unmanaged> >
slice (const VT& v, Int i) { return ko::subview(v, i, ko::ALL()); }
// An explicitly const 1D slice of an array.
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
ko::View<typename VT::const_value_type*, ko::LayoutStride, typename VT::device_type,
         ko::MemoryTraits<ko::Unmanaged> >
const_slice (const VT& v, Int i) { return ko::subview(v, i, ko::ALL()); }
#else
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::value_type*
slice (const VT& v, Int i) {
  assert(i >= 0 && i < v.extent_int(0));
  return v.data() + v.extent_int(1)*i;
}
template <typename VT> KOKKOS_FORCEINLINE_FUNCTION
typename VT::const_value_type*
const_slice (const VT& v, Int i) {
  assert(i >= 0 && i < v.extent_int(0));
  return v.data() + v.extent_int(1)*i;
}
#endif

// Number of slices in a 2D array, where each row is a slice.
template <typename A2D> KOKKOS_FORCEINLINE_FUNCTION
Int nslices (const A2D& a) { return static_cast<Int>(a.extent_int(0)); }

// Number of entries in a 2D array's row.
template <typename A2D> KOKKOS_FORCEINLINE_FUNCTION
Int szslice (const A2D& a) { return static_cast<Int>(a.extent_int(1)); }

template <typename V, typename CV>
KOKKOS_INLINE_FUNCTION
static void copy (V dst, CV src, const Int n) {
  for (Int i = 0; i < n; ++i) dst[i] = src[i];
}

template <typename DV, typename SV>
void resize_and_copy (DV& d, const SV& s,
                      typename std::enable_if<DV::rank == 1>::type* = 0) {
  ko::resize(d, nslices(s));
  ko::deep_copy(d, s);
}

template <typename DV, typename SV>
void resize_and_copy (DV& d, const SV& s,
                      typename std::enable_if<DV::rank == 2>::type* = 0) {
  ko::resize(d, nslices(s), szslice(s));
  ko::deep_copy(d, s);
}

template <typename DV, typename SA>
void hm_resize_and_copy (DV& d, const SA& s, const Int n) {
  ko::resize(d, n);
  auto d_hm = ko::create_mirror_view(d);
  for (Int i = 0; i < n; ++i) d_hm[i] = s[i];
  ko::deep_copy(d, d_hm);
}

// GPU-friendly replacements for std::min/max.
template <typename T> KOKKOS_INLINE_FUNCTION
const T& min (const T& a, const T& b) { return a < b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
const T& max (const T& a, const T& b) { return a > b ? a : b; }
template <typename T> KOKKOS_INLINE_FUNCTION
void swap (T& a, T&b) {
  T tmp = a;
  a = b;
  b = tmp;
}
template <typename T> KOKKOS_INLINE_FUNCTION constexpr T square (const T& x) { return x*x; }

template<typename T> KOKKOS_INLINE_FUNCTION
T sign (const T& a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }

} // namespace siqk

#endif // INCLUDE_SIQK_DEFS_HPP

#ifndef INCLUDE_SIQK_GEOMETRY_HPP
#define INCLUDE_SIQK_GEOMETRY_HPP

//#include "siqk_defs.hpp"
//#include "siqk_quadrature.hpp"

namespace siqk {

// Vectors and points are 2D. Thus, if you're working on planes in 3D, project
// to a 2D space before calling these.
struct PlaneGeometry {
  template <typename V> KOKKOS_INLINE_FUNCTION
  static void scale (const Real& a, V v) {
    v[0] *= a; v[1] *= a;
  }
  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real dot_c_amb (const CV c, const CV a, const CV b) {
    return c[0]*(a[0] - b[0]) + c[1]*(a[1] - b[1]);
  }
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void combine (const CV u, const CV v, const Real& a, V x) {
    const Real& oma = 1 - a;
    x[0] = oma*u[0] + a*v[0];
    x[1] = oma*u[1] + a*v[1];
  }
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void axpy (const Real& a, const CV x, V y) {
    y[0] += a*x[0];
    y[1] += a*x[1];
  }

  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void edge_normal (const CV e1, const CV e2, V en) {
    en[0] = e1[1] - e2[1];
    en[1] = e2[0] - e1[0];
  }

  template <typename CV> KOKKOS_INLINE_FUNCTION
  static bool inside (const CV v, const CV e1, const CV en) {
    return dot_c_amb(en, v, e1) >= 0;
  }

  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void intersect (const CV v1, const CV v2, const CV e1, const CV en,
                         V intersection) {
    Real a; {
      const Real
        num = dot_c_amb(en, e1, v1),
        den = dot_c_amb(en, v2, v1);
      a = num == 0 || den == 0 ? 0 : num/den;
      a = a < 0 ? 0 : a > 1 ? 1 : a;
    }
    combine(v1, v2, a, intersection);
  }

  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static bool output (const CV v, Int& no, const V vo) {
#ifdef SIQK_DEBUG
    if (no >= nslices(vo)) {
      std::stringstream ss;
      ss << "output: No room in vo; vo.n() is " << nslices(vo) << " but no is "
         << no << "\n";
      message(ss.str().c_str());
    }
#endif
    if (no >= nslices(vo)) return false;
    vo(no,0) = v[0];
    vo(no,1) = v[1];
    ++no;
    return true;
  }

  template <typename CV2s>
  KOKKOS_INLINE_FUNCTION
  static Real calc_area_formula (const CV2s& v, const Int n) {
    Real area = 0;
    for (Int i = 1, ilim = n - 1; i < ilim; ++i)
      area += calc_tri_jacobian(slice(v,0), slice(v,i), slice(v,i+1));
    return 0.5*area;
  }

  template <typename CV, typename CA>
  KOKKOS_INLINE_FUNCTION
  static void bary2coord (const CV v1, const CV v2, const CV v3, const CA alpha,
                          Real u[2]) {
    for (Int k = 0; k < 2; ++k) u[k] = 0;
    axpy(alpha[0], v1, u);
    axpy(alpha[1], v2, u);
    axpy(alpha[2], v3, u);
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION
  static Real calc_tri_jacobian (const CV v1, const CV v2, const CV v3) {
    Real r1[2], r2[2];
    r1[0] = v2[0] - v1[0];
    r1[1] = v2[1] - v1[1];
    r2[0] = v3[0] - v1[0];
    r2[1] = v3[1] - v1[1];
    const Real a = r1[0]*r2[1] - r1[1]*r2[0];
    return a;
  }
};

// All inputs and outputs are relative to the unit-radius sphere. Vectors and
// points are 3D.
struct SphereGeometry {
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void cross (const CV a, const CV b, V c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
  }
  template <typename CVA, typename CVB> KOKKOS_INLINE_FUNCTION
  static Real dot (const CVA a, const CVB b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }
  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real norm2 (const CV v) {
    return dot(v, v);
  }
  template <typename V> KOKKOS_INLINE_FUNCTION
  static void scale (const Real& a, V v) {
    v[0] *= a; v[1] *= a; v[2] *= a;
  }
  template <typename V> KOKKOS_INLINE_FUNCTION
  static void normalize (V v) {
    scale(1.0/std::sqrt(norm2(v)), v);
  }
  template <typename CVC, typename CVA, typename CVB> KOKKOS_INLINE_FUNCTION
  static Real dot_c_amb (const CVC c, const CVA a, const CVB b) {
    return c[0]*(a[0] - b[0]) + c[1]*(a[1] - b[1]) + c[2]*(a[2] - b[2]);
  }
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void axpy (const Real& a, const CV x, V y) {
    y[0] += a*x[0];
    y[1] += a*x[1];
    y[2] += a*x[2];
  }
  template <typename CVX, typename CVY, typename VZ> KOKKOS_INLINE_FUNCTION
  static void axpbyz (const Real& a, const CVX x, const Real& b, const CVY y,
                      VZ z) {
    z[0] = a*x[0] + b*y[0];
    z[1] = a*x[1] + b*y[1];
    z[2] = a*x[2] + b*y[2];
  }
  template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
  static void copy (V d, const CV s) {
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
  }
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void combine (const CV u, const CV v, const Real& a, V x) {
    const Real& oma = 1 - a;
    x[0] = oma*u[0] + a*v[0];
    x[1] = oma*u[1] + a*v[1];
    x[2] = oma*u[2] + a*v[2];
  }

  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void edge_normal (const CV a, const CV b, V en) {
    cross(a, b, en);
    normalize(en);
  }

  // Is v inside the line anchored at a with inward-facing normal n?
  template <typename CVV, typename CVA, typename CVN> KOKKOS_INLINE_FUNCTION
  static bool inside (const CVV v, const CVA a, const CVN n) {
    return dot_c_amb(n, v, a) >= 0;
  }

  /* Let
       en = edge normal
       e1 = edge starting point
       d = en' e1
       v(a) = (1 - a) v1 + a v2.
     Solve n' v = d for a:
       a = (en' (e1 - v1)) / (en' (v2 - v1)).
     Then uvec(v(a)) is the intersection point on the unit sphere. Assume
     intersection exists. (Already filtered by 'inside'.)
  */
  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static void intersect (const CV v1, const CV v2, const CV e1, const CV en,
                         V intersection) {
    Real a; {
      const Real
        num = dot_c_amb(en, e1, v1),
        den = dot_c_amb(en, v2, v1);
      a = num == 0 || den == 0 ? 0 : num/den;
      a = a < 0 ? 0 : a > 1 ? 1 : a;
    }
    combine(v1, v2, a, intersection);
    normalize(intersection);
  }

  template <typename CV, typename V> KOKKOS_INLINE_FUNCTION
  static bool output (const CV v, Int& no, V vo) {
#ifdef SIQK_DEBUG
    if (no >= nslices(vo)) {
      std::stringstream ss;
      ss << "output: No room in vo; vo.n() is " << nslices(vo) << " but no is "
         << no << "\n";
      message(ss.str().c_str());
    }
#endif
    if (no >= nslices(vo)) return false;
    vo(no,0) = v[0];
    vo(no,1) = v[1];
    vo(no,2) = v[2];
    ++no;
    return true;
  }

  //todo Handle non-convex case.
  // This uses a terrible formula, but it's just for testing.
  template <typename CV3s>
  KOKKOS_INLINE_FUNCTION
  static Real calc_area_formula (const CV3s& v, const Int n) {
    Real area = 0;
    for (Int i = 1, ilim = n - 1; i < ilim; ++i) {
      const Real a = calc_arc_length(slice(v,0), slice(v,i));
      const Real b = calc_arc_length(slice(v,i), slice(v,i+1));
      const Real c = calc_arc_length(slice(v,i+1), slice(v,0));
      const Real s = 0.5*(a + b + c);
      const Real d = (std::tan(0.5*s)*std::tan(0.5*(s-a))*
                      std::tan(0.5*(s-b))*std::tan(0.5*(s-c)));
      if (d <= 0) continue;
      area += 4*std::atan(std::sqrt(d));
    }
    return area;
  }
  template <typename CV> KOKKOS_INLINE_FUNCTION
  static Real calc_arc_length (const CV a, const CV b) {
    const Real d = dot(a, b);
    if (d >= 1) return 0;
    return acos(d);
  }

  template <typename CV, typename CA>
  KOKKOS_INLINE_FUNCTION
  static Real calc_tri_jacobian (const CV v1, const CV v2, const CV v3,
                                 const CA alpha, Real u[3]) {
    // V(:,i) is vertex i of the spherical triangle on the unit sphere. The
    // coefs
    //     alpha = [a1, a2, 1 - a1 - a2]'
    //           = [1 0; 0 1; -1 -1] [a1, a2]'
    //           = alpha_a a
    // (barycentric coords) give the location
    //     v = V alpha
    // on the planar triangle, and u = uvec(v) is the point on the unit sphere.
    //   For a planar tri in 3D, the jacobian is
    //     v_a = v_alpha alpha_a
    //         = V [1 0; 0 1; -1 -1]
    //     J = norm(cross(v_a(:,1), v_a(:,2))).
    // For a spherical tri with the same vertices,
    //     u = v/(v' v)^{1/2}
    //     u_a = u_alpha alpha_a
    //         = (v'v)^{-1/2} (I - u u') V alpha_a
    //         = (v'v)^{-1/2} (I - u u') v_a
    //     J = norm(cross(u_a(:,1), u_a(:,2))).
    for (Int k = 0; k < 3; ++k) u[k] = 0;
    axpy(alpha[0], v1, u);
    axpy(alpha[1], v2, u);
    axpy(alpha[2], v3, u);
    const auto oovn = 1/std::sqrt(norm2(u));
    scale(oovn, u);
    Real u_a[3][3];
    axpbyz(1, v1, -1, v3, u_a[0]);
    axpbyz(1, v2, -1, v3, u_a[1]);
    for (int i = 0; i < 2; ++i) {
      axpy(-dot(u, u_a[i]), u, u_a[i]);
      scale(oovn, u_a[i]);
    }
    cross(u_a[0], u_a[1], u_a[2]);
    return std::sqrt(norm2(u_a[2]));
  }
};

} // namespace siqk

#endif // INCLUDE_SIQK_GEOMETRY_HPP

#ifndef INCLUDE_SIQK_INTERSECT_HPP
#define INCLUDE_SIQK_INTERSECT_HPP

//#include "siqk_defs.hpp"
//#include "siqk_geometry.hpp"
//#include "siqk_search.hpp"
//#include "siqk_quadrature.hpp"

namespace siqk {

/* A mesh is described by the following arrays:
       p: 3 x #nodes, the array of vertices.
       e: max(#verts) x #elems, the array of element base-0 indices.
       nml: 3 x #edges, the array of edge normals.
       en: max(#verts) x #elems, the array of edge-normal base-0 indices.
     e. e indexes p. e(i,j) == -1 in column j indicates that j:end are not used.
     nml. As a mesh is refined, cancellation error makes an edge normal based
   off of an element's vertices increasingly inaccurate. Roughly, if an edge
   subtends angle phi of the sphere, -log10(phi/(2 pi)) digits are lost in the
   edge normal. Therefore, we compute edge normals offline, since in certain
   meshes, they can be computed by an accurate means. E.g., in a cubed-sphere
   mesh, the whole line of a square face can be used to compute the edge
   normal. Furthermore, there are far fewer unique edge normals than edges.
 */
template <typename ES = ko::DefaultExecutionSpace>
struct Mesh {
  typedef typename InExeSpace<ConstVec3s, ES>::type RealArray;
  typedef typename InExeSpace<ConstIdxs, ES>::type IntArray;

  RealArray p, nml;
  IntArray e, en;

  Mesh () {}

  Mesh (const Mesh<ko::HostSpace>& m) {
    typename InExeSpace<Vec3s, ES>::type tp, tnml;
    typename InExeSpace<Idxs, ES>::type te, ten;
    resize_and_copy(tp, m.p); p = tp;
    resize_and_copy(tnml, m.nml); nml = tnml;
    resize_and_copy(te, m.e); e = te;
    resize_and_copy(ten, m.en); en = ten;
  }
};

namespace test {
static constexpr Int max_nvert = 20;
static constexpr Int max_hits = 25; // Covers at least a 2-halo.

// Inward-oriented normal. In practice, we want to form high-quality normals
// using information about the cubed-sphere mesh. This is a low-quality
// brute-force calculation.
template <typename geo>
void fill_normals (Mesh<ko::HostSpace>& m) {
  // Count number of edges.
  Int ne = 0;
  for (Int ip = 0; ip < nslices(m.e); ++ip)
    for (Int iv = 0; iv < szslice(m.e); ++iv)
      if (m.e(ip,iv) == -1) break; else ++ne;
  // Fill.
  Idxs::HostMirror en("en", nslices(m.e), szslice(m.e));
  ko::deep_copy(en, -1);
  Vec3s::HostMirror nml("nml", ne);
  Int ie = 0;
  for (Int ip = 0; ip < nslices(m.e); ++ip)
    for (Int iv = 0; iv < szslice(m.e); ++iv)
      if (m.e(ip,iv) == -1)
        break;
      else {
        // Somewhat complicated next node index.
        const Int iv_next = (iv+1 == szslice(m.e) ? 0 :
                             (m.e(ip,iv+1) == -1 ? 0 : iv+1));
        geo::edge_normal(slice(m.p, m.e(ip, iv)), slice(m.p, m.e(ip, iv_next)),
                         slice(nml, ie));
        en(ip,iv) = ie;
        ++ie;
      }
  m.en = en;
  m.nml = nml;
}
} // namespace test
} // namespace siqk

#endif // INCLUDE_SIQK_INTERSECT_HPP

#ifndef INCLUDE_SIQK_SQR_HPP
#define INCLUDE_SIQK_SQR_HPP

//#include "siqk_defs.hpp"
//#include "siqk_intersect.hpp"

namespace siqk {
namespace sqr { // spherical quadrilateral <-> reference square
/* Let p be a 3x4 matrix with p(:,i) the i'th vertex in a spherical quad in CCW
   order. Let (a,b) be coordinates in the reference square [0,1]^2. (Here we
   choose [0,1] instead of [-1,1].) (a,b) = (0,0) corresponds to p(:,1); (1,0)
   is p(:,2); (1,1) is p(:,3); (0,1) is p(:,4).
     The map from reference square to bilinear quad can be written
       T = p*[ 1 -1 1 -1
              -1  1 0  0
              -1  0 0  1
               1  0 0  0]';
       f(a,b) = T(:,1)*a*b + T(:,2)*a + T(:,3)*b + T(:,4);
   The map to the sphere is then completed with
       g(a,b) = norm(f(a,b))
       q = f(a,b) / g(a,b).
   The Jacobian matrix for q is given by
       q_a = f_a/g - (f g_a)/g^2
       g_a = g_f f_a
   and similarly for q_b.
*/

namespace impl {
// Compute T(i,:).
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_T_row (const ConstVec3sT& p, const Quad& e, const Int i,
                 Real& t1, Real& t2, Real& t3, Real& t4) {
  t4 = p(e[0],i);
  t3 = -t4 + p(e[3],i);
  t2 = -t4 + p(e[1],i);
  t1 = -t2 + p(e[2],i) - p(e[3],i);
}

// Compute T(:,1)*a*b + T(:,2)*a + T(:,3)*b + T(:,4).
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_ref_to_bilinear (const ConstVec3sT& p, const Quad& e,
                           Real a, Real b, Real q[3]) {
  a = 0.5*(a + 1);
  b = 0.5*(b + 1);
  for (Int i = 0; i < 3; ++i) {
    Real t1, t2, t3, t4;
    impl::calc_T_row(p, e, i, t1, t2, t3, t4);
    q[i] = t1*a*b + t2*a + t3*b + t4;
  }
}

// The residual function is r(a,b) = f(a,b)/g(a,b) - q.
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_residual (const ConstVec3sT& p, const Quad& e, const Real a,
                    const Real b, const Real q[3], Real r[3]) {
  calc_ref_to_bilinear(p, e, a, b, r);
  const Real rnorm = std::sqrt(SphereGeometry::norm2(r));
  for (Int i = 0; i < 3; ++i)
    r[i] = r[i]/rnorm - q[i];  
}

// Compute the Jacobian matrix of the residual function: Jacobian(ref square ->
// sphere).
//   TODO Consider rewriting this in terms of the p=1 basis isoparametric
// interpolation formulation. Better performance? See
// calc_isoparametric_jacobian in slmmir.cpp.
template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_Jacobian (const ConstVec3sT& p, const Quad& e, Real a, Real b,
                    Real J[6]) {
  a = 0.5*(a + 1);
  b = 0.5*(b + 1);  
  Real r[3];
  for (Int i = 0; i < 3; ++i) {
    Real t1, t2, t3, t4;
    calc_T_row(p, e, i, t1, t2, t3, t4);
    r[  i] = t1*a*b + t2*a + t3*b + t4;
    J[  i] = t1*b + t2;
    J[3+i] = t1*a + t3;
  }
  Real rtJ[2] = {0};
  for (Int j = 0; j < 2; ++j) {
    const Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      rtJ[j] += r[i]*Jj[i];
  }
  const Real rnorm2 = SphereGeometry::norm2(r), rnorm = std::sqrt(rnorm2);
  for (Int j = 0; j < 2; ++j) {
    Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      Jj[i] = (Jj[i] - r[i]*rtJ[j]/rnorm2)/rnorm;
  }
}

// Solve J dx = r.
KOKKOS_INLINE_FUNCTION
void solve_Jxr (Real J[6], const Real r[3], Real dx[2]) {
  // QR factorization: J -> J [n1 a; 0 n2].
  const Real n1 = std::sqrt(SphereGeometry::norm2(J));
  SphereGeometry::scale(1/n1, J);
  const Real a = SphereGeometry::dot(J, J+3);
  SphereGeometry::axpy(-a, J, J+3);
  const Real n2 = std::sqrt(SphereGeometry::norm2(J+3));
  SphereGeometry::scale(1/n2, J+3);
  // r -> Q' r.
  Real Qtr[2] = {0};
  for (Int j = 0; j < 2; ++j) {
    const Real* const Jj = J + 3*j;
    for (Int i = 0; i < 3; ++i)
      Qtr[j] += Jj[i]*r[i];
  }
  // dx = R \ (Q' r).
  dx[1] = 2*(Qtr[1] / n2);
  dx[0] = 2*((Qtr[0] - a*dx[1]) / n1);
}
} // namespace impl

struct Info {
  bool success;
  Int n_iterations;
};

template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_ref_to_sphere (
  // The spherical quad containing the point.
  const ConstVec3sT& p, const Quad& e,
  // (a,b) in [-1,1]
  const Real a, const Real b,
  // The point on the sphere.
  Real q[3])
{
  impl::calc_ref_to_bilinear(p, e, a, b, q);
  SphereGeometry::normalize(q);
}

template <typename ConstVec3sT, typename Quad>
KOKKOS_INLINE_FUNCTION
void calc_sphere_to_ref (
  // The spherical quad containing the point.
  const ConstVec3sT& p, const Quad& e,
  // The point on the sphere.
  const Real q[3],
  // (a,b) in [-1,1]
  Real& a, Real& b,
  // Optional info output.
  Info* const info = nullptr,
  // Max number of iterations before returning with failure.
  const Int max_its = 10,
  // Tolerance for Newton iteration.
  const Real tol = 1e2*std::numeric_limits<Real>::epsilon())
{
  const Real tol2 = square(tol);
  Real rnorm2 = 1;
  a = b = 0;
  Int it = 0;
  for (it = 1; it <= max_its; ++it) { // Newton's method.
    Real r[3], J[6];
    impl::calc_residual(p, e, a, b, q, r);
    rnorm2 = SphereGeometry::norm2(r);
    if (rnorm2 <= tol2) break;
    impl::calc_Jacobian(p, e, a, b, J);
    Real dx[2];
    impl::solve_Jxr(J, r, dx);
    a -= dx[0];
    b -= dx[1];
  }
  if (info) {
    info->success = rnorm2 <= tol2;
    info->n_iterations = it;
  }
}

// Ref coords, packed (x,y), CCW, starting from (-1,-1).
KOKKOS_INLINE_FUNCTION
const Real* get_ref_vertices () {
  static const Real c[] = {-1, -1, 1, -1, 1, 1, -1, 1};
  return c;
}

namespace test {
struct Info {
  Int sum_nits, max_nits, nfails;
};

class TestSphereToRefKernel {
  const Real a_test[9] = {-0.1, -1e-16, 0, 1e-15, 0.1, 0.7, 1, 1-1e-14, 1.1};
  const Int n_a_test = sizeof(a_test)/sizeof(*a_test);

  const Real tol_;
  mutable ConstVec3s p_;
  mutable ConstIdxs e_;

public:
  typedef Info value_type;

  TestSphereToRefKernel (const ConstVec3s::HostMirror& p_hm,
                         const ConstIdxs::HostMirror& e_hm,
                         const Real tol = 1e1*std::numeric_limits<Real>::epsilon())
    : tol_(tol)
  {
    { Vec3s p; resize_and_copy(p, p_hm); p_ = p; }
    { Idxs e; resize_and_copy(e, e_hm); e_ = e; }
  }

  Int n () const { return nslices(e_)*square(n_a_test); }
  const Real& tol () const { return tol_; }

  KOKKOS_INLINE_FUNCTION
  void operator() (const Int k, value_type& jinfo) const {
    const Int
      ei = k / square(n_a_test),
      ij = k % square(n_a_test),
      i = ij / n_a_test,
      j = ij % n_a_test;
    const Real a_t = a_test[i], b_t = a_test[j];
    Real q[3];
    sqr::calc_ref_to_sphere(p_, slice(e_, ei), a_t, b_t, q);
    Real a, b;
    sqr::Info info;
    sqr::calc_sphere_to_ref(p_, slice(e_, ei), q, a, b, &info, 100, tol_);
    const Real err = std::sqrt(square(a_t - a) + square(b_t - b));
    // tol is on dx, not (a,b), so adjust slightly.
    if ( ! info.success || err > 1e4*tol_) {
      jinfo.nfails++;
      printf("calc_sphere_to_ref ei %d i %d j %d: nits %d re %1.1e\n",
             ei, i, j, info.n_iterations, err);
    }
    jinfo.sum_nits += info.n_iterations;
    jinfo.max_nits = max(jinfo.max_nits, info.n_iterations);
  }

  KOKKOS_INLINE_FUNCTION
  void init (value_type& info) {
    info.sum_nits = 0;
    info.max_nits = 0;
    info.nfails = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join (volatile value_type& dst, volatile value_type const& src) const {
    dst.max_nits = max(dst.max_nits, src.max_nits);
    dst.sum_nits += src.sum_nits;
    dst.nfails += src.nfails;
  }
};

inline Int test_sphere_to_ref (const ConstVec3s::HostMirror& p,
                               const ConstIdxs::HostMirror& e) {
  TestSphereToRefKernel k(p, e);
  Info info;
  ko::parallel_reduce(k.n(), k, info);
  return info.nfails;
}
} // namespace test
} // namespace sqr
} // namespace siqk

#endif // INCLUDE_SIQK_SQR_HPP

// Unit tests.
#include <limits>

//#include "siqk.hpp"
namespace siqk {

#ifdef SLMM_MAIN
# define INSTANTIATE_PLANE
#endif

} // namespace siqk

//> end SIQK

#endif
