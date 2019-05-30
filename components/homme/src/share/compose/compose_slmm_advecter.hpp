#ifndef INCLUDE_COMPOSE_SLMM_ADVECTER_HPP
#define INCLUDE_COMPOSE_SLMM_ADVECTER_HPP

#include "compose_slmm.hpp"
#include "compose_slmm_siqk.hpp"
#include "compose_slmm_departure_point.hpp"

#include <limits>
#include <memory>

namespace slmm {
typedef Kokkos::View<Int*, ko::HostSpace> Ints;

//TODO We might switch to one 
// Local mesh patch centered on the target element.
struct LocalMesh : public siqk::Mesh<ko::HostSpace> {
  typedef siqk::Mesh<ko::HostSpace> Super;
  typedef typename Super::IntArray IntArray;
  typedef typename Super::RealArray RealArray;

  // tgt_elem is the index of the target element in this mesh.
  Int tgt_elem;
};

// Wrap call to siqk::sqr::calc_sphere_to_ref. That impl supports only
// cubed_sphere_map=2, and we want to keep it that way. This wrapper supports,
// in addition, cubed_sphere_map=0.
struct SphereToRef {
  void init (const Int cubed_sphere_map, const Int nelem_global,
             const Ints::const_type& lid2facenum) {
    cubed_sphere_map_ = cubed_sphere_map;
    ne_ = static_cast<Int>(std::round(std::sqrt((nelem_global / 6))));
    slmm_throw_if( ! (cubed_sphere_map_ != 0 || 6*ne_*ne_ == nelem_global),
                  "If cubed_sphere_map = 0, then the mesh must be a "
                  "regular cubed-sphere.");
    lid2facenum_ = lid2facenum;
  }

  Real tol () const {
    return 1e3 * ne_ * std::numeric_limits<Real>::epsilon();
  }

  // See siqk::sqr::calc_sphere_to_ref for docs.
  void calc_sphere_to_ref (
    const Int& ie, const LocalMesh& m,
    const Real q[3],
    Real& a, Real& b,
    siqk::sqr::Info* const info = nullptr,
    const Int max_its = 10,
    const Real tol = 1e2*std::numeric_limits<Real>::epsilon()) const
  {
    if (cubed_sphere_map_ == 2)
      siqk::sqr::calc_sphere_to_ref(m.p, slice(m.e, m.tgt_elem), q, a, b,
                                    info, max_its, tol);
    else {
      const Int face = lid2facenum_(ie); //assume: ie corresponds to m.tgt_elem.
      map_sphere_coord_to_face_coord(face-1, q[0], q[1], q[2], a, b);
      a = map_face_coord_to_cell_ref_coord(a);
      b = map_face_coord_to_cell_ref_coord(b);
      if (info) { info->success = true; info->n_iterations = 1; }
    }
  }

  bool check (const Int& ie, const LocalMesh& m) const {
    if (cubed_sphere_map_ != 0) return true;
    const Int face = lid2facenum_(ie); //assume: ie corresponds to m.tgt_elem.
    Real cent[3] = {0};
    const auto cell = slice(m.e, m.tgt_elem);
    for (int i = 0; i < 4; ++i)
      for (int d = 0; d < 3; ++d)
        cent[d] += 0.25*m.p(cell[i], d);
    const Int cf = get_cube_face_idx(cent[0], cent[1], cent[2]) + 1;
    return face == cf;
  }

private:
  Int ne_, cubed_sphere_map_;
  Ints::const_type lid2facenum_;

  // Follow the description given in
  //     coordinate_systems_mod::unit_face_based_cube_to_unit_sphere.
  static Int get_cube_face_idx (const Real& x, const Real& y, const Real& z) {
    const Real ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
    if (ax >= ay) {
      if (ax >= az) return x > 0 ? 0 : 2;
      else return z > 0 ? 5 : 4;
    } else {
      if (ay >= az) return y > 0 ? 1 : 3;
      else return z > 0 ? 5 : 4;
    }
  }

  static void map_sphere_coord_to_face_coord (
    const Int& face_idx, const Real& x, const Real& y, const Real& z,
    Real& fx, Real& fy)
  {
    static constexpr Real theta_max = 0.25*M_PI;
    Real d;
    switch (face_idx) {
    case  0: d = std::abs(x); fx =  y/d; fy =  z/d; break;
    case  1: d = std::abs(y); fx = -x/d; fy =  z/d; break;
    case  2: d = std::abs(x); fx = -y/d; fy =  z/d; break;
    case  3: d = std::abs(y); fx =  x/d; fy =  z/d; break;
    case  4: d = std::abs(z); fx =  y/d; fy =  x/d; break;
    default: d = std::abs(z); fx =  y/d; fy = -x/d;
    }
    fx = std::atan(fx) / theta_max;
    fy = std::atan(fy) / theta_max;
  }

  Real map_face_coord_to_cell_ref_coord (Real a) const {
    a = (0.5*(1 + a))*ne_;
    a = 2*(a - std::floor(a)) - 1;
    return a;
  }
};

// Advecter has purely mesh-local knowledge, with once exception noted below.
class Advecter {
  typedef nearest_point::MeshNearestPointData<ko::HostSpace> MeshNearestPointData;

public:
  typedef std::shared_ptr<Advecter> Ptr;
  typedef std::shared_ptr<const Advecter> ConstPtr;

  struct Alg {
    enum Enum {
      jct,             // Cell-integrated Jacobian-combined transport.
      qos,             // Cell-integrated quadrature-on-sphere transport.
      csl_gll,         // Intended to mimic original Fortran CSL.
      csl_gll_subgrid, // Stable np=4 subgrid bilinear interp.
      csl_gll_exp,     // Stabilized np=4 subgrid reconstruction.
    };
    static Enum convert (Int alg) {
      switch (alg) {
      case 2: case 29:  return jct;
      case 3: case 39:  return qos;
      case 10: case 18: return csl_gll;
      case 11: case 17: return csl_gll_subgrid;
      case 12: case 19: return csl_gll_exp;
      default: slmm_throw_if(true, "transport_alg " << alg << " is invalid.");
      }
    }
    static bool is_cisl (const Enum& alg) { return alg == jct || alg == qos; }
  };

  Advecter (const Int np, const Int nelem, const Int transport_alg,
            const Int cubed_sphere_map, const Int nearest_point_permitted_lev_bdy)
    : alg_(Alg::convert(transport_alg)),
      np_(np), np2_(np*np), np4_(np2_*np2_),
      cubed_sphere_map_(cubed_sphere_map),
      tq_order_(alg_ == Alg::qos ? 14 : 12),
      nearest_point_permitted_lev_bdy_(nearest_point_permitted_lev_bdy)
  {
    slmm_throw_if(cubed_sphere_map == 0 && Alg::is_cisl(alg_),
                  "When cubed_sphere_map = 0, SLMM supports only ISL methods.");
    local_mesh_.resize(nelem);
    if (Alg::is_cisl(alg_))
      mass_mix_.resize(np4_);
    if (nearest_point_permitted_lev_bdy_ >= 0)
      local_mesh_nearest_point_data_.resize(nelem);
  }

  Int np  () const { return np_ ; }
  Int np2 () const { return np2_; }
  Int np4 () const { return np4_; }
  Int nelem () const { return local_mesh_.size(); }
  Int tq_order () const { return tq_order_; }
  Alg::Enum alg () const { return alg_; }
  bool is_cisl () const { return Alg::is_cisl(alg_); }

  Int cubed_sphere_map () const { return cubed_sphere_map_; }
  const Ints& lid2facenum () const { return lid2facenum_; }

  // nelem_global is used only if cubed_sphere_map = 0, to deduce ne in
  // nelem_global = 6 ne^2. That is b/c cubed_sphere_map = 0 is supported in
  // Homme only for regular meshes (not RRM), and ne is essential to using the
  // efficiency it provides.
  void init_meta_data(const Int nelem_global, const Int* lid2facenum);

  template <typename Array3D>
  void init_local_mesh_if_needed(const Int ie, const Array3D& corners,
                                 const Real* p_inside);

  // Check that our ref <-> sphere map agrees with Homme's. p_homme is a GLL
  // point on the sphere. Check that we map it to a GLL ref point.
  void check_ref2sphere(const Int ie, const Real* p_homme);

  const LocalMesh& local_mesh (const Int ie) const {
    slmm_assert(ie < static_cast<Int>(local_mesh_.size()));
    return local_mesh_[ie];
  }
  LocalMesh& local_mesh (const Int ie) {
    slmm_assert(ie < static_cast<Int>(local_mesh_.size()));
    return local_mesh_[ie];
  }

  const MeshNearestPointData& nearest_point_data (const Int ie) const {
    slmm_assert(ie < static_cast<Int>(local_mesh_nearest_point_data_.size()));
    return local_mesh_nearest_point_data_[ie];
  }

  std::vector<Real>& rhs_buffer (const Int qsize) {
    rhs_.resize(np2_*qsize);
    return rhs_;
  }

  std::vector<Real>& mass_mix_buffer () { return mass_mix_; }

  const Real* M_tgt (const Int& ie) {
    slmm_assert(ie >= 0 && ie < nelem());
    return alg_ == Alg::jct ?
      mass_tgt_.data() :
      mass_tgt_.data() + ie*np4_;
  }

  bool nearest_point_permitted (const Int& lev) const {
    return lev <= nearest_point_permitted_lev_bdy_;
  }

  const SphereToRef& s2r () const { return s2r_; }

private:
  const Alg::Enum alg_;
  const Int np_, np2_, np4_, cubed_sphere_map_;
  std::vector<LocalMesh> local_mesh_;
  // For CISL:
  const Int tq_order_;
  std::vector<Real> mass_tgt_, mass_mix_, rhs_;
  // For recovery from get_src_cell failure:
  Int nearest_point_permitted_lev_bdy_;
  std::vector<MeshNearestPointData> local_mesh_nearest_point_data_;
  // Meta data obtained at initialization that can be used later.
  Ints lid2facenum_;
  SphereToRef s2r_;
};

template <typename Array3D>
void Advecter::init_local_mesh_if_needed (const Int ie, const Array3D& corners,
                                          const Real* p_inside) {
  slmm_assert(ie < static_cast<Int>(local_mesh_.size()));
  if (local_mesh_[ie].p.extent_int(0) != 0) return;
  auto& m = local_mesh_[ie];
  const Int
    nd = 3,
    nvert = corners.extent_int(1),
    ncell = corners.extent_int(2),
    N = nvert*ncell;
  m.p = typename LocalMesh::RealArray("p", N);
  m.e = typename LocalMesh::IntArray("e", ncell, nvert);
  for (Int ci = 0, k = 0; ci < ncell; ++ci)
    for (Int vi = 0; vi < nvert; ++vi, ++k) {
      for (int j = 0; j < nd; ++j)
        m.p(k,j) = corners(j,vi,ci);
      m.e(ci,vi) = k;
    }
  siqk::test::fill_normals<siqk::SphereGeometry>(m);
  m.tgt_elem = slmm::get_src_cell(m, p_inside);
  slmm_assert(m.tgt_elem >= 0 &&
              m.tgt_elem < ncell);
  if (nearest_point_permitted_lev_bdy_ >= 0)
    nearest_point::fill_perim(local_mesh_[ie],
                              local_mesh_nearest_point_data_[ie]);
}

}

#endif
