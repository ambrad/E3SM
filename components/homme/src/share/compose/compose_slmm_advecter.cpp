#include "compose_slmm_advecter.hpp"

namespace slmm {

template <typename MT>
void Advecter<MT>
::init_meta_data (const Int nelem_global, const Int* lid2facenum) {
  const auto nelemd = local_mesh_h_.extent_int(0);
  lid2facenum_ = Ints<DES>("Advecter::lid2facenum", nelemd);
  std::copy(lid2facenum, lid2facenum + nelemd, lid2facenum_.data());
  s2r_.init(cubed_sphere_map_, nelem_global, lid2facenum_);
}

template <typename MT>
void Advecter<MT>::check_ref2sphere (const Int ie, const Real* p_homme) {
  const auto& m = local_mesh_host(ie);
  Real ref_coord[2];
  siqk::sqr::Info info;
  const Real tol = s2r_.tol();
  s2r_.calc_sphere_to_ref(ie, m, p_homme, ref_coord[0], ref_coord[1], &info);
  const slmm::Basis basis(4, 0);
  const slmm::GLL gll;
  const Real* x, * wt;
  gll.get_coef(basis, x, wt);
  int fnd[2] = {0};
  Real min[2] = {1,1};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 2; ++j) {
      const Real d = std::abs(ref_coord[j] - x[i]);
      min[j] = std::min(min[j], d);
      if (d < tol)
        fnd[j] = 1;
    }
  if ( ! fnd[0] || ! fnd[1])
    printf("COMPOSE check_ref2sphere: %1.15e %1.15e (%1.2e %1.2e) %d %d\n",
           ref_coord[0], ref_coord[1], min[0], min[1],
           info.success, info.n_iterations);
  if ( ! s2r_.check(ie, m))
    printf("COMPOSE SphereToRef::check return false: ie = %d\n", ie);
}

template <typename MT>
void deep_copy (typename Advecter<MT>::LocalMeshesD& d,
                const typename Advecter<MT>::LocalMeshesH& s) {
#ifdef COMPOSE_PORT
  const Int nlm = s.extent_int(0);
  d = typename Advecter<MT>::LocalMeshesD("LocalMeshes", nlm);
  const auto m = ko::create_mirror_view(d);
  for (Int i = 0; i < nlm; ++i)
    deep_copy(m(i), s(i));
  ko::deep_copy(d, m);
#endif
}

template <typename MT>
void Advecter<MT>::sync_to_device() {
  if (slmm::OnGpu<typename MT::DES>::value)
    deep_copy<MT>(local_mesh_d_, local_mesh_h_);
  else
    local_mesh_d_ = local_mesh_h_;
}

template class Advecter<>;

} // namespace slmm
