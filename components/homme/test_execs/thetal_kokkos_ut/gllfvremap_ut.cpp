#include "GllFvRemapImpl.hpp"

#include "Types.hpp"
#include "Context.hpp"
#include "mpi/Comm.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "TimeLevel.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "ReferenceElement.hpp"
#include "SphereOperators.hpp"
#include "ElementOps.hpp"
#include "profiling.hpp"
#include "ErrorDefs.hpp"
#include "VerticalRemapManager.hpp"

#include "utilities/TestUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <catch2/catch.hpp>
#include <random>

using namespace Homme;

extern int hommexx_catch2_argc;
extern char** hommexx_catch2_argv;

extern "C" {
  void init_gllfvremap_f90(int ne, const Real* hyai, const Real* hybi, const Real* hyam,
                           const Real* hybm, Real ps0, Real* dvv, Real* mp, int qsize,
                           bool is_sphere);
  void init_geometry_f90();
  void gfr_init_f90(int nf, int ftype);
  void gfr_init_hxx();
  void gfr_finish_f90();
  void run_gfr_test(int* nerr);
  void run_gfr_check_api(int* nerr);
  void limiter1_clip_and_sum_f90(int n, double* spheremp, double* qmin, double* qmax,
                                 double* dp, double* q);
  void calc_dp_fv_f90(int nf, double* ps, double* dp_fv);
  void gfr_dyn_to_fv_phys_f90(int nt, double* ps, double* phis, double* T, double* uv,
                              double* omega_p, double* q);
} // extern "C"

using CA1d = Kokkos::View<Real*,     Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA2d = Kokkos::View<Real**,    Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA4d = Kokkos::View<Real**** , Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA5d = Kokkos::View<Real*****, Kokkos::LayoutRight, Kokkos::HostSpace>;

template <typename V>
decltype(Kokkos::create_mirror_view(V())) cmv (const V& v) {
  return Kokkos::create_mirror_view(v);
}

template <typename V>
decltype(Kokkos::create_mirror_view(V())) cmvdc (const V& v) {
  const auto h = Kokkos::create_mirror_view(v);
  deep_copy(h, v);
  return h;
}

class Random {
  using rngalg = std::mt19937_64;
  using rpdf = std::uniform_real_distribution<Real>;
  using ipdf = std::uniform_int_distribution<int>;
  std::random_device rd;
  unsigned int seed;
  rngalg engine;
public:
  Random (unsigned int seed_ = Catch::rngSeed()) : seed(seed_ == 0 ? rd() : seed_), engine(seed) {}
  unsigned int gen_seed () { return seed; }
  Real urrng (const Real lo = 0, const Real hi = 1) { return rpdf(lo, hi)(engine); }
  int  uirng (const int lo, const int hi) { return ipdf(lo, hi)(engine); }
};

template <typename V>
void fill (Random& r, const V& a, const Real scale = 1,
           typename std::enable_if<V::rank == 3>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int s = 0; s < VECTOR_SIZE; ++s)
          am(i,j,k)[s] = scale*r.urrng(-1,1); 
  deep_copy(a, am);
}

template <typename V>
void fill (Random& r, const V& a,
           typename std::enable_if<V::rank == 4>::type* = 0) {
  const auto am = cmvdc(a);
  for (int i = 0; i < a.extent_int(0); ++i)
    for (int j = 0; j < a.extent_int(1); ++j)
      for (int k = 0; k < a.extent_int(2); ++k)
        for (int l = 0; l < a.extent_int(3); ++l)
          for (int s = 0; s < VECTOR_SIZE; ++s)
            am(i,j,k,l)[s] = r.urrng(-1,1); 
  deep_copy(a, am);
}

struct Session {
  int ne;
  bool is_sphere;
  HybridVCoord h;
  Random r;
  std::shared_ptr<Elements> e;
  int nelemd, qsize, nlev, np;
  FunctorsBuffersManager fbm;

  //Session () : r(269041989) {}

  void init () {
    const auto seed = r.gen_seed();
    printf("seed %u\n", seed);

    parse_command_line();
    assert(is_sphere); // planar isn't available in Hxx yet

    auto& c = Context::singleton();

    c.create<HybridVCoord>().random_init(seed);
    h = c.get<HybridVCoord>();

    auto& p = c.create<SimulationParams>();
    p.qsize = qsize;
    p.hypervis_scaling = 0;
    p.transport_alg = 0;
    p.params_set = true;

    const auto hyai = cmvdc(h.hybrid_ai);
    const auto hybi = cmvdc(h.hybrid_bi);
    const auto hyam = cmvdc(h.hybrid_am);
    const auto hybm = cmvdc(h.hybrid_bm);
    auto& ref_FE = c.create<ReferenceElement>();
    std::vector<Real> dvv(NP*NP), mp(NP*NP);
    init_gllfvremap_f90(ne, hyai.data(), hybi.data(), &hyam(0)[0], &hybm(0)[0], h.ps0,
                        dvv.data(), mp.data(), qsize, is_sphere);
    ref_FE.init_mass(mp.data());
    ref_FE.init_deriv(dvv.data());

    nelemd = c.get<Connectivity>().get_num_local_elements();
    auto& bmm = c.create<MpiBuffersManagerMap>();
    bmm.set_connectivity(c.get_ptr<Connectivity>());
    e = c.get_ptr<Elements>();
    c.create<TimeLevel>();

    init_geometry_f90();    
    auto& geo = c.get<ElementsGeometry>();

    auto& sphop = c.create<SphereOperators>();
    sphop.setup(geo, ref_FE);

    auto& gfr = c.create<GllFvRemap>();
    gfr.reset(p);
    fbm.request_size(gfr.requested_buffer_size());
    fbm.allocate();
    gfr.init_buffers(fbm);
    gfr.init_boundary_exchanges();

    nlev = NUM_PHYSICAL_LEV;
    assert(nlev > 0);
    np = NP;
    assert(np == 4);
  }

  void cleanup () {
    auto& c = Context::singleton();
    c.finalize_singleton();
  }

  static Session& singleton () {
    if ( ! s_session) {
      s_session = std::make_shared<Session>();
      s_session->init();
    }
    return *s_session;
  }

  // Call only in last line of last TEST_CASE.
  static void delete_singleton () {
    if (s_session) s_session->cleanup();
    s_session = nullptr;
  }

  Comm& get_comm () const { return Context::singleton().get<Comm>(); }

private:
  static std::shared_ptr<Session> s_session;

  // compose_ut hommexx -ne NE -qsize QSIZE
  void parse_command_line () {
    const bool am_root = get_comm().root();
    ne = 2;
    qsize = QSIZE_D;
    is_sphere = true;
    bool ok = true;
    int i;
    for (i = 0; i < hommexx_catch2_argc; ++i) {
      const std::string tok(hommexx_catch2_argv[i]);
      if (tok == "-ne") {
        if (i+1 == hommexx_catch2_argc) { ok = false; break; }
        ne = std::atoi(hommexx_catch2_argv[++i]);
      } else if (tok == "-qsize") {
        if (i+1 == hommexx_catch2_argc) { ok = false; break; }
        qsize = std::atoi(hommexx_catch2_argv[++i]);
      } else if (tok == "-planar") {
        is_sphere = false;
      }
    }
    ne = std::max(2, std::min(128, ne));
    qsize = std::max(1, std::min(QSIZE_D, qsize));
    if ( ! ok && am_root)
      printf("gllfvremap_ut> Failed to parse command line, starting with: %s\n",
             hommexx_catch2_argv[i]);
    if (am_root)
      printf("gllfvremap_ut> ne %d qsize %d\n", ne, qsize);
  }
};

std::shared_ptr<Session> Session::s_session;

static bool almost_equal (const Real& a, const Real& b,
                          const Real tol = 0) {
  const auto re = std::abs(a-b)/(1 + std::abs(a));
  const bool good = re <= tol;
  if ( ! good)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e tol %9.2e\n",
           a, b, re, tol);
  return good;
}

static bool equal (const Real& a, const Real& b,
                   // Used only if not defined HOMMEXX_BFB_TESTING.
                   const Real tol = 0) {
#ifdef HOMMEXX_BFB_TESTING
  if (a != b)
    printf("equal: a,b = %23.16e %23.16e re = %23.16e\n",
           a, b, std::abs((a-b)/a));
  return a == b;
#else
  return almost_equal(a, b, tol);
#endif
}

typedef ExecViewUnmanaged<Real*[NP][NP][NUM_LEV*VECTOR_SIZE]> RNlev;
typedef ExecViewUnmanaged<Real**[NP][NP][NUM_LEV*VECTOR_SIZE]> RsNlev;
typedef ExecViewUnmanaged<Real***[NP][NP][NUM_LEV*VECTOR_SIZE]> RssNlev;
typedef HostView<Real*[NP][NP][NUM_LEV*VECTOR_SIZE]> RNlevH;
typedef HostView<Real**[NP][NP][NUM_LEV*VECTOR_SIZE]> RsNlevH;
typedef HostView<Real***[NP][NP][NUM_LEV*VECTOR_SIZE]> RssNlevH;

static void test_calc_dp_fv (Random& r, const HybridVCoord& hvcoord) {
  using Kokkos::deep_copy;
  using g = GllFvRemapImpl;
  
  const int nf = 3, ncol = nf*nf;
  
  CA1d ps_f90("ps", ncol);
  CA2d dp_fv_f90("dp_fv", g::num_phys_lev, ncol);
  for (int i = 0; i < ncol; ++i) ps_f90(i) = r.urrng(0.9e5, 1.05e5);
  calc_dp_fv_f90(nf, ps_f90.data(), dp_fv_f90.data());

  const ExecView<Real*> ps_d("ps", ncol);
  const ExecView<Scalar**> dp_fv_p("dp_fv", ncol, g::num_lev_pack);
  const auto ps_h = cmv(ps_d);
  for (int i = 0; i < ncol; ++i) ps_h(i) = ps_f90(i);
  deep_copy(ps_d, ps_h);
  Kokkos::parallel_for(
    Homme::get_default_team_policy<ExecSpace>(1),
    KOKKOS_LAMBDA (const g::MT& team) {
      g::calc_dp_fv(team, hvcoord, ncol, g::num_lev_pack, ps_d, dp_fv_p);
    });
  const ExecViewUnmanaged<Real**> dp_fv_d(g::pack2real(dp_fv_p), ncol, g::num_lev_aligned);
  const auto dp_fv_h = cmvdc(dp_fv_d);
  
  for (int i = 0; i < ncol; ++i)
    for (int k = 0; k < g::num_phys_lev; ++k)
      REQUIRE(dp_fv_f90(k,i) == dp_fv_d(i,k));
}

static void sfwd_remapd (const int m, const int n,
                         const Real* A, const Real* d1, const Real* d2,
                         const Real* x, Real* y) {
  for (int i = 0; i < m; ++i) {
    y[i] = 0;
    for (int j = 0; j < n; ++j)
      y[i] += A[n*i + j] * (x[j] * d1[j]);
    y[i] /= d2[i];
  }
}

static void sfwd_remapd (const int m, const int n,
                         const Real* A, const Real* d1, const Real* d2,
                         const Real* Dinv, const Real* D,
                         const Real* x, Real* wx, Real* wy, Real* y) {
  for (int d = 0; d < 2; ++d) {
    const int os = d*n;
    for (int i = 0; i < n; ++i)
      wx[os+i] = Dinv[4*i+2*d]*x[i] + Dinv[4*i+2*d+1]*x[n+i];
  }
  for (int d = 0; d < 2; ++d)
    sfwd_remapd(m, n, A, d1, d2, wx + d*n, wy + d*m);
  for (int d = 0; d < 2; ++d) {
    const int os = d*m;
    for (int i = 0; i < m; ++i)
      y[os+i] = D[4*i+2*d]*wy[i] + D[4*i+2*d+1]*wy[m+i];
  }  
}

// Comparison of straightforwardly computed remapd vs GllFvRemapImpl version.
static void test_remapds (Random& r, const int m, const int n, const int nlev) {
  using Kokkos::deep_copy;
  using g = GllFvRemapImpl;

  // Random data for both scalar and vector remapd.

  std::vector<Real> A(m*n), d1(n), d2(m), x(2*n), y(2*m), Dinv(4*n), D(4*m);
  for (int i = 0; i < m*n; ++i) A [i] = r.urrng(0.1, 1.3);
  for (int i = 0; i <   n; ++i) d1[i] = r.urrng(0.2, 1.2);
  for (int i = 0; i < m  ; ++i) d2[i] = r.urrng(0.3, 1.1);
  for (int i = 0; i < 2*n; ++i) x [i] = r.urrng(0.4, 0.9);
  for (int i = 0; i < 4*n; ++i) Dinv[i] = r.urrng(0.5, 1.1);
  for (int i = 0; i < 4*m; ++i) D   [i] = r.urrng(0.6, 1.2);

  const int nlevpk = (nlev + g::packn - 1)/g::packn;
  const int nlevsk = nlevpk*g::packn;
  // Size arrays larger than needed because we want to support that case.
  const ExecView<Real*[2][2]> Dinv_d("Dinv", n), D_d("D", m);
  const ExecView<Real**> A_d("A", m+1, n+2);
  const ExecView<Real*> d1_d("d1", n+1), d2_d("d2", m+3);
  const ExecView<Scalar**> x_p("x", n+1, nlevpk), y_p("y", m+2, nlevpk);
  const ExecView<Real**> x_d(g::pack2real(x_p), n+1, nlevsk),
    y_d(g::pack2real(y_p), m+2, nlevsk);
  const auto Dinv_h = cmv(Dinv_d), D_h = cmv(D_d);
  const auto A_h = cmv(A_d);
  const auto d1_h = cmv(d1_d), d2_h = cmv(d2_d);
  const auto x_h = cmv(x_d), y_h = cmv(y_d);
    for (int d1 = 0; d1 < 2; ++d1)
      for (int d2 = 0; d2 < 2; ++d2) {
        for (int i = 0; i < n; ++i) Dinv_h(i,d1,d2) = Dinv[4*i + 2*d1 + d2];
        for (int i = 0; i < m; ++i) D_h   (i,d1,d2) = D   [4*i + 2*d1 + d2];
      }
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      A_h(i,j) = A[n*i + j];
  for (int i = 0; i < n; ++i) d1_h(i) = d1[i];
  for (int i = 0; i < m; ++i) d2_h(i) = d2[i];
  for (int k = 0; k < nlev; ++k)
    for (int i = 0; i < n; ++i)
      x_h(i,k) = x[i];
  deep_copy(Dinv_d, Dinv_h); deep_copy(D_d, D_h);
  deep_copy(A_d, A_h); deep_copy(d1_d, d1_h); deep_copy(d2_d, d2_h);
  deep_copy(x_d, x_h);

  // Scalar remapd.
  sfwd_remapd(m, n, A.data(), d1.data(), d2.data(), x.data(), y.data());
  Kokkos::parallel_for(
    Homme::get_default_team_policy<ExecSpace>(1),
    KOKKOS_LAMBDA (const g::MT& team) {
      g::remapd(team, m, n, nlevpk, A_d, d1_d, d2_d, x_p, x_p, y_p); });
  deep_copy(x_h, x_d); deep_copy(y_h, y_d);
  for (int k = 0; k < nlev; ++k) {
    for (int i = 0; i < n; ++i) REQUIRE(x_h(i,0) == x[i]*d1[i]);
    for (int i = 0; i < m; ++i) REQUIRE(y_h(i,k) == y[i]);
  }

  // Vector remapd.
  std::vector<Real> wx(2*n), wy(2*m);
  sfwd_remapd(m, n, A.data(), d1.data(), d2.data(), Dinv.data(), D.data(),
              x.data(), wx.data(), wy.data(), y.data());
  { // x(i,d), y(d,i)
    const ExecView<Scalar***> x2_p("x2", n+1, 2, nlevpk), y2_p("y", 2, m+2, nlevpk);
    const ExecView<Real***> x2_d(g::pack2real(x2_p), n+1, 2, nlevsk),
      y2_d(g::pack2real(y2_p), 2, m+2, nlevsk);
    const auto x2_h = cmv(x2_d), y2_h = cmv(y2_d);
    for (int k = 0; k < nlev; ++k)
      for (int i = 0; i < n; ++i)
        for (int d = 0; d < 2; ++d)
          x2_h(i,d,k) = x[d*n+i];
    deep_copy(x2_d, x2_h);
    Kokkos::parallel_for(
      Homme::get_default_team_policy<ExecSpace>(1),
      KOKKOS_LAMBDA (const g::MT& team) {
        g::remapd<true>(team, m, n, nlevpk, A_d, d1_d, d2_d, Dinv_d, D_d,
                        x2_p, x2_p, y2_p); });
    deep_copy(y2_h, y2_d);
    for (int k = 0; k < nlev; ++k)
      for (int i = 0; i < m; ++i)
        for (int d = 0; d < 2; ++d)
          REQUIRE(y2_h(d,i,k) == y[d*m+i]);
  }
  { // x(d,i), y(i,d)
    const ExecView<Scalar***> x2_p("x2", 2, n+1, nlevpk), y2_p("y", m+2, 2, nlevpk);
    const ExecView<Real***> x2_d(g::pack2real(x2_p), 2, n+1, nlevsk),
      y2_d(g::pack2real(y2_p), m+2, 2, nlevsk);
    const auto x2_h = cmv(x2_d), y2_h = cmv(y2_d);
    for (int k = 0; k < nlev; ++k)
      for (int i = 0; i < n; ++i)
        for (int d = 0; d < 2; ++d)
          x2_h(d,i,k) = x[d*n+i];
    deep_copy(x2_d, x2_h);
    Kokkos::parallel_for(
      Homme::get_default_team_policy<ExecSpace>(1),
      KOKKOS_LAMBDA (const g::MT& team) {
        g::remapd<false>(team, m, n, nlevpk, A_d, d1_d, d2_d, Dinv_d, D_d,
                         x2_p, x2_p, y2_p); });
    deep_copy(y2_h, y2_d);
    for (int k = 0; k < nlev; ++k)
      for (int i = 0; i < m; ++i)
        for (int d = 0; d < 2; ++d)
          REQUIRE(y2_h(i,d,k) == y[d*m+i]);
  }
}

template <typename V1, typename V2, typename V2q>
static void
assert_limiter_properties (const int n, const int nlev, const V1& spheremp,
                           const V1& qmin, const V1& qmax,
                           const V2& dp, const V2& qorig, const V2q& q,
                           const bool too_tight) {
  static const auto eps = std::numeric_limits<Real>::epsilon();
  const int n2 = n*n;
  int noteq = 0;
  for (int k = 0; k < nlev; ++k) {
    Real qm0 = 0, qm = 0;
    for (int i = 0; i < n2; ++i) {
      REQUIRE(q(i,k) >= (1 - 1e1*eps)*qmin(k));
      REQUIRE(q(i,k) <= (1 + 1e1*eps)*qmax(k));
      qm0 += spheremp(i)*dp(i,k)*qorig(i,k);
      qm  += spheremp(i)*dp(i,k)*q(i,k);
      if (q(i,k) != qorig(i,k)) ++noteq;
    }
    REQUIRE(almost_equal(qm0, qm, 1e2*eps));
    if (too_tight && k % 2 == 1)
      for (int i = 1; i < n2; ++i)
        REQUIRE(almost_equal(q(i,k), q(0,k), 1e2*eps));
  }
  REQUIRE(noteq > 0);
}

static void test_limiter (const int nlev, const int n, Random& r, const bool too_tight) {
  using Kokkos::deep_copy;
  using g = GllFvRemapImpl;

  const int n2 = n*n;
  const int nlevpk = (nlev + g::packn - 1)/g::packn;
  const int nlevsk = nlevpk*g::packn;

  const ExecView<Real*> spheremp_d("spheremp", n2);
  const ExecView<Scalar*> qmin_p("qmin", nlevpk), qmax_p("qmax", nlevpk);
  const ExecView<Scalar**> dp_p("dp", n2, nlevpk), qorig_p("qorig", n2, nlevpk),
    q_p("q", n2, nlevpk), wrk_p("wrk", n2, nlevpk);
  const ExecView<Real*> qmin_d(g::pack2real(qmin_p), nlevsk),
    qmax_d(g::pack2real(qmax_p), nlevsk);
  const ExecView<Real**> dp_d(g::pack2real(dp_p), n2, nlevsk),
    qorig_d(g::pack2real(qorig_p), n2, nlevsk), q_d(g::pack2real(q_p), n2, nlevsk);

  const auto spheremp = cmv(spheremp_d);
  const auto qmin = cmv(qmin_d), qmax = cmv(qmax_d);
  const auto dp = cmv(dp_d), qorig = cmv(qorig_d), q = cmv(q_d);

  for (int k = 0; k < nlev; ++k) {
    Real mass = 0, qmass = 0;
    qmin(k) = 1; qmax(k) = 0;
    for (int i = 0; i < n2; ++i) {
      if (k == 0) spheremp(i) = r.urrng(0.5, 1.5);
      dp(i,k) = r.urrng(0.5, 1.5);
      qorig(i,k) = q(i,k) = r.urrng(1e-2, 3e-2);
      mass += spheremp(i)*dp(i,k);
      qmass += spheremp(i)*dp(i,k)*q(i,k);
      qmin(k) = std::min(qmin(k), q(i,k));
      qmax(k) = std::max(qmax(k), q(i,k));
    }
    const auto q0 = qmass/mass;
    if (too_tight && k % 2 == 1) {
      qmin(k) = 0.5*(q0 + qmin(k));
      qmax(k) = 0.1*qmin(k) + 0.9*q0;
    } else {
      qmin(k) = 0.5*(q0 + qmin(k));
      qmax(k) = 0.5*(q0 + qmax(k));
    }
  }

  deep_copy(spheremp_d, spheremp);
  deep_copy(qmin_d, qmin); deep_copy(qmax_d, qmax);
  deep_copy(dp_d, dp); deep_copy(qorig_d, qorig); deep_copy(q_d, q);

  // F90 limiter properties.
  CA2d qf90("q", n2, nlevsk);
  deep_copy(qf90, qorig);
  CA1d dpk("dpk", n2), qk("qk", n2);
  for (int k = 0; k < nlev; ++k) {
    for (int i = 0; i < n2; ++i) dpk(i) = dp(i,k);
    for (int i = 0; i < n2; ++i) qk(i) = qf90(i,k);
    limiter1_clip_and_sum_f90(n, spheremp.data(), &qmin(k), &qmax(k), dpk.data(),
                              qk.data());
    for (int i = 0; i < n2; ++i) qf90(i,k) = qk(i);
  }
  assert_limiter_properties(n, nlev, spheremp, qmin, qmax, dp, qorig, qf90, too_tight);

  // C++ limiter properties.
  Kokkos::parallel_for(
    Homme::get_default_team_policy<ExecSpace>(1),
    KOKKOS_LAMBDA (const g::MT& team) {
      g::limiter_clip_and_sum(team, n2, nlevpk, spheremp_d, qmin_p, qmax_p, dp_p,
                              wrk_p, q_p); });
  deep_copy(q, q_d);
  assert_limiter_properties(n, nlev, spheremp, qmin, qmax, dp, qorig, q, too_tight);

  // BFB C++ vs F90.
  for (int k = 0; k < nlev; ++k)
    for (int i = 0; i < n2; ++i)
      REQUIRE(equal(qf90(i,k), q(i,k)));
}

static void test_dyn_to_fv_phys (const int nf, const int ftype) {
  const int nt = 1; // time index

  gfr_init_f90(nf, ftype);
  gfr_init_hxx();
#if 0
  CA1d fps("ps", ncol);
    
  gfr_dyn_to_fv_phys_f90(nt, fps.data(), fphis.data(), fT.data(), fuv.data(),
                         fomega.data(), fq.data());
#endif
  gfr_finish_f90();
}

TEST_CASE ("compose_transport_testing") {
  static constexpr Real tol = std::numeric_limits<Real>::epsilon();

  auto& s = Session::singleton(); try {
    // calc_dp_fv BFB.
    test_calc_dp_fv(s.r, s.h);
    
    // Core scalar and vector remapd routines.
    test_remapds(s.r, 7, 11, 13);
    test_remapds(s.r, 11, 7, 13);
    test_remapds(s.r, 16, 4, 8 );
    test_remapds(s.r, 4, 16, 8 );

    // Limiter.
    for (const auto too_tight : {false, true}) {
      test_limiter(16, 7, s.r, too_tight);
      test_limiter(16, 4, s.r, too_tight);
    }

    // Existing F90 gllfvremap unit tests.
    int nerr;
    run_gfr_test(&nerr);
    REQUIRE(nerr == 0);
    run_gfr_test(&nerr);
    REQUIRE(nerr == 0);

    for (const int nf : {2,3,4})
      for (const int ftype : {0,2})
        test_dyn_to_fv_phys(nf, ftype);
  } catch (...) {}
  Session::delete_singleton();
}
