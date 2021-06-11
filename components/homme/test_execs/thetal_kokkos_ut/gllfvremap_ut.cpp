#include "/home/ambrad/repo/sik/hommexx/dbg.hpp"

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
  void run_gfr_test(int* nerr);
  void run_gfr_check_api(int* nerr);
  void limiter1_clip_and_sum_f90(int n, double* spheremp, double* qmin, double* qmax,
                                 double* dp, double* q);
} // extern "C"

using CA1d = Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA2d = Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using CA4d = Kokkos::View<Real****, Kokkos::LayoutRight, Kokkos::HostSpace>;
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

#if 0
    auto& ct = c.create<ComposeTransport>();
    ct.reset(p);
    fbm.request_size(ct.requested_buffer_size());
    fbm.allocate();
    ct.init_buffers(fbm);
    ct.init_boundary_exchanges();
#endif

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

template <typename V1, typename V2, typename V2q>
static void
assert_limiter_properties (const int nlev, const int n, const V1& spheremp,
                           const V1& qmin, const V1& qmax,
                           const V2& dp, const V2& qorig, const V2q& q,
                           const bool too_tight) {
  static const auto eps = std::numeric_limits<Real>::epsilon();
  const int n2 = n*n;
  int noteq = 0;
  for (int k = 0; k < nlev; ++k) {
    Real qm0 = 0, qm = 0;
    for (int i = 0; i < n2; ++i) {
      REQUIRE(q(k,i) >= (1 - 1e1*eps)*qmin(k));
      REQUIRE(q(k,i) <= (1 + 1e1*eps)*qmax(k));
      qm0 += spheremp(i)*dp(k,i)*qorig(k,i);
      qm  += spheremp(i)*dp(k,i)*q(k,i);
      if (q(k,i) != qorig(k,i)) ++noteq;
    }
    REQUIRE(almost_equal(qm0, qm, 1e2*eps));
    if (too_tight && k % 2 == 1)
      for (int i = 1; i < n2; ++i)
        REQUIRE(almost_equal(q(k,i), q(k,0), 1e2*eps));
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
  const ExecView<Scalar**> dp_p("dp", nlevpk, n2), qorig_p("qorig", nlevpk, n2), q_p("q", nlevpk, n2);
  const ExecView<Real*> qmin_d(g::pack2real(qmin_p), nlevsk), qmax_d(g::pack2real(qmax_p), nlevsk);
  const ExecView<Real**> dp_d(g::pack2real(dp_p), nlevsk, n2),
    qorig_d(g::pack2real(qorig_p), nlevsk, n2), q_d(g::pack2real(q_p), nlevsk, n2);

  const auto spheremp = cmv(spheremp_d);
  const auto qmin = cmv(qmin_d), qmax = cmv(qmax_d);
  const auto dp = cmv(dp_d), qorig = cmv(qorig_d), q = cmv(q_d);

  for (int k = 0; k < nlev; ++k) {
    Real mass = 0, qmass = 0;
    qmin(k) = 1; qmax(k) = 0;
    for (int i = 0; i < n2; ++i) {
      if (k == 0) spheremp(i) = r.urrng(0.5, 1.5);
      dp(k,i) = r.urrng(0.5, 1.5);
      qorig(k,i) = q(k,i) = r.urrng(1e-2, 3e-2);
      mass += spheremp(i)*dp(k,i);
      qmass += spheremp(i)*dp(k,i)*q(k,i);
      qmin(k) = std::min(qmin(k), q(k,i));
      qmax(k) = std::max(qmax(k), q(k,i));
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

  CA2d qf90("q", nlevsk, n2);
  deep_copy(qf90, qorig);
  CA1d dpk("dpk", n2), qk("qk", n2);
  for (int k = 0; k < nlev; ++k) {
    for (int i = 0; i < n2; ++i) dpk(i) = dp(k,i);
    for (int i = 0; i < n2; ++i) qk(i) = qf90(k,i);
    limiter1_clip_and_sum_f90(n, spheremp.data(), &qmin(k), &qmax(k), dpk.data(), qk.data());
    for (int i = 0; i < n2; ++i) qf90(k,i) = qk(i);
  }
  assert_limiter_properties(nlev, n, spheremp, qmin, qmax, dp, qorig, qf90, too_tight);

  pr("now C++");
  const auto f = KOKKOS_LAMBDA (const g::MT& team) {
    g::limiter_clip_and_sum(nlev, n, spheremp_d, qmin_p, qmax_p, dp_p, q_p);
  };
  Kokkos::parallel_for(Homme::get_default_team_policy<ExecSpace>(1), f);
  assert_limiter_properties(nlev, n, spheremp, qmin, qmax, dp, qorig, q, too_tight);

  // BFB
}

TEST_CASE ("compose_transport_testing") {
  static constexpr Real tol = std::numeric_limits<Real>::epsilon();

  auto& s = Session::singleton(); try {
    int nerr;
    // Run existing F90 gllfvremap unit tests.
    run_gfr_test(&nerr);
    REQUIRE(nerr == 0);
    run_gfr_test(&nerr);
    REQUIRE(nerr == 0);

    for (const auto too_tight : {false, true})
      test_limiter(11, 7, s.r, too_tight);
  } catch (...) {}
  Session::delete_singleton();
}
