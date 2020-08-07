#include "/home/ambrad/repo/sik/hommexx/dbg.hpp"

#include "ComposeTransport.hpp"
#include "compose_test.hpp"

#include "Types.hpp"
#include "Context.hpp"
#include "mpi/Comm.hpp"
#include "mpi/Connectivity.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "ElementOps.hpp"
#include "profiling.hpp"
#include "ErrorDefs.hpp"

#include "utilities/TestUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/ViewUtils.hpp"

#include <catch2/catch.hpp>
#include <random>

using namespace Homme;

extern "C" {
  void init_compose_f90(int ne, const Real* hyai, const Real* hybi,
                        const Real* hyam, const Real* hybm, Real ps0);
  void cleanup_compose_f90();
  void run_compose_standalone_test_f90();
} // extern "C"

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

struct Session {
  static const int ne = 4;
  HybridVCoord h;
  Random r;
  Elements e;
  int nelemd, qsize;

  //Session () : r(269041989) {}

  void init () {
    printf("seed %u\n", r.gen_seed());

    qsize = 5;

    auto& c = Context::singleton();

    c.create<HybridVCoord>().random_init(r.gen_seed());
    h = c.get<HybridVCoord>();

    const auto hyai = cmvdc(h.hybrid_ai);
    const auto hybi = cmvdc(h.hybrid_bi);
    const auto hyam = cmvdc(h.hybrid_am);
    const auto hybm = cmvdc(h.hybrid_bm);

    init_compose_f90(ne, hyai.data(), hybi.data(), &hyam(0)[0], &hybm(0)[0], h.ps0);

    c.create<Elements>();
    e = c.get<Elements>();
    const int nelemd = c.get<Connectivity>().get_num_local_elements();
    c.create<SimulationParams>();
    c.create<Tracers>(nelemd, qsize);
  }

  void cleanup () {
    cleanup_compose_f90();
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

  MPI_Comm get_mpi_comm () const {
    return Context::singleton().get<Comm>().mpi_comm();
  }

private:
  static std::shared_ptr<Session> s_session;
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

TEST_CASE ("compose_transport_testing") {
  auto& s = Session::singleton();

  REQUIRE(compose::test::slmm_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest(s.get_mpi_comm()) == 0);

  ComposeTransport ct;
  const auto fails = ct.run_unit_tests();
  for (const auto& e : fails) printf("%s %d\n", e.first.c_str(), e.second);
  REQUIRE(fails.empty());

  run_compose_standalone_test_f90();

  Session::delete_singleton();
}
