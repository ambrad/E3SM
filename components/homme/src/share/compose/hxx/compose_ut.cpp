#include "/home/ambrad/repo/sik/hommexx/dbg.hpp"

#include "ComposeTransport.hpp"
#include "compose_test.hpp"

#include "Types.hpp"
#include "Context.hpp"
#include "mpi/Comm.hpp"
#include "mpi/Connectivity.hpp"
#include "mpi/MpiBuffersManager.hpp"
#include "FunctorsBuffersManager.hpp"
#include "SimulationParams.hpp"
#include "Elements.hpp"
#include "Tracers.hpp"
#include "HybridVCoord.hpp"
#include "KernelVariables.hpp"
#include "PhysicalConstants.hpp"
#include "ReferenceElement.hpp"
#include "SphereOperators.hpp"
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
                        const Real* hyam, const Real* hybm, Real ps0,
                        Real* dvv, Real* mp);
  void cleanup_compose_f90();
  void run_compose_standalone_test_f90();
  void run_trajectory_f90(Real t0, Real t1, bool independent_time_steps, Real* dep);
} // extern "C"

using FA5d = Kokkos::View<Real*****, Kokkos::LayoutLeft, Kokkos::HostSpace>;

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

static void init_elems (int nelemd, Random& r, const HybridVCoord& hvcoord,
                        Elements& e) {
  using Kokkos::create_mirror_view;
  using Kokkos::deep_copy;
  using Kokkos::subview;

  const int nlev = NUM_PHYSICAL_LEV, np = NP;
  const auto all = Kokkos::ALL();

  e.init(nelemd, false, true);
  const auto max_pressure = 1000 + hvcoord.ps0;
  auto& geo = e.m_geometry;
  e.m_geometry.randomize(r.gen_seed());
  e.m_state.randomize(r.gen_seed(), max_pressure, hvcoord.ps0, hvcoord.hybrid_ai0,
                      geo.m_phis);
  e.m_derived.randomize(r.gen_seed(), 10);

  // We want mixed signs in w_i, v, and gradphis.
  for (int ie = 0; ie < e.num_elems(); ++ie)
    for (int i = 0; i < 3; ++i) {
      fill(r, subview(e.m_state.m_w_i,ie,i,all,all,all), 5 ); // Pa/s
      fill(r, subview(e.m_state.m_v,ie,i,0,all,all,all), 10); // m/s
      fill(r, subview(e.m_state.m_v,ie,i,1,all,all,all), 10);
    }
  const auto gpm = create_mirror_view(e.m_geometry.m_gradphis);
  for (int ie = 0; ie < e.num_elems(); ++ie)
    for (int d = 0; d < 2; ++d)
      for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j)
          gpm(ie,d,i,j) = r.urrng(-1,1);
  deep_copy(e.m_geometry.m_gradphis, gpm);

  // Make sure dphi <= -g.
  const auto phis = cmvdc(geo.m_phis);
  const auto phinh_i = cmvdc(e.m_state.m_phinh_i);
  for (int ie = 0; ie < e.num_elems(); ++ie)
    for (int t = 0; t < NUM_TIME_LEVELS; ++t)
      for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j) {
          Real* const phi = &phinh_i(ie,t,i,j,0)[0];
          phi[nlev] = phis(ie,i,j);
          for (int k = nlev-1; k >= 0; --k)
            if (phi[k] - phi[k+1] < PhysicalConstants::g)
              for (int k1 = k; k1 >= 0; --k1)
                phi[k1] += PhysicalConstants::g;
        }
  deep_copy(e.m_state.m_phinh_i, phinh_i);
}

struct Session {
  static const int ne = 4;
  HybridVCoord h;
  Random r;
  std::shared_ptr<Elements> e;
  int nelemd, qsize, nlev, np;
  bool independent_time_steps;
  FunctorsBuffersManager fbm;

  //Session () : r(269041989) {}

  void init () {
    const auto seed = r.gen_seed();
    printf("seed %u\n", seed);

    assert(QSIZE_D >= 4);
    qsize = QSIZE_D;

    auto& c = Context::singleton();

    c.create<HybridVCoord>().random_init(seed);
    h = c.get<HybridVCoord>();

    const auto hyai = cmvdc(h.hybrid_ai);
    const auto hybi = cmvdc(h.hybrid_bi);
    const auto hyam = cmvdc(h.hybrid_am);
    const auto hybm = cmvdc(h.hybrid_bm);
    auto& ref_FE = c.create<ReferenceElement>();
    std::vector<Real> dvv(NP*NP), mp(NP*NP);
    init_compose_f90(ne, hyai.data(), hybi.data(), &hyam(0)[0], &hybm(0)[0], h.ps0,
                     dvv.data(), mp.data());
    ref_FE.init_mass(mp.data());
    ref_FE.init_deriv(dvv.data());

    nelemd = c.get<Connectivity>().get_num_local_elements();
    auto& bmm = c.create<MpiBuffersManagerMap>();
    bmm.set_connectivity(c.get_ptr<Connectivity>());
    c.create<Elements>();
    e = c.get_ptr<Elements>();
    c.create<SimulationParams>();
    c.create<Tracers>(nelemd, qsize);

    init_elems(nelemd, r, h, *e);

    auto& geo = e->m_geometry;
    auto& sphop = c.create<SphereOperators>();
    sphop.setup(geo, ref_FE);
    
    auto& ct = c.create<ComposeTransport>();
    auto& p = c.get<SimulationParams>();
    p.qsize = qsize;
    ct.reset(p);
    fbm.request_size(ct.requested_buffer_size());
    fbm.allocate();
    ct.init_buffers(fbm);
    ct.init_boundary_exchanges();

    nlev = NUM_PHYSICAL_LEV;
    assert(nlev > 0);
    np = NP;
    assert(np == 4);
    independent_time_steps = false; // until impl'ed
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
  static constexpr Real tol = std::numeric_limits<Real>::epsilon();

  auto& s = Session::singleton(); try {

  REQUIRE(compose::test::slmm_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest() == 0);
  REQUIRE(compose::test::cedr_unittest(s.get_mpi_comm()) == 0);

  auto& ct = Context::singleton().get<ComposeTransport>();
  const auto fails = ct.run_unit_tests();
  for (const auto& e : fails) printf("%s %d\n", e.first.c_str(), e.second);
  REQUIRE(fails.empty());

  if (1) {
    const Real twelve_days = 3600 * 24 * 12;
    const Real t0 = 0.13*twelve_days, t1 = 0.22*twelve_days;
    FA5d depf("depf", 3, s.np, s.np, s.nlev, s.nelemd);
    depf(2,0,0,0,0) = 42;
    run_trajectory_f90(t0, t1, s.independent_time_steps, depf.data());
    const auto depc = ct.test_trajectory(t0, t1, s.independent_time_steps);
    REQUIRE(depc.extent_int(0) == s.nelemd);
    REQUIRE(depc.extent_int(2) == s.np*s.np);
    REQUIRE(depc.extent_int(3) == 3);
    for (int ie = 0; ie < s.nelemd; ++ie)
      for (int lev = 0; lev < s.nlev; ++lev)
        for (int j = 0, k = 0; j < s.np; ++j)
          for (int i = 0; i < s.np; ++i, ++k)
            for (int d = 0; d < 3; ++d)
              REQUIRE(equal(depf(d,i,j,lev,ie), depc(ie,lev,k,d), 1e4*tol));
  }

  run_compose_standalone_test_f90();

  } catch (...) { Session::delete_singleton(); }
  Session::delete_singleton();
}
