/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransport.hpp"
#include "ComposeTransportImpl.hpp"
#include "Context.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>

namespace Homme {

ComposeTransport::ComposeTransport () {
  m_compose_impl.reset(new ComposeTransportImpl());
}

void ComposeTransport::reset (const SimulationParams& params) {
  m_compose_impl->reset(params);
}

ComposeTransport::~ComposeTransport () = default;

int ComposeTransport::requested_buffer_size () const {
  return m_compose_impl->requested_buffer_size();
}

void ComposeTransport::init_buffers (const FunctorsBuffersManager& fbm) {
  m_compose_impl->init_buffers(fbm);
}

void ComposeTransport::init_boundary_exchanges () {
  m_compose_impl->init_boundary_exchanges();
}

void ComposeTransport::run (const TimeLevel& tl, const Real dt) {
  GPTLstart("compute_step");
  m_compose_impl->run(tl, dt);
  GPTLstop("compute_step");
}

std::vector<std::pair<std::string, int> >
ComposeTransport::run_unit_tests () {
  std::vector<std::pair<std::string, int> > fails;
  return fails;
}

ComposeTransport::TestDepView::HostMirror ComposeTransport::
test_trajectory (Real t0, Real t1, const bool independent_time_steps) {
  return m_compose_impl->test_trajectory(t0, t1, independent_time_steps);
}

void ComposeTransport::test_2d (const bool bfb, const int nstep, std::vector<Real>& eval) {
  m_compose_impl->test_2d(bfb, nstep, eval);
}

} // Namespace Homme
