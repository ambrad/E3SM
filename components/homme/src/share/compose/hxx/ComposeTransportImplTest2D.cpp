/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "compose_test.hpp"
#include "profiling.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

static void cp_v_to_vstar(const ElementsState& s, const ElementsDerivedState& d) {
  
}

static void fill_v (const ElementsState& s, const int np1) {
  
}

void ComposeTransportImpl::test_2d (const int nstep) {
  SimulationParams& params = Context::singleton().get<SimulationParams>();
  assert(params.params_set);

  TimeLevel& tl = Context::singleton().get<TimeLevel>();
  tl.nstep = 0;
  tl.update_tracers_levels(params.qsplit);
  const Real twelve_days = 3600 * 24 * 12, dt = twelve_days/nstep;

  GPTLstart("compose_stt_step");
  for (int i = 0; i < nstep; ++i) {
    const auto tprev = dt*i;
    const auto t = dt*i;
    cp_v_to_vstar(m_state, m_derived);
    fill_v(m_state, tl.np1);
    tl.nstep += params.qsplit;
    run(tl, dt);
  }
  GPTLstop("compose_stt_step");
}

} // namespace Homme
