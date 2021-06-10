/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "GllFvRemap.hpp"
#include "GllFvRemapImpl.hpp"
#include "Context.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>

namespace Homme {

GllFvRemap::GllFvRemap () {
  m_impl.reset(new GllFvRemapImpl());
}

void GllFvRemap::reset (const SimulationParams& params) {
  m_impl->reset(params);
}

GllFvRemap::~GllFvRemap () = default;

int GllFvRemap::requested_buffer_size () const {
  return m_impl->requested_buffer_size();
}

void GllFvRemap::init_buffers (const FunctorsBuffersManager& fbm) {
  m_impl->init_buffers(fbm);
}

void GllFvRemap::init_boundary_exchanges () {
  m_impl->init_boundary_exchanges();
}

void GllFvRemap
::run_dyn_to_fv (const int time_idx, const Phys0T& ps, const Phys0T& phis,
                 const Phys1T& T, const Phys1T& omega, const Phys2T& uv,
                 const Phys2T& q) {
  m_impl->run_dyn_to_fv(time_idx, ps, phis, T, omega, uv, q);
}

void GllFvRemap
::run_fv_to_dyn (const int time_idx, const Real dt, const CPhys1T& T,
                 const CPhys2T& uv, const CPhys2T& q) {
  m_impl->run_fv_to_dyn(time_idx, dt, T, uv, q);
}

} // Namespace Homme
