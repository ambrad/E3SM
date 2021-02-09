/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"
#include "Context.hpp"
#include "VerticalRemapManager.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

void ComposeTransportImpl
::remap_v (const ExecViewUnmanaged<const Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>& dp3d,
           const int np1, const ExecViewManaged<const Scalar*[NP][NP][NUM_LEV]>& m_divdp,
           const ExecViewManaged<Scalar*[2][NP][NP][NUM_LEV]>& m_vn0) {
  const auto vrm = Context::singleton().get<VerticalRemapManager>();
}

} // namespace Homme
