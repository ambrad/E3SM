/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Config.hpp"
#ifdef HOMME_ENABLE_COMPOSE

#include "ComposeTransportImpl.hpp"
#include "PhysicalConstants.hpp"

#include "compose_test.hpp"

namespace Homme {
using cti = ComposeTransportImpl;

void ComposeTransportImpl::calc_enhanced_trajectory (const int np1, const Real dt) {
}

int ComposeTransportImpl::run_enhanced_trajectory_unit_tests () {
  return 0;
}

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
