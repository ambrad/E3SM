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
using CSNlev  = cti::CSNlev;
using CRNlev  = cti::CRNlev;
using CSNlevp = cti::CSNlevp;
using CRNlevp = cti::CRNlevp;
using CS2Nlev = cti::CS2Nlev;
using SNlev   = cti::SNlev;
using RNlev   = cti::RNlev;
using SNlevp  = cti::SNlevp;
using RNlevp  = cti::RNlevp;
using S2Nlev  = cti::S2Nlev;
using R2Nlev  = cti::R2Nlev;
using S2Nlevp = cti::S2Nlevp;

KOKKOS_FUNCTION static void
linterp () {
  
}

void ComposeTransportImpl::calc_enhanced_trajectory (const int np1, const Real dt) {
}

int ComposeTransportImpl::run_enhanced_trajectory_unit_tests () {
  printf("hi\n");
  return 0;
}

} // namespace Homme

#endif // HOMME_ENABLE_COMPOSE
