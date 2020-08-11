/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransportImpl.hpp"

namespace Homme {

/*
  m_geometry.m_vec_sph2cart
*/

void ComposeTransportImpl::calc_trajectory () {
  assert( ! m_data.independent_time_steps); // until impl'ed

  
}

ComposeTransport::TestDepView::HostMirror ComposeTransportImpl::
test_trajectory(Real t0, Real t1, bool independent_time_steps) {
  ComposeTransport::TestDepView dep("dep", m_data.nelemd, num_phys_lev, np2, 3);
  const auto deph = Kokkos::create_mirror_view(dep);
  for (int ie = 0; ie < m_data.nelemd; ++ie)
    for (int lev = 0; lev < num_phys_lev; ++lev)
      for (int k = 0; k < np2; ++k)
          for (int d = 0; d < 3; ++d)
            deph(ie,lev,k,d) = ie*(lev + k) + d;
  return deph;
}

} // namespace Homme
