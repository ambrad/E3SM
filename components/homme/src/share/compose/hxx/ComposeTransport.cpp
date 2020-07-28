/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ComposeTransport.hpp"
#include "ComposeTransportImpl.hpp"
#include "Context.hpp"

#include "cedr_local.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>

namespace Homme {

ComposeTransport::ComposeTransport (int nelem) {
  m_compose_impl.reset(new ComposeTransportImpl(nelem));
}

ComposeTransport::~ComposeTransport () = default;

int ComposeTransport::requested_buffer_size () const {
  return m_compose_impl->requested_buffer_size();
}

void ComposeTransport::init_buffers (const FunctorsBuffersManager& fbm) {
  m_compose_impl->init_buffers(fbm);
}

void ComposeTransport::run () {
  GPTLstart("compute_step");
  GPTLstop("compute_step");
}

std::vector<std::pair<std::string, int> >
ComposeTransport::run_unit_tests () {
  std::vector<std::pair<std::string, int> > fails;
  int ne;
  ne = cedr::local::unittest();
  if (ne) fails.push_back(std::make_pair(std::string("cedr::local"), ne));
  return fails;
}

} // Namespace Homme
