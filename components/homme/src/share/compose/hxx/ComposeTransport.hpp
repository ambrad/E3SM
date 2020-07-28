/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_COMPOSE_TRANSPORT_HPP
#define HOMMEXX_COMPOSE_TRANSPORT_HPP

#include "Types.hpp"
#include <memory>

namespace Homme {

class FunctorsBuffersManager;
class Elements;
class HybridVCoord;
class ComposeTransportImpl;

class ComposeTransport {
public:
  ComposeTransport(const int nelem);
  ComposeTransport(const ComposeTransport &) = delete;
  ComposeTransport &operator=(const ComposeTransport &) = delete;

  ~ComposeTransport();

  int requested_buffer_size() const;
  void init_buffers(const FunctorsBuffersManager& fbm);

  void run();

  static std::vector<std::pair<std::string, int> > run_unit_tests();

private:
  std::unique_ptr<ComposeTransportImpl> m_compose_impl;
};

} // Namespace Homme

#endif // HOMMEXX_COMPOSE_TRANSPORT_HPP
