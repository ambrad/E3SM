/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_GLLFVREMAP_HPP
#define HOMMEXX_GLLFVREMAP_HPP

#include "Types.hpp"
#include <memory>

namespace Homme {

class FunctorsBuffersManager;
class SimulationParams;
class GllFvRemapImpl;

class GllFvRemap {
public:
  GllFvRemap();
  GllFvRemap(const GllFvRemap &) = delete;
  GllFvRemap &operator=(const GllFvRemap &) = delete;

  ~GllFvRemap();

  void reset(const SimulationParams& params);

  int requested_buffer_size() const;
  void init_buffers(const FunctorsBuffersManager& fbm);
  void init_boundary_exchanges();

  typedef ExecViewUnmanaged<Real*>   Phys0T; // point
  typedef ExecViewUnmanaged<Real**>  Phys1T; // point, lev
  typedef ExecViewUnmanaged<Real***> Phys2T; // point, idx, lev
  typedef Phys1T::const_type CPhys1T;
  typedef Phys2T::const_type CPhys2T;

  void run_dyn_to_fv(const int ncol, const int nq, const int time_idx,
                     // ps,phis(col)
                     const Phys0T& ps, const Phys0T& phis,
                     // T,omega(col,lev)
                     const Phys1T& T, const Phys1T& omega,
                     // uv(col, 0 or 1, lev)
                     const Phys2T& uv, 
                     // q(col,idx,lev)
                     const Phys2T& q);
  void run_fv_to_dyn(const int ncol, const int nq, const int time_idx, const Real dt,
                     const CPhys1T& T, const CPhys2T& uv, const CPhys2T& q);

private:
  std::unique_ptr<GllFvRemapImpl> m_impl;
};

} // Namespace Homme

#endif // HOMMEXX_GLLFVREMAP_HPP
