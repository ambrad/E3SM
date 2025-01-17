/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#ifndef HOMMEXX_INDEX_UTILS_HPP
#define HOMMEXX_INDEX_UTILS_HPP

#include "Dimensions.hpp"

namespace Homme {

KOKKOS_INLINE_FUNCTION
void get_ie_igp_jgp_midlevpack (const int idx, int& ie, int& igp, int& jgp, int& ilev) {
  ie   =  idx / (NUM_LEV*NP*NP);
  igp  = (idx / (NUM_LEV*NP)) % NP;
  jgp  = (idx / NUM_LEV) % NP;
  ilev =  idx % NUM_LEV;  
}

} // namespace Homme

#endif // HOMMEXX_INDEX_UTILS_HPP
