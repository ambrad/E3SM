/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Hash.hpp"

namespace Homme {

void hash (const ExecViewManaged<Scalar******>& v,
           int n0, int n1, int n2, int n3, int n4, int n5,
           HashType& accum_out) {
  HashType accum;
  Kokkos::parallel_reduce(
    MDRangePolicy<ExecSpace, 6>({0, 0, 0, 0, 0, 0}, {n0, n1, n2, n3, n4, n5}),
    KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5, HashType& accum) {
      const auto* vcol = &v(i0,i1,i2,i3,i4,0)[0];
      Homme::hash(vcol[i5], accum);
    }, HashReducer<ExecSpace>(accum));
  hash(accum, accum_out);
}

} // Homme
