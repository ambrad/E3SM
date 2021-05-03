#ifndef INCLUDE_COMPOSE_TEST_HPP
#define INCLUDE_COMPOSE_TEST_HPP

#include <mpi.h>

extern "C"
void compose_repro_sum(const double* send, double* recv,
                       int nlocal, int nfld, int fcomm);

namespace compose {
namespace test {

int slmm_unittest();
int cedr_unittest();
int cedr_unittest(MPI_Comm comm);

} // namespace test
} // namespace compose

#endif
