#include "compose/cedr.hpp"
#include "compose/cedr_util.hpp"

namespace homme {
typedef int Int;
typedef double Real;

} // namespace homme

extern "C" {
void kokkos_init () {
  Kokkos::InitArguments args;
  args.disable_warnings = true;
  Kokkos::initialize(args);
}

void kokkos_finalize () { Kokkos::finalize_all(); }

extern "C" void cedr_unittest (const homme::Int fcomm, homme::Int* nerrp) {
}

extern "C" void cedr_set_ie2gci (const homme::Int ie, const homme::Int gci) {
}

extern "C" homme::Int cedr_forcing_init (
  const homme::Int np, const homme::Int nlev, const homme::Int qsize,
  const homme::Int qsized, const homme::Int timelevels,
  const homme::Int need_conservation)
{
  return 1;
}

extern "C" void cedr_forcing_set_qdp (homme::Int ie, homme::Real* v, homme::Int n0_qdp,
                                      homme::Int n1_qdp)
{}

extern "C" void cedr_forcing_run (const homme::Real* minq, const homme::Real* maxq,
                                  homme::Int nets, homme::Int nete) {
  cedr_assert(minq != maxq);
}

extern "C" void cedr_forcing_run_local (const homme::Real* minq, const homme::Real* maxq,
                                        homme::Int nets, homme::Int nete, homme::Int use_ir,
                                        homme::Int limiter_option) {
  cedr_assert(minq != maxq);
}

extern "C" void cedr_forcing_check (const homme::Real* minq, const homme::Real* maxq,
                                    homme::Int nets, homme::Int nete) {
}
} // extern "C"
