#ifndef HASH_HPP
#define HASH_HPP

#include <cstdint>

namespace Homme {

typedef std::uint64_t HashType;

// Accumulate v into accum using a hash of its bits.
KOKKOS_INLINE_FUNCTION void hash (const double v_, HashType& accum) {
  constexpr auto first_bit = 1ULL << 63;
  // reinterpret_cast leads to -Wstrict-aliasing warnings; memcpy instead.
  HashType v;
  std::memcpy(&v, &v_, sizeof(HashType));
  accum += ~first_bit & v; // no overflow
  accum ^=  first_bit & v; // handle most significant bit
}

} // Homme

#endif // HASH_HPP
