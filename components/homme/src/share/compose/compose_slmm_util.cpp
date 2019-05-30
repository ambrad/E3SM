#include "compose_slmm.hpp"

namespace slmm {

std::string to_string (const siqk::Mesh<siqk::ko::DefaultExecutionSpace>& m) {
  std::stringstream ss;
  ss.precision(17);
  ss << "(mesh nnode " << nslices(m.p) << " nelem " << nslices(m.e);
  for (Int ie = 0; ie < nslices(m.e); ++ie) {
    ss << "\n  (elem " << ie;
    ss << " (";
    for (Int iv = 0; iv < szslice(m.e); ++iv) ss << " " << m.e(ie,iv);
    ss << ")";
    for (Int iv = 0; iv < szslice(m.e); ++iv) {
      ss << "\n     (p";
      for (Int d = 0; d < 3; ++d)
        ss << " " << m.p(m.e(ie,iv),d);
      ss << ")";
      ss << "\n     (nml";
      for (Int d = 0; d < 3; ++d)
        ss << " " << m.nml(m.en(ie,iv),d);
      ss << ")";
    }
    ss << "))";
  }
  ss << ")";
  return ss.str();
}

} // namespace slmm
