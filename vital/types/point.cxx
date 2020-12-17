// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/point.h>

#include <iomanip>
#include <stdexcept>

namespace kwiver {
namespace vital {

template class point< 2, int >;
template class point< 2, double >;
template class point< 2, float >;
template class point< 3, double >;
template class point< 3, float >;
template class point< 4, double >;
template class point< 4, float >;

// ----------------------------------------------------------------------------
template < unsigned N, typename T >
std::ostream&
out(std::ostream& str, point< N, T >const& p)
{
  str << "point" << N << "D\n";

  auto const v = p.value();
  str << " - value : ";
  switch (N)
  {
  case 2:
    str << "[ " << v[0] << ", " << v[1] << " ]\n";
    break;
  case 3:
    str << "[ " << v[0] << ", " << v[1] << ", " << v[2] << " ]\n";
    break;
  case 4:
    str << "[ " << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << " ]\n";
    break;
  default:
    str << "Unexpected dimension for point.\n";
    break;
  }

  str << " - covariance : ";
  auto const c = p.covariance().matrix();
  switch (N)
  {
  case 2:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << " ]\n";
    break;
  case 3:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << ", " << c(0, 2) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << ", " << c(1, 2) << "\n                  "
                << c(2, 0) << ", " << c(2, 1) << ", " << c(2, 2) << " ]\n";
    break;
  case 4:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << ", " << c(0, 2) << ", " << c(0, 3) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << ", " << c(1, 2) << ", " << c(1, 3) << "\n                  "
                << c(2, 0) << ", " << c(2, 1) << ", " << c(2, 2) << ", " << c(2, 3) << "\n                  "
                << c(3, 0) << ", " << c(3, 1) << ", " << c(3, 2) << ", " << c(3, 3) << " ]\n";
    break;
  default:
    str << "Unexpected dimension for point.\n";
    break;
  }

  return str;
}

// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& str, const point_2i& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_2d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_2f& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_3d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_3f& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_4d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_4f& p) { return out(str, p); }

} } // end namespace
