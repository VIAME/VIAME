// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Mathematical constants.
 *
 * This file contains definitions for standard mathematical constants.
 */

#ifndef KWIVER_VITAL_MATH_CONSTANTS_H
#define KWIVER_VITAL_MATH_CONSTANTS_H

namespace kwiver {
namespace vital {

// Source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
constexpr double pi = 3.14159265358979323;
constexpr double two_pi = pi*2;
constexpr double pi_over_2 = pi/2;

constexpr double deg_to_rad = pi/180.0;

constexpr double rad_to_deg = 180.0/pi;

} } // end namespace

#endif // KWIVER_VITAL_MATH_CONSTANTS_H
