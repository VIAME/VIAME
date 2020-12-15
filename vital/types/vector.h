// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for Eigen fixed size column vector typedefs
 */

#ifndef VITAL_VECTOR_H_
#define VITAL_VECTOR_H_

#include <Eigen/Core>

namespace kwiver {
namespace vital {

/// \cond DoxygenSuppress
typedef Eigen::Vector2i vector_2i;
typedef Eigen::Vector2d vector_2d;
typedef Eigen::Vector2f vector_2f;
typedef Eigen::Vector3d vector_3d;
typedef Eigen::Vector3f vector_3f;
typedef Eigen::Vector4d vector_4d;
typedef Eigen::Vector4f vector_4f;
/// \endcond

} } // end namespace vital

#endif // VITAL_VECTOR_H_
