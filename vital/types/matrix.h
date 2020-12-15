// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Typedefs for Eigen matrices
 */

#ifndef VITAL_MATRIX_H_
#define VITAL_MATRIX_H_

#include <Eigen/Core>

namespace kwiver {
namespace vital {

/// \cond DoxygenSuppress
typedef Eigen::MatrixXd matrix_d;
typedef Eigen::MatrixXf matrix_f;

typedef Eigen::Matrix< double, 2, 2 > matrix_2x2d;
typedef Eigen::Matrix< float, 2, 2 >  matrix_2x2f;
typedef Eigen::Matrix< double, 2, 3 > matrix_2x3d;
typedef Eigen::Matrix< float, 2, 3 >  matrix_2x3f;
typedef Eigen::Matrix< double, 3, 2 > matrix_3x2d;
typedef Eigen::Matrix< float, 3, 2 >  matrix_3x2f;
typedef Eigen::Matrix< double, 3, 3 > matrix_3x3d;
typedef Eigen::Matrix< float, 3, 3 >  matrix_3x3f;
typedef Eigen::Matrix< double, 3, 4 > matrix_3x4d;
typedef Eigen::Matrix< float, 3, 4 >  matrix_3x4f;
typedef Eigen::Matrix< double, 4, 3 > matrix_4x3d;
typedef Eigen::Matrix< float, 4, 3 >  matrix_4x3f;
typedef Eigen::Matrix< double, 4, 4 > matrix_4x4d;
typedef Eigen::Matrix< float, 4, 4 >  matrix_4x4f;
/// \endcond

} } // end namespace vital

#endif // VITAL_MATRIX_H_
