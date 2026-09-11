/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief A `cv::Mat` as an `opencv-matrix` node, for the calibration writers
///
/// P7-T05 replaced `cv::FileStorage` with `library/file_io/opencv_yaml`, and
/// these three calibration algorithms are what wrote calibration files with
/// it. They still hold their results as `cv::Mat` -- they are OpenCV
/// calibration, and P7-T06 is where that changes -- so this is the one
/// conversion the change needs.

#ifndef VIAME_OPENCV_YAML_MATRIX_H
#define VIAME_OPENCV_YAML_MATRIX_H

#include <viame/file_io/opencv_yaml.h>

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

namespace viame {

namespace opencv {

// ----------------------------------------------------------------------------
/// \p mat as an `!!opencv-matrix` node.
///
/// The `dt` is chosen from the matrix's depth the way FileStorage chose it,
/// so a double matrix still writes `dt: d` and reads back identically. A
/// multi-channel matrix has no `dt` of its own here: none of the calibration
/// outputs is one, and guessing at the layout would write a file nothing
/// could read back.
inline file_io::node
mat_to_node( cv::Mat const& mat )
{
  if( mat.channels() != 1 )
  {
    throw file_io::parse_error(
      "cannot write a " + std::to_string( mat.channels() ) +
      " channel matrix as an opencv-matrix" );
  }

  std::string dt;
  switch( mat.depth() )
  {
    case CV_8U:  dt = "u"; break;
    case CV_8S:  dt = "c"; break;
    case CV_16U: dt = "w"; break;
    case CV_16S: dt = "s"; break;
    case CV_32S: dt = "i"; break;
    case CV_32F: dt = "f"; break;
    case CV_64F: dt = "d"; break;
    default:
      throw file_io::parse_error( "matrix has no FileStorage type code" );
  }

  // Through double whatever the depth: the node keeps its `dt` and writes
  // each value back in that type's spelling.
  cv::Mat as_double;
  mat.convertTo( as_double, CV_64F );

  std::vector< double > data;
  data.reserve( static_cast< size_t >( as_double.total() ) );

  for( int row = 0; row < as_double.rows; ++row )
  {
    for( int col = 0; col < as_double.cols; ++col )
    {
      data.push_back( as_double.at< double >( row, col ) );
    }
  }

  return file_io::node::matrix( mat.rows, mat.cols, dt, data );
}

} // namespace opencv

} // namespace viame

#endif // VIAME_OPENCV_YAML_MATRIX_H
