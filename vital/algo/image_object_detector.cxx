// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief image_object_detector algorithm instantiation
 */

#include <vital/algo/algorithm.txx>
#include <vital/algo/image_object_detector.h>

namespace kwiver {
namespace vital {
namespace algo {

image_object_detector
::image_object_detector()
{
  attach_logger( "algo.image_object_detector" ); // specify a logger
}

std::vector< vital::detected_object_set_sptr >
image_object_detector
::batch_detect( const std::vector< image_container_sptr >& images ) const
{
  std::vector< vital::detected_object_set_sptr > output;

  for( auto image : images )
  {
    output.push_back( this->detect( image ) );
  }

  return output;
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::image_object_detector);
/// \endcond
