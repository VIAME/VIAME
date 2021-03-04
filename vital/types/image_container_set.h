// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core image_container_set interface
 */

#ifndef VITAL_IMAGE_CONTAINER_SET_H_
#define VITAL_IMAGE_CONTAINER_SET_H_

#include "image_container.h"

#include <vital/vital_export.h>
#include <vital/set.h>
#include <vital/logger/logger.h>
#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {

/// An abstract ordered collection of feature images.
/**
 * The base class image_container_set is abstract and provides an interface
 * for returning a vector of images.  There is a simple derived class
 * that stores the data as a vector of images and returns it.  Other
 * derived classes can store the data in other formats and convert on demand.
 */
class image_container_set
  : public set< image_container_sptr >
  , private noncopyable
{
public:
  /// Destructor
  virtual ~image_container_set() = default;

protected:
  image_container_set()
   : m_logger( kwiver::vital::get_logger( "vital.image_container_set" ) )
  {}

  kwiver::vital::logger_handle_t m_logger;
};

/// Shared pointer for base image_container_set type
typedef std::shared_ptr< image_container_set > image_container_set_sptr;

} } // end namespace vital

#endif // VITAL_IMAGE_CONTAINER_SET_H_
