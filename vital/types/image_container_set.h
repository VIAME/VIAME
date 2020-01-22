/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
