/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
\file This file contains functions to support converting sprokit datum types to
more python friendly types.
 */

#include "vital_type_converters.h"

#include <vital/types/image_container.h>
#include <vital/types/track_set.h>

#include <vital/logger/logger.h>

#include <boost/any.hpp>

static kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital_type_converters" ) );


// ------------------------------------------------------------------
/**
 * @brief Convert datum to image container handle
 *
 * @param args PyCapsule object
 *
 * @return image container handle
 */
vital_image_container_t* vital_image_container_from_datum( PyObject* args )
{
  // Get capsule from args - or arg may be the capsule
  sprokit::datum* dptr = (sprokit::datum*) PyCapsule_GetPointer( args, "sprokit::datum" );

  try
  {
    // Get boost::any from the datum
    boost::any const any = dptr->get_datum< boost::any >();

    // Get sptr from boost::any
    kwiver::vital::image_container_sptr sptr = boost::any_cast< kwiver::vital::image_container_sptr >(any);

    // Register this object with the main image_container interface
    vital_image_container_t* ptr =  vital_image_container_from_sptr( reinterpret_cast< void* >(&sptr) );
    return ptr;
  }
  catch (boost::bad_any_cast const& e)
  {
    // This is a warning because this converter should only be called
    // if there is good reason to believe that the object really is an
    // image_container.
    LOG_WARN( logger, "Conversion error" << e.what() );
  }

  return NULL;
}


// ------------------------------------------------------------------
/**
 * @brief Convert from datum to track_set handle.
 *
 * @param args PyCapsule object
 *
 * @return track_set handle
 */
vital_trackset_t* vital_trackset_from_datum( PyObject* args )
{
  // Get capsule from args - or arg may be the capsule
  sprokit::datum* dptr = (sprokit::datum*) PyCapsule_GetPointer( args, "sprokit::datum" );

  try
  {
    boost::any const any = dptr->get_datum< boost::any >();
    kwiver::vital::track_set_sptr sptr = boost::any_cast< kwiver::vital::track_set_sptr >(any);

    // Register this object with the main track_set interface
    vital_trackset_t* ptr =  vital_trackset_from_sptr( reinterpret_cast< void* >(&sptr) );
    return ptr;
  }
  catch (boost::bad_any_cast const& e)
  {
    // This is a warning because this converter should only be called
    // if there is good reason to believe that the object really is an
    // track_set.
    LOG_WARN( logger, "Conversion error" << e.what() );
  }

  return 0;
}


// more to come
//
