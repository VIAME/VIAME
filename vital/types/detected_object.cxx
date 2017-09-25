/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief Implementation for detected_object class
 */

#include "detected_object.h"

namespace kwiver {
namespace vital {


detected_object::detected_object( const bounding_box_d& bbox,
                                  double              confidence,
                                  detected_object_type_sptr classifications )
  : m_bounding_box( std::make_shared< bounding_box_d >( bbox ) )
  , m_confidence( confidence )
  , m_type( classifications )
  , m_index( 0 )
{
}


// ------------------------------------------------------------------
detected_object_sptr
detected_object
::clone() const
{
  detected_object_type_sptr new_type;
  if (this->m_type )
  {
    new_type = std::make_shared<detected_object_type>( *this->m_type );
  }

  auto new_obj = std::make_shared<kwiver::vital::detected_object>(
    *this->m_bounding_box, this->m_confidence, new_type );

  new_obj->m_mask_image = this->m_mask_image; // being cheap - not copying image mask
  new_obj->m_index = this->m_index;
  new_obj->m_detector_name = this->m_detector_name;
  new_obj->m_descriptor = this->m_descriptor;

  return new_obj;
}


// ------------------------------------------------------------------
bounding_box_d
detected_object
::bounding_box() const
{
  return *m_bounding_box;
}


// ------------------------------------------------------------------
void
detected_object
::set_bounding_box( const bounding_box_d& bbox )
{
  m_bounding_box = std::make_shared< bounding_box_d >( bbox );
}


// ------------------------------------------------------------------
double
detected_object
::confidence() const
{
  return m_confidence;
}


// ------------------------------------------------------------------
void
detected_object
::set_confidence( double d )
{
  m_confidence = d;
}


// ------------------------------------------------------------------
image_container_sptr
detected_object
::mask()
{
  return m_mask_image;
}


// ------------------------------------------------------------------
void
detected_object
::set_mask( image_container_sptr m )
{
  m_mask_image = m;
}


// ------------------------------------------------------------------
detected_object_type_sptr
detected_object
::type()
{
  return m_type;
}


// ------------------------------------------------------------------
void
detected_object
::set_type( detected_object_type_sptr c )
{
  m_type = c;
}

// ------------------------------------------------------------------
uint64_t
detected_object
::index() const
{
  return m_index;
}


// ------------------------------------------------------------------
void
detected_object
::set_index( uint64_t idx )
{
  m_index = idx;
}


// ------------------------------------------------------------------
const std::string&
detected_object
::detector_name() const
{
  return m_detector_name;
}


// ------------------------------------------------------------------
void
detected_object
::set_detector_name( const std::string& name )
{
  m_detector_name = name;
}


// ------------------------------------------------------------------
detected_object::descriptor_sptr
detected_object
::descriptor() const
{
  return m_descriptor;
}


// ------------------------------------------------------------------
void
detected_object
::set_descriptor( descriptor_sptr d )
{
  m_descriptor = d;
}


} } // end namespace
