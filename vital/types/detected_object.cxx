// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object class
 */

#include "detected_object.h"

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
detected_object
::detected_object( double confidence,
                   detected_object_type_sptr classifications )
  : m_confidence{ confidence },
    m_type{ classifications }
{
}

// ----------------------------------------------------------------------------
detected_object
::detected_object( bounding_box_d const& bbox,
                   double confidence,
                   detected_object_type_sptr classifications )
  : m_bounding_box{ bbox },
    m_confidence{ confidence },
    m_type{ classifications }
{
}

// ----------------------------------------------------------------------------
detected_object
::detected_object( kwiver::vital::geo_point const& gp,
                   double confidence,
                   detected_object_type_sptr classifications )
  : m_geo_point{ gp },
    m_confidence{ confidence },
    m_type{ classifications }
{
}

// ----------------------------------------------------------------------------
detected_object_sptr
detected_object
::clone() const
{
  detected_object_type_sptr new_type;
  if ( this->m_type )
  {
    new_type = std::make_shared< detected_object_type >( *this->m_type );
  }

  auto new_obj = std::make_shared< kwiver::vital::detected_object >(
    this->m_bounding_box, this->m_confidence, new_type );

  // Be cheap and don't deep copy the image mask or descriptor; we can get away
  // with this because these can't be modified via the detected object, only
  // replaced by a different instance
  new_obj->m_mask_image = this->m_mask_image;
  new_obj->m_descriptor = this->m_descriptor;

  // Copy everything else (value copies)
  new_obj->m_index = this->m_index;
  new_obj->m_detector_name = this->m_detector_name;
  new_obj->m_geo_point = this->m_geo_point;
  new_obj->m_keypoints = this->m_keypoints;
  new_obj->m_notes = this->m_notes;

  return new_obj;
}

// ----------------------------------------------------------------------------
kwiver::vital::geo_point
detected_object
::geo_point() const
{
  return m_geo_point;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_geo_point( kwiver::vital::geo_point const& gp )
{
  m_geo_point = gp;
}

// ----------------------------------------------------------------------------
bounding_box_d
detected_object
::bounding_box() const
{
  return m_bounding_box;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_bounding_box( bounding_box_d const& bbox )
{
  m_bounding_box = bbox;
}

// ----------------------------------------------------------------------------
double
detected_object
::confidence() const
{
  return m_confidence;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_confidence( double d )
{
  m_confidence = d;
}

// ----------------------------------------------------------------------------
image_container_scptr
detected_object
::mask() const
{
  return m_mask_image;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_mask( image_container_scptr m )
{
  m_mask_image = m;
}

// ----------------------------------------------------------------------------
detected_object_type_sptr
detected_object
::type() const
{
  return m_type;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_type( detected_object_type_sptr c )
{
  m_type = c;
}

// ----------------------------------------------------------------------------
uint64_t
detected_object
::index() const
{
  return m_index;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_index( uint64_t idx )
{
  m_index = idx;
}

// ----------------------------------------------------------------------------
std::string
detected_object
::detector_name() const
{
  return m_detector_name;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_detector_name( std::string const& name )
{
  m_detector_name = name;
}

// ----------------------------------------------------------------------------
detected_object::descriptor_scptr
detected_object
::descriptor() const
{
  return m_descriptor;
}

// ----------------------------------------------------------------------------
void
detected_object
::set_descriptor( descriptor_scptr d )
{
  m_descriptor = d;
}

// ----------------------------------------------------------------------------
std::vector< std::string >
detected_object
::notes() const
{
  return m_notes;
}

// ----------------------------------------------------------------------------
void
detected_object
::add_note( std::string const& note )
{
  m_notes.push_back( note );
}

// ----------------------------------------------------------------------------
void
detected_object
::clear_notes()
{
  m_notes.clear();
}

// ----------------------------------------------------------------------------
std::map< std::string, vital::point_2d >
detected_object
::keypoints() const
{
  return m_keypoints;
}

// ----------------------------------------------------------------------------
void
detected_object
::add_keypoint( std::string const& id, vital::point_2d const& p )
{
  m_keypoints[ id ] = p;
}

// ----------------------------------------------------------------------------
void
detected_object
::clear_keypoints()
{
  m_keypoints.clear();
}

} // namespace vital

} // namespace kwiver
