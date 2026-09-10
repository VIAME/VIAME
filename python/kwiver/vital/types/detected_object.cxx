// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/core_types/attribute_set.h>
#include <viame/core_types/detected_object.h>

#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

typedef kwiver::vital::detected_object det_obj;

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

// We want to be able to add a mask in the python constructor
// so we need a pass-through cstor
std::shared_ptr< det_obj >
new_detected_object(
  bounding_box< double > bbox,
  double conf,
  kwiver::vital::detected_object_type_sptr type,
  kwiver::vital::image_container_sptr mask )
{
  std::shared_ptr< det_obj > new_obj( new det_obj( bbox, conf, type ) );

  if( mask )
  {
    new_obj->set_mask( mask );
  }

  return new_obj;
}

// Pybind casts away all const-ness, and since a few getters/setters
// in the detected_object class have pointers to const, we have to copy
// them in order to avoid undefined behavior.
descriptor_sptr
det_obj_const_safe_descriptor( detected_object const& self )
{
  auto desc = self.descriptor();
  if( desc )
  {
    // Create a pointer to a copy so we don't violate const
    return desc->clone();
  }
  return nullptr;
}

void
det_obj_const_safe_set_descriptor( detected_object& self, descriptor_sptr desc )
{
  if( desc )
  {
    // Return a pointer to a copy
    // clone() returns pointer to base
    auto cloned_desc = desc->clone();
    auto des_dyn_sptr =
      std::dynamic_pointer_cast< descriptor_dynamic< double > >(
        cloned_desc );

    // Check conversion worked
    if( !des_dyn_sptr )
    {
      throw std::runtime_error(
        "Downcasting descriptor_dynamic<double> from base pointer failed" );
    }
    self.set_descriptor( des_dyn_sptr );
  }
  else
  {
    self.set_descriptor( nullptr );
  }
}

// TODO: uncomment these when rebased on latest master with metadata API changes
// Those changes will make copying metadata objects much easier.
// metadata_sptr
metadata_sptr
copy_metadata( metadata_sptr m )
{
  auto m_clone = std::make_shared< metadata >();
  auto eix = m->end();
  auto ix = m->begin();
  for(; ix != eix; ix++ )
  {
    m_clone->add_copy( ix->second );
  }
  return m_clone;
}

image_container_sptr
det_obj_const_safe_mask( detected_object const& self )
{
  auto mask = self.mask();
  if( mask )
  {
    // image_container does not have a clone method
    // manual copy must be made
    auto im = image( mask->get_image() );
    auto meta = mask->get_metadata();
    if( meta )
    {
      auto md = copy_metadata( meta );
      return std::make_shared< simple_image_container >( im, md );
    }
    return std::make_shared< simple_image_container >( im );
  }
  return nullptr;
}

void
det_obj_const_safe_set_mask( detected_object& self, image_container_sptr mask )
{
  if( mask )
  {
    auto im = image( mask->get_image() );
    auto meta = mask->get_metadata();
    if( meta )
    {
      auto md = copy_metadata( meta );
      auto ptr = std::make_shared< simple_image_container >( im, md );
      self.set_mask( ptr );
    }
    else
    {
      auto ptr = std::make_shared< simple_image_container >( im );
      self.set_mask( ptr );
    }
  }
  else
  {
    self.set_mask( nullptr );
  }
}

// Helper function to set an attribute from a Python object
void
det_obj_set_attribute(
  detected_object& self, std::string const& key,
  py::object value )
{
  // Convert Python object to appropriate C++ type and store
  if( py::isinstance< py::bool_ >( value ) )
  {
    self.set_attribute( key, value.cast< bool >() );
  }
  else if( py::isinstance< py::int_ >( value ) )
  {
    self.set_attribute( key, value.cast< int64_t >() );
  }
  else if( py::isinstance< py::float_ >( value ) )
  {
    self.set_attribute( key, value.cast< double >() );
  }
  else if( py::isinstance< py::str >( value ) )
  {
    self.set_attribute( key, value.cast< std::string >() );
  }
  else
  {
    throw std::runtime_error(
      "Unsupported attribute type. Supported types: bool, int, float, str" );
  }
}

// Helper function to get an attribute as a Python object
py::object
det_obj_get_attribute( detected_object const& self, std::string const& key )
{
  auto attrs = self.attributes();
  if( !attrs )
  {
    throw attribute_set_exception(
      "No attribute set exists for this detection" );
  }
  if( !attrs->has( key ) )
  {
    throw attribute_set_exception( "Attribute not found: " + key );
  }

  auto data = attrs->data( key );

  // Try to convert to known types
  if( data.type() == typeid( bool ) )
  {
    return py::cast( kwiver::vital::any_cast< bool >( data ) );
  }
  else if( data.type() == typeid( int ) )
  {
    return py::cast( kwiver::vital::any_cast< int >( data ) );
  }
  else if( data.type() == typeid( int64_t ) )
  {
    return py::cast( kwiver::vital::any_cast< int64_t >( data ) );
  }
  else if( data.type() == typeid( double ) )
  {
    return py::cast( kwiver::vital::any_cast< double >( data ) );
  }
  else if( data.type() == typeid( std::string ) )
  {
    return py::cast( kwiver::vital::any_cast< std::string >( data ) );
  }
  else
  {
    throw std::runtime_error(
      "Attribute has unsupported type for Python conversion" );
  }
}

// Helper to check if attribute exists
bool
det_obj_has_attribute( detected_object const& self, std::string const& key )
{
  return self.has_attribute( key );
}

// Helper to get all attribute keys
std::vector< std::string >
det_obj_attribute_keys( detected_object const& self )
{
  std::vector< std::string > keys;
  auto attrs = self.attributes();
  if( attrs )
  {
    for( auto it = attrs->begin(); it != attrs->end(); ++it )
    {
      keys.push_back( it->first );
    }
  }
  return keys;
}

} // namespace python

} // namespace vital

} // namespace kwiver

using namespace kwiver::vital;

PYBIND11_MODULE( detected_object, m )
{
  py::module::import( "kwiver.vital.types.geo_point" );
  py::module::import( "kwiver.vital.types.point" );

  /*
   *
   *  Developer:
   *     python -c "import vital.types; help(vital.types.DetectedObject)"
   *     python -m xdoctest vital.types DetectedObject --xdoc-dynamic
   *
   */

  py::class_< det_obj, std::shared_ptr< det_obj > >(
    m, "DetectedObject",
    R"(
    Represents a detected object within an image

    Example:
        >>> from kwiver.vital.types import *
        >>> from PIL import Image as PILImage
        >>> from kwiver.vital.util import VitalPIL
        >>> import numpy as np
        >>> bbox = BoundingBox(0, 10, 100, 50)
        >>> # Construct an object without a mask
        >>> dobj1 = DetectedObject(bbox, 0.2)
        >>> assert dobj1.mask is None
        >>> # Construct an object with a mask
        >>> pil_img = PILImage.fromarray(np.zeros((10, 10), dtype=np.uint8))
        >>> vital_img = VitalPIL.from_pil(pil_img)
        >>> mask = ImageContainer(vital_img)
        >>> self = DetectedObject(bbox, 1.0, mask=mask)
        >>> assert self.mask is mask
        >>> print(self)
        <DetectedObject(conf=1.0)>
    )" )
    .def(
    py::init( &python::new_detected_object ),
    py::arg( "bbox" ), py::arg( "confidence" ) = 1.0,
    py::arg( "classifications" ) = kwiver::vital::detected_object_type_sptr(),
    py::arg( "mask" ) = kwiver::vital::image_container_sptr(),
    py::doc(
      R"(
      Args:
          bbox: coarse localization of the object in image coordinates
          confidence: confidence in this detection (default=1.0)
          classifications: optional object classification (default=None)
    ")" )
    )
    .def(
      "__nice__", [](det_obj& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec( R"(
        retval = 'conf={}'.format(self.confidence)
    )", py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def(
      "__repr__", [](py::object& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec(
          R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
    )",
          py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def(
      "__str__", [](py::object& self) -> std::string {
        auto locals = py::dict( py::arg( "self" ) = self );
        py::exec(
          R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s)>' % (classname, devnice)
    )",
          py::globals(), locals );
        return locals[ "retval" ].cast< std::string >();
      } )
    .def( "clone", &det_obj::clone )
    .def( "add_note", &det_obj::add_note )
    .def( "clear_notes", &det_obj::clear_notes )
    .def( "add_keypoint", &det_obj::add_keypoint )
    .def( "clear_keypoints", &det_obj::clear_keypoints )
    // TODO: Uncomment after above const-safe methods are implemented for mask
    .def_property(
      "mask", &python::det_obj_const_safe_mask,
      &python::det_obj_const_safe_set_mask )

    // Convey that users can't access the the underlying descriptor directly.
    // Must go through the setter. This is because of the const-issue discussed
    // above.
    .def( "descriptor_copy", &python::det_obj_const_safe_descriptor )
    .def( "set_descriptor", &python::det_obj_const_safe_set_descriptor )
    .def_property(
      "bounding_box", &det_obj::bounding_box,
      &det_obj::set_bounding_box )
    .def_property( "geo_point", &det_obj::geo_point, &det_obj::set_geo_point )
    .def_property(
      "confidence", &det_obj::confidence,
      &det_obj::set_confidence )
    .def_property( "index", &det_obj::index, &det_obj::set_index )
    .def_property(
      "detector_name", &det_obj::detector_name,
      &det_obj::set_detector_name )
    .def_property( "type", &det_obj::type, &det_obj::set_type )
    .def_property_readonly( "notes", &det_obj::notes )
    .def_property_readonly( "keypoints", &det_obj::keypoints )
    .def( "set_flattened_polygon", &det_obj::set_flattened_polygon )
    .def( "get_flattened_polygon", &det_obj::get_flattened_polygon )
    .def(
      "set_attribute", &python::det_obj_set_attribute,
      py::arg( "key" ), py::arg( "value" ),
      R"(
      Set an attribute value for this detection.

      Creates the attribute set automatically if it doesn't exist.
      This method is thread-safe.

      Args:
          key: The attribute name/key (string)
          value: The attribute value (bool, int, float, or str)

      Raises:
          RuntimeError: If value type is not supported

      Example:
          >>> det.set_attribute("species", "fish")
          >>> det.set_attribute("length_cm", 42.5)
          >>> det.set_attribute("is_verified", True)
      )" )
    .def(
      "get_attribute", &python::det_obj_get_attribute,
      py::arg( "key" ),
      R"(
      Get an attribute value from this detection.

      Args:
          key: The attribute name/key (string)

      Returns:
          The attribute value (type depends on what was stored)

      Raises:
          RuntimeError: If no attributes exist or key not found

      Example:
          >>> species = det.get_attribute("species")
      )" )
    .def(
      "has_attribute", &python::det_obj_has_attribute,
      py::arg( "key" ),
      R"(
      Check if an attribute exists.

      Args:
          key: The attribute name/key (string)

      Returns:
          True if the attribute exists, False otherwise

      Example:
          >>> if det.has_attribute("species"):
          >>>     print(det.get_attribute("species"))
      )" )
    .def(
      "attribute_keys", &python::det_obj_attribute_keys,
      R"(
      Get list of all attribute keys.

      Returns:
          List of attribute key strings

      Example:
          >>> for key in det.attribute_keys():
          >>>     print(f"{key}: {det.get_attribute(key)}")
      )" )
  ;
}
