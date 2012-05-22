/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/utilities/homography.h>
#include <vistk/utilities/homography_types.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/operators.hpp>

/**
 * \file homography.cxx
 *
 * \brief Python bindings for homography.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(homography)
{
  /// \todo Bind transform_t.
  class_<vistk::homography_base::transform_t>("HomographyTransform"
    , "A transformation matrix for a homography."
    , no_init)
  ;

  class_<vistk::homography_base>("HomographyBase"
    , "The base class for homographies."
    , no_init)
    .def("transform", &vistk::homography_base::transform
      , "The transform matrix for the homography."
      , return_internal_reference<1>())
    .def("is_valid", &vistk::homography_base::is_valid
      , "True if the homography is valid, False otherwise.")
    .def("is_new_reference", &vistk::homography_base::is_new_reference
      , "True if the homography is a new reference frame, False otherwise.")
    .def("set_transform", &vistk::homography_base::set_transform
      , (arg("transform"))
      , "Sets the transformation matrix.")
    .def("set_identity", &vistk::homography_base::set_identity
      , "Sets the homography transformation matrix to identity.")
    .def("set_valid", &vistk::homography_base::set_valid
      , (arg("valid"))
      , "Sets whether the homography is valid or not.")
    .def("set_new_reference", &vistk::homography_base::set_new_reference
      , "Sets whether the homography is a new reference or not.")
    .def(self == self)
  ;

  /// \todo How to do multiplication?
#define HOMOGRAPHY_CLASS(type, name, desc)            \
  do                                                  \
  {                                                   \
    class_<type, bases<vistk::homography_base> >(name \
      , desc)                                         \
      .def("source", &type::source                    \
        , "The source plane data.")                   \
      .def("destination", &type::destination          \
        , "The destination plane data.")              \
      .def("set_source", &type::set_source            \
        , "Sets the source plane data.")              \
      .def("set_destination", &type::set_destination  \
        , "Sets the destination plane data.")         \
      .def("inverse", &type::inverse                  \
        , "The inverse homography.")                  \
      .def(self == self)                              \
    ;                                                 \
                                                      \
    vistk::python::register_type<type>(20);           \
  } while (false)

  HOMOGRAPHY_CLASS(vistk::image_to_image_homography,
                   "ImageToImageHomography",
                   "A homography from one image plane to another.");
  /// \todo Wrap up other plane ref types.

#undef HOMOGRAPHY_CLASS
}
