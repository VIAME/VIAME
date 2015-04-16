/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <kwiver/core/homography.h>

#include <sprokit/python/any_conversion/prototypes.h>
#include <sprokit/python/any_conversion/registration.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/operators.hpp>

/**
 * \file homography.cxx
 *
 * \brief Python bindings for \link kwiver::homography\endlink.
 */

using namespace boost::python;

static double transform_get(kwiver::f2f_homography const& self_, unsigned row, unsigned col);
static void transform_set(kwiver::f2f_homography& self_, unsigned row, unsigned col, double val);
static unsigned from_id(kwiver::f2f_homography& self_ );
static unsigned to_id(kwiver::f2f_homography& self_ );

// name must match name of .so
// Should add kwiver in the name to differentiate from VidTK homography
BOOST_PYTHON_MODULE(libkwiver_python_convert_homography)
{
  /// \todo Use NumPy instead.
  class_<kwiver::f2f_homography>("HomographyTransform"
      , "A transformation matrix for a Kwiver homography."
   , no_init)
    .def("get", &transform_get
      , (arg("row"), arg("column"))
      , "Get a value from the transformation.")
    .def("set", &transform_set
      , (arg("row"), arg("column"), arg("value"))
      , "Set a value in the transformation.")
    .def("from_id", &from_id, "Get source id for transformation." )
    .def("to_id", &to_id, "Get destination id for transformation." )
    //+ need setters too
  ;

/*
  class_<kwiver::f2f_homography_base>("HomographyBase"
    , "The base class for homographies."
    , no_init)
    .def("transform", &kwiver::f2f_homography_base::transform
      , "The transform matrix for the homography."
      , return_internal_reference<1>())

    .def("is_valid", &kwiver::f2f_homography_base::is_valid
      , "True if the homography is valid, False otherwise.")

    .def("is_new_reference", &kwiver::f2f_homography_base::is_new_reference
      , "True if the homography is a new reference frame, False otherwise.")
    .def("set_transform", &kwiver::f2f_homography_base::set_transform
      , (arg("transform"))
      , "Sets the transformation matrix.")
    .def("set_identity", &kwiver::f2f_homography_base::set_identity
      , "Sets the homography transformation matrix to identity.")
    .def("set_valid", &kwiver::f2f_homography_base::set_valid
      , (arg("valid"))
      , "Sets whether the homography is valid or not.")
    .def("set_new_reference", &kwiver::f2f_homography_base::set_new_reference
      , "Sets whether the homography is a new reference or not.")
    .def(self == self)
  ;

  /// \todo How to do multiplication?
#define HOMOGRAPHY_CLASS(type, name, desc)            \
  do                                                  \
  {                                                   \
    class_<type, bases<kwiver::f2f_homography> >(name \
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
    kwiver::python::register_type<type>(20);           \
  } while (false)

  HOMOGRAPHY_CLASS(kwiver::f2f_homography,
                   "ImageToImageHomography",
                   "A homography from one image plane to another.");
  /// \todo Wrap up other plane ref types.

#undef HOMOGRAPHY_CLASS
  */

sprokit::python::register_type<kwiver::f2f_homography>(20);
}

double
transform_get(kwiver::f2f_homography const& self_, unsigned row, unsigned col)
{
  return self_(row, col);
}

void
transform_set(kwiver::f2f_homography& self_, unsigned row, unsigned col, double val)
{
  self_(row, col) = val;
}

unsigned
from_id(kwiver::f2f_homography& self_ )
{
  return self_.from_id();
}


unsigned
to_id(kwiver::f2f_homography& self_ )
{
  return self_.to_id();
}
