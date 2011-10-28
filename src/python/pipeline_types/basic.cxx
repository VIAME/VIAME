/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_types/basic_types.h>

#include <boost/python/module.hpp>
#include <boost/python/scope.hpp>

/**
 * \file basic.cxx
 *
 * \brief Python bindings for basic pipeline types.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(basic)
{
  scope s;
  s.attr("t_bool") = vistk::basic_types::t_bool;
  s.attr("t_char") = vistk::basic_types::t_char;
  s.attr("t_string") = vistk::basic_types::t_string;
  s.attr("t_integer") = vistk::basic_types::t_integer;
  s.attr("t_unsigned") = vistk::basic_types::t_unsigned;
  s.attr("t_float") = vistk::basic_types::t_float;
  s.attr("t_double") = vistk::basic_types::t_double;
  s.attr("t_byte") = vistk::basic_types::t_byte;
  s.attr("t_vec_char") = vistk::basic_types::t_vec_char;
  s.attr("t_vec_string") = vistk::basic_types::t_vec_string;
  s.attr("t_vec_integer") = vistk::basic_types::t_vec_integer;
  s.attr("t_vec_unsigned") = vistk::basic_types::t_vec_unsigned;
  s.attr("t_vec_float") = vistk::basic_types::t_vec_float;
  s.attr("t_vec_double") = vistk::basic_types::t_vec_double;
  s.attr("t_vec_byte") = vistk::basic_types::t_vec_byte;
}
