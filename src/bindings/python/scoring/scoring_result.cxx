/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/scoring/scoring_result.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/make_shared.hpp>

/**
 * \file scoring_result.cxx
 *
 * \brief Python bindings for scoring_result.
 */

using namespace boost::python;

static vistk::scoring_result_t new_result(vistk::scoring_result::count_t hit, vistk::scoring_result::count_t miss, vistk::scoring_result::count_t truth);
static vistk::scoring_result::count_t result_get_hit(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_get_miss(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_get_truth(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_get_percent_detection(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_get_precision(vistk::scoring_result_t const& self);
static vistk::scoring_result_t result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs);

BOOST_PYTHON_MODULE(scoring_result)
{
  class_<vistk::scoring_result_t>("ScoringResult"
    , "A result from a scoring algorithm."
    , no_init)
    .def("__init__", &new_result
      , (arg("hit"), arg("miss"), arg("truth"))
      , "Constructor.")
    .def("hit_count", &result_get_hit)
    .def("miss_count", &result_get_miss)
    .def("truth_count", &result_get_truth)
    .def("percent_detection", &result_get_percent_detection)
    .def("precision", &result_get_precision)
    .def("__add__", &result_add
      , (arg("lhs"), arg("rhs")))
  ;

  vistk::python::register_type<vistk::scoring_result_t>(20);
}

vistk::scoring_result_t
new_result(vistk::scoring_result::count_t hit, vistk::scoring_result::count_t miss, vistk::scoring_result::count_t truth)
{
  return boost::make_shared<vistk::scoring_result>(hit, miss, truth);
}

vistk::scoring_result::count_t
result_get_hit(vistk::scoring_result_t const& self)
{
  return self->hit_count;
}

vistk::scoring_result::count_t
result_get_miss(vistk::scoring_result_t const& self)
{
  return self->miss_count;
}

vistk::scoring_result::count_t
result_get_truth(vistk::scoring_result_t const& self)
{
  return self->truth_count;
}

vistk::scoring_result::result_t
result_get_percent_detection(vistk::scoring_result_t const& self)
{
  return self->percent_detection();
}

vistk::scoring_result::result_t
result_get_precision(vistk::scoring_result_t const& self)
{
  return self->precision();
}

vistk::scoring_result_t
result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs)
{
  return (lhs + rhs);
}
