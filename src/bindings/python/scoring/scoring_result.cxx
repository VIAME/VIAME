/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
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

static vistk::scoring_result_t new_result(vistk::scoring_result::count_t true_positive,
                                          vistk::scoring_result::count_t false_positive,
                                          vistk::scoring_result::count_t total_true,
                                          vistk::scoring_result::count_t possible);
static vistk::scoring_result::count_t result_get_true_positives(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_get_false_positives(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_get_total_trues(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_get_total_possible(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_get_percent_detection(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_get_precision(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_get_specificity(vistk::scoring_result_t const& self);
static vistk::scoring_result_t result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs);

BOOST_PYTHON_MODULE(scoring_result)
{
  class_<vistk::scoring_result_t>("ScoringResult"
    , "A result from a scoring algorithm."
    , no_init)
    .def("__init__", &new_result
      , (arg("true_positive"), arg("false_positive"), arg("total_true"), arg("possible") = 0)
      , "Constructor.")
    .def("true_positives", &result_get_true_positives)
    .def("false_positives", &result_get_false_positives)
    .def("total_trues", &result_get_total_trues)
    .def("total_possible", &result_get_total_possible)
    .def("percent_detection", &result_get_percent_detection)
    .def("precision", &result_get_precision)
    .def("specificity", &result_get_specificity)
    .def("__add__", &result_add
      , (arg("lhs"), arg("rhs")))
  ;

  vistk::python::register_type<vistk::scoring_result_t>(20);
}

vistk::scoring_result_t
new_result(vistk::scoring_result::count_t true_positive,
           vistk::scoring_result::count_t false_positive,
           vistk::scoring_result::count_t total_true,
           vistk::scoring_result::count_t possible)
{
  return boost::make_shared<vistk::scoring_result>(true_positive, false_positive, total_true, possible);
}

vistk::scoring_result::count_t
result_get_true_positives(vistk::scoring_result_t const& self)
{
  return self->true_positives;
}

vistk::scoring_result::count_t
result_get_false_positives(vistk::scoring_result_t const& self)
{
  return self->false_positives;
}

vistk::scoring_result::count_t
result_get_total_trues(vistk::scoring_result_t const& self)
{
  return self->total_trues;
}

vistk::scoring_result::count_t
result_get_total_possible(vistk::scoring_result_t const& self)
{
  return self->total_possible;
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

vistk::scoring_result::result_t
result_get_specificity(vistk::scoring_result_t const& self)
{
  return self->specificity();
}

vistk::scoring_result_t
result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs)
{
  return (lhs + rhs);
}
