/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/scoring/scoring_result.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/make_shared.hpp>

/**
 * \file scoring_result.cxx
 *
 * \brief Python bindings for \link vistk::scoring_result\endlink.
 */

using namespace boost::python;

static vistk::scoring_result_t new_result(vistk::scoring_result::count_t true_positive,
                                          vistk::scoring_result::count_t false_positive,
                                          vistk::scoring_result::count_t total_true,
                                          vistk::scoring_result::count_t possible);
static vistk::scoring_result_t new_result_def(vistk::scoring_result::count_t true_positive,
                                              vistk::scoring_result::count_t false_positive,
                                              vistk::scoring_result::count_t total_true);
static vistk::scoring_result::count_t result_true_positives(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_false_positives(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_total_trues(vistk::scoring_result_t const& self);
static vistk::scoring_result::count_t result_total_possible(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_percent_detection(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_precision(vistk::scoring_result_t const& self);
static vistk::scoring_result::result_t result_specificity(vistk::scoring_result_t const& self);
static vistk::scoring_result_t result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs);

BOOST_PYTHON_MODULE(scoring_result)
{
  def("new_result", &new_result);
  def("new_result", &new_result_def);

  class_<vistk::scoring_result_t>("ScoringResult"
    , "A result from a scoring algorithm."
    , no_init)
    .def("true_positives", &result_true_positives)
    .def("false_positives", &result_false_positives)
    .def("total_trues", &result_total_trues)
    .def("total_possible", &result_total_possible)
    .def("percent_detection", &result_percent_detection)
    .def("precision", &result_precision)
    .def("specificity", &result_specificity)
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

vistk::scoring_result_t
new_result_def(vistk::scoring_result::count_t true_positive,
               vistk::scoring_result::count_t false_positive,
               vistk::scoring_result::count_t total_true)
{
  return boost::make_shared<vistk::scoring_result>(true_positive, false_positive, total_true);
}

vistk::scoring_result::count_t
result_true_positives(vistk::scoring_result_t const& self)
{
  return self->true_positives;
}

vistk::scoring_result::count_t
result_false_positives(vistk::scoring_result_t const& self)
{
  return self->false_positives;
}

vistk::scoring_result::count_t
result_total_trues(vistk::scoring_result_t const& self)
{
  return self->total_trues;
}

vistk::scoring_result::count_t
result_total_possible(vistk::scoring_result_t const& self)
{
  return self->total_possible;
}

vistk::scoring_result::result_t
result_percent_detection(vistk::scoring_result_t const& self)
{
  return self->percent_detection();
}

vistk::scoring_result::result_t
result_precision(vistk::scoring_result_t const& self)
{
  return self->precision();
}

vistk::scoring_result::result_t
result_specificity(vistk::scoring_result_t const& self)
{
  return self->specificity();
}

vistk::scoring_result_t
result_add(vistk::scoring_result_t const& lhs, vistk::scoring_result_t const& rhs)
{
  return (lhs + rhs);
}
