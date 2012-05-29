/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scoring_result.h"

#include <boost/make_shared.hpp>

/**
 * \file scoring_result.cxx
 *
 * \brief Implementation of a scoring result class.
 */

namespace vistk
{

scoring_result
::scoring_result(count_t true_positive, count_t false_positive, count_t total_true, count_t possible)
  : true_positives(true_positive)
  , false_positives(false_positive)
  , total_trues(total_true)
  , total_possible(possible)
{
}

scoring_result
::~scoring_result()
{
}

scoring_result::result_t
scoring_result
::percent_detection() const
{
  if (!total_trues)
  {
    return result_t(0);
  }

  return (result_t(true_positives) / result_t(total_trues));
}

scoring_result::result_t
scoring_result
::precision() const
{
  count_t const total = true_positives + false_positives;

  if (!total)
  {
    return result_t(0);
  }

  return (result_t(true_positives) / result_t(total));
}

scoring_result::result_t
scoring_result
::specificity() const
{
  if (!total_possible)
  {
    return result_t(0);
  }

  count_t const truth_negative = total_possible - true_positives;
  count_t const true_negative = truth_negative - false_positives;

  if (!truth_negative)
  {
    return result_t(0);
  }

  return (result_t(true_negative) / result_t(truth_negative));
}

scoring_result_t
operator + (scoring_result_t const& lhs, scoring_result_t const& rhs)
{
  return boost::make_shared<scoring_result>(
    lhs->true_positives + rhs->true_positives,
    lhs->false_positives + rhs->false_positives,
    lhs->total_trues + rhs->total_trues,
    lhs->total_possible + rhs->total_possible);
}

}
