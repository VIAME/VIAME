/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scoring_result.h"

#include <boost/make_shared.hpp>

namespace vistk
{

scoring_result
::scoring_result(count_t hit, count_t miss, count_t truth)
  : hit_count(hit)
  , miss_count(miss)
  , truth_count(truth)
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
  result_t const hit = hit_count;
  result_t const truth = truth_count;

  return (hit / truth);
}

scoring_result::result_t
scoring_result
::precision() const
{
  result_t const hit = hit_count;
  result_t const miss = miss_count;

  return (hit / (hit + miss));
}

scoring_result_t
operator + (scoring_result_t const& lhs, scoring_result_t const& rhs)
{
  return boost::make_shared<scoring_result>(
    lhs->hit_count + rhs->hit_count,
    lhs->miss_count + rhs->miss_count,
    lhs->truth_count + rhs->truth_count);
}

}
