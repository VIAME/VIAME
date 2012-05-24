/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scoring_statistics.h"

namespace vistk
{

class scoring_statistics::priv
{
  public:
    priv();
    ~priv();

    statistics_t percent_detection;
    statistics_t precision;
    statistics_t specificity;
};

scoring_statistics
::scoring_statistics()
  : d(new priv)
{
}

scoring_statistics
::~scoring_statistics()
{
}

void
scoring_statistics
::add_score(scoring_result_t const& score)
{
  d->percent_detection->add_point(score->percent_detection());
  d->precision->add_point(score->precision());
  d->specificity->add_point(score->specificity());
}

statistics_t
scoring_statistics
::percent_detection_stats() const
{
  return d->percent_detection;
}

statistics_t
scoring_statistics
::precision_stats() const
{
  return d->precision;
}

statistics_t
scoring_statistics
::specificity_stats() const
{
  return d->specificity;
}

scoring_statistics::priv
::priv()
  : percent_detection(new statistics)
  , precision(new statistics)
  , specificity(new statistics)
{
}

scoring_statistics::priv
::~priv()
{
}

}
