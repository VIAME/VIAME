/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "statistics.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <limits>

/**
 * \file statistics.cxx
 *
 * \brief Implementation of a statistics class.
 */

namespace ba = boost::accumulators;

namespace vistk
{

class statistics::priv
{
  public:
    priv();
    ~priv();

    typedef ba::accumulator_set<data_point_t, ba::stats
      < ba::tag::count
      , ba::tag::sum
      , ba::tag::min
      , ba::tag::max
      , ba::tag::mean
      , ba::tag::median(ba::with_p_square_quantile)
      , ba::tag::variance(ba::lazy)
      > > stats_t;

    stats_t acc;
    data_points_t data;
};

statistics
::statistics(data_points_t const& pts)
  : d(new priv)
{
  add_points(pts);
}

statistics
::~statistics()
{
}

void
statistics
::add_point(data_point_t pt)
{
  d->acc(pt);
  d->data.push_back(pt);
}

void
statistics
::add_points(data_points_t const& pts)
{
  d->acc = std::for_each(pts.begin(), pts.end(), d->acc);
  d->data.insert(d->data.end(), pts.begin(), pts.begin());
}

statistics::data_points_t
statistics
::data() const
{
  return d->data;
}

size_t
statistics
::count() const
{
  return ba::count(d->acc);
}

statistics::data_point_t
statistics
::sum() const
{
  return ba::sum(d->acc);
}

statistics::data_point_t
statistics
::minimum() const
{
  return ba::min(d->acc);
}

statistics::data_point_t
statistics
::maximum() const
{
  return ba::max(d->acc);
}

statistics::data_point_t
statistics
::range() const
{
  return (maximum() - minimum());
}

double
statistics
::mean() const
{
  return ba::mean(d->acc);
}

double
statistics
::median() const
{
  return ba::median(d->acc);
}

double
statistics
::variance() const
{
  return ba::variance(d->acc);
}

double
statistics
::standard_deviation() const
{
  return sqrt(variance());
}

statistics::priv
::priv()
  : acc()
  , data()
{
}

statistics::priv
::~priv()
{
}

}
