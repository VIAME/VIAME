/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCORING_STATISTICS_H
#define VISTK_SCORING_STATISTICS_H

#include "scoring-config.h"

#include <boost/scoped_ptr.hpp>

#include <vector>

namespace vistk
{

class VISTK_SCORING_EXPORT statistics
{
  public:
    typedef double data_point_t;
    typedef std::vector<data_point_t> data_points_t;

    explicit statistics(data_points_t const& pts = data_points_t());
    ~statistics();

    void add_point(data_point_t pt);
    void add_points(data_points_t const& pts);

    data_points_t data() const;

    size_t count() const;
    data_point_t sum() const;
    data_point_t minimum() const;
    data_point_t maximum() const;
    data_point_t range() const;
    double mean() const;
    double median() const;
    double variance() const;
    double standard_deviation() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_SCORING_STATISTICS_H
