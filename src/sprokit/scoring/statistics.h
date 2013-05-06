/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_SCORING_STATISTICS_H
#define SPROKIT_SCORING_STATISTICS_H

#include "scoring-config.h"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>

/**
 * \file statistics.h
 *
 * \brief Declaration of a statistics class.
 */

namespace sprokit
{

/**
 * \class statistics statistics.h <sprokit/scoring/statistics.h>
 *
 * \brief Statistics about a sample set.
 */
class SPROKIT_SCORING_EXPORT statistics
{
  public:
    /// The type of a sample.
    typedef double data_point_t;
    /// A collection of samples.
    typedef std::vector<data_point_t> data_points_t;

    /**
     * \brief Constructor.
     *
     * \param pts The initial data sample.
     */
    explicit statistics(data_points_t const& pts = data_points_t());
    /**
     * \brief Destructor.
     */
    ~statistics();

    /**
     * \brief Add a sample to the set.
     *
     * \param pt The data sample.
     */
    void add_point(data_point_t pt);
    /**
     * \brief Add a set of samples to the set.
     *
     * \param pts The data samples.
     */
    void add_points(data_points_t const& pts);

    /**
     * \brief Query for the raw data.
     *
     * \returns The collection of all the data samples.
     */
    data_points_t data() const;

    /**
     * \brief Query for the size of the sample set.
     *
     * \returns The size of the sample set.
     */
    size_t count() const;
    /**
     * \brief Query for the sum of the data.
     *
     * \returns The sum of the data.
     */
    data_point_t sum() const;
    /**
     * \brief Query for the minimum of the data.
     *
     * \returns The minimum of the data.
     */
    data_point_t minimum() const;
    /**
     * \brief Query for the maximum of the data.
     *
     * \returns The maximum of the data.
     */
    data_point_t maximum() const;
    /**
     * \brief Query for the range of the data.
     *
     * \returns The range of the data.
     */
    data_point_t range() const;
    /**
     * \brief Query for the mean of the data.
     *
     * \returns The mean of the data.
     */
    double mean() const;
    /**
     * \brief Query for the median of the data.
     *
     * \returns The median of the data.
     */
    double median() const;
    /**
     * \brief Query for the variance of the data.
     *
     * \returns The variance of the data.
     */
    double variance() const;
    /**
     * \brief Query for the standard deviation of the data.
     *
     * \returns The standard deviation of the data.
     */
    double standard_deviation() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

/// A handle to statistics class.
typedef boost::shared_ptr<statistics> statistics_t;

}

#endif // SPROKIT_SCORING_STATISTICS_H
