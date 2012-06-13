/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCORING_SCORING_STATISTICS_H
#define VISTK_SCORING_SCORING_STATISTICS_H

#include "scoring-config.h"

#include "scoring_result.h"
#include "statistics.h"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

/**
 * \file scoring_statistics.h
 *
 * \brief Declaration of a scoring result statistics class.
 */

namespace vistk
{

/**
 * \class scoring_statistics scoring_statistics.h <vistk/scoring/scoring_statistics.h>
 *
 * \brief Statistics about scoring results.
 */
class VISTK_SCORING_EXPORT scoring_statistics
{
  public:
    /**
     * \brief Constructor.
     */
    scoring_statistics();
    /**
     * \brief Destructor.
     */
    ~scoring_statistics();

    /**
     * \brief Add a score for statistics calculations.
     *
     * \param score The score to add as a sample.
     */
    void add_score(scoring_result_t const& score);

    /**
     * \brief Query for the statistics about the percent detection.
     *
     * \returns Percent detection statistics.
     */
    statistics_t percent_detection_stats() const;
    /**
     * \brief Query for the statistics about the precision.
     *
     * \returns Precision statistics.
     */
    statistics_t precision_stats() const;
    /**
     * \brief Query for the statistics about the specificity.
     *
     * \returns Specificity statistics.
     */
    statistics_t specificity_stats() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

/// A handle to a set of scoring result statistics.
typedef boost::shared_ptr<scoring_statistics> scoring_statistics_t;

}

#endif // VISTK_SCORING_SCORING_STATISTICS_H
