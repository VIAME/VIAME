/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCORING_SCORING_STATISTICS_H
#define VISTK_SCORING_SCORING_STATISTICS_H

#include "scoring-config.h"

#include "scoring_result.h"
#include "statistics.h"

#include <boost/scoped_ptr.hpp>

namespace vistk
{

class VISTK_SCORING_EXPORT scoring_statistics
{
  public:
    scoring_statistics();
    ~scoring_statistics();

    void add_score(scoring_result_t const& score);

    statistics_t percent_detection_stats() const;
    statistics_t precision_stats() const;
    statistics_t specificity_stats() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_SCORING_SCORING_STATISTICS_H
