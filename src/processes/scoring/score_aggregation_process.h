/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SCORING_SCORE_AGGREGATION_PROCESS_H
#define VISTK_PROCESSES_SCORING_SCORE_AGGREGATION_PROCESS_H

#include "scoring-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file score_aggregation_process.h
 *
 * \brief Declaration of the score aggregation process.
 */

namespace vistk
{

/**
 * \class score_aggregation_process
 *
 * \brief A process which generates an aggregate score.
 *
 * \process Aggregate scores from a scoring process.
 *
 * \iports
 *
 * \iport{score} The score.
 *
 * \oports
 *
 * \oport{aggregate} The aggregate score.
 * \oport{statistics} The aggregate score statistics.
 *
 * \reqs
 *
 * \req The ports \port{score} and \port{aggregate} must be connected.
 *
 * \ingroup process_scoring
 */
class VISTK_PROCESSES_SCORING_NO_EXPORT score_aggregation_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    score_aggregation_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~score_aggregation_process();
  protected:
    /**
     * \brief Step the subclass.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SCORING_SCORE_AGGREGATION_PROCESS_H
