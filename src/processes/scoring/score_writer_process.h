/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SCORING_SCORE_WRITER_PROCESS_H
#define VISTK_PROCESSES_SCORING_SCORE_WRITER_PROCESS_H

#include "scoring-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file score_writer_process.h
 *
 * \brief Declaration of the score writer process.
 */

namespace vistk
{

/**
 * \class score_writer_process
 *
 * \brief A process which writes out scores to a file.
 *
 * \process Write scores from a scoring process.
 *
 * \iports
 *
 * \iport{score} The score.
 *
 * \configs
 *
 * \oport{path} The path to output the scores into.
 *
 * \reqs
 *
 * \req The port \port{score} must be connected.
 *
 * \ingroup process_scoring
 */
class VISTK_PROCESSES_SCORING_NO_EXPORT score_writer_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    score_writer_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~score_writer_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Reset the process.
     */
    void _reset();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SCORING_SCORE_WRITER_PROCESS_H
