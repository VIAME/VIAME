/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SCORING_MASK_SCORING_PROCESS_H
#define VISTK_PROCESSES_SCORING_MASK_SCORING_PROCESS_H

#include "scoring-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file mask_scoring_process.h
 *
 * \brief Declaration of the mask scoring process.
 */

namespace vistk
{

/**
 * \class mask_scoring_process
 *
 * \brief A process which scores binary masks against each other.
 *
 * \process Score a mask against another.
 *
 * \iports
 *
 * \iport{computed_mask} The computed mask.
 * \iport{truth_mask} The expected mask.
 *
 * \oports
 *
 * \oport{result} The scoring results.
 *
 * \reqs
 *
 * \req The input ports \port{computed_mask} and \port{truth_mask} must be connected.
 * \req The output port \port{result} must be connected.
 *
 * \ingroup process_scoring
 */
class VISTK_PROCESSES_SCORING_NO_EXPORT mask_scoring_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    mask_scoring_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~mask_scoring_process();
  protected:
    /**
     * \brief Collate data from the input edges.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SCORING_MASK_SCORING_PROCESS_H
