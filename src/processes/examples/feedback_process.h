/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file feedback_process.h
 *
 * \brief Declaration of the feedback process.
 */

namespace sprokit
{

/**
 * \class feedback_process
 *
 * \brief A process with its own backwards edge.
 *
 * \process A process with its own backwards edge.
 *
 * \iports
 *
 * \iport{input} The datum generated the previous step.
 *
 * \oports
 *
 * \oport{output} The datum generated for the step.
 *
 * \reqs
 *
 * \req The ports \port{input} and \port{output} must be connected.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT feedback_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    feedback_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~feedback_process();
  protected:
    /**
     * \brief Flush the process.
     */
    void _flush();
    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H
