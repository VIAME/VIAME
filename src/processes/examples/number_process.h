/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file number_process.h
 *
 * \brief Declaration of the number process.
 */

namespace vistk
{

/**
 * \class number_process
 *
 * \brief A process which generates increasing numbers within a range.
 *
 * \process A process for generating numbers.
 *
 * \iports
 *
 * \iport{color} The color to use for the output stamps.
 *
 * \oports
 *
 * \oport{number} The number generated for the step.
 *
 * \configs
 *
 * \config{start} The start of the range.
 * \config{end} The end of the range.
 *
 * \reqs
 *
 * \req \key{start} must be less than \key{end}.
 * \req The \port{number} output must be connected to at least one edge.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~number_process();
  protected:
    /**
     * \brief Checks the configuration.
     */
    void _configure();

    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
