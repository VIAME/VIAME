/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file print_number_process.h
 *
 * \brief Declaration of the number printer process.
 */

namespace sprokit
{

/**
 * \class print_number_process
 *
 * \brief A process for printing numbers to a file.
 *
 * \process Prints numbers to a file.
 *
 * \iports
 *
 * \iport{number} The source of numbers to print.
 *
 * \configs
 *
 * \config{output} Where to output the numbers.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT print_number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    print_number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~print_number_process();
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

#endif // SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
