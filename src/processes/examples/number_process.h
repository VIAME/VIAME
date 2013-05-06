/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file number_process.h
 *
 * \brief Declaration of the number process.
 */

namespace sprokit
{

/**
 * \class number_process
 *
 * \brief Generates numbers.
 *
 * \process Generates numbers.
 *
 * \oports
 *
 * \oport{number} The generated number.
 *
 * \configs
 *
 * \config{start} The first number to use.
 * \config{end} The last number to use.
 *
 * \reqs
 *
 * \req \key{start} must be less than \key{end}.
 * \req The \port{number} output must be connected.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~number_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
