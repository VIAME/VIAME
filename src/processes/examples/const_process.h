/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file const_process.h
 *
 * \brief Declaration of the const process.
 */

namespace sprokit
{

/**
 * \class const_process
 *
 * \brief A process with a const output port.
 *
 * \process A process with a const output port.
 *
 * \oports
 *
 * \oport{const} A constant datum.
 *
 * \reqs
 *
 * \req The \port{const} output must be connected.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT const_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    const_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~const_process();
  protected:
    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H
