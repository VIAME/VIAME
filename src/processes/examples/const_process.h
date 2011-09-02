/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_CONST_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_CONST_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file const_process.h
 *
 * \brief Declaration of the const process.
 */

namespace vistk
{

/**
 * \class const_process
 *
 * \brief A process which has a const output port.
 *
 * \process A process with a const output port.
 *
 * \oports
 *
 * \oport{const} The datum generated for the step.
 *
 * \reqs
 *
 * \req The \port{const} output must be connected to at least one edge.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT const_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    const_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~const_process();
  protected:
    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_CONST_PROCESS_H
