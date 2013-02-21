/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_SHARED_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_SHARED_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file shared_process.h
 *
 * \brief Declaration of the shared process.
 */

namespace vistk
{

/**
 * \class shared_process
 *
 * \brief A process with a shared output port.
 *
 * \process A process with a shared output port.
 *
 * \oports
 *
 * \oport{shared} A shared datum.
 *
 * \reqs
 *
 * \req The \port{shared} output must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT shared_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    shared_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~shared_process();
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

#endif // VISTK_PROCESSES_EXAMPLES_SHARED_PROCESS_H
