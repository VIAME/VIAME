/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_PASS_PROCESS_H
#define SPROKIT_PROCESSES_PASS_PROCESS_H

#include "flow-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file pass_process.h
 *
 * \brief Declaration of the pass process.
 */

namespace sprokit
{

/**
 * \class pass_process
 *
 * \brief A process to pass through a data stream.
 *
 * \process Passes through incoming data.
 *
 * \iports
 *
 * \iport{pass} The datum to pass.
 *
 * \oports
 *
 * \oport{pass} The passed datum.
 *
 * \reqs
 *
 * \req The \port{pass} ports must be connected.
 *
 * \ingroup process_flow
 */
class SPROKIT_PROCESSES_FLOW_NO_EXPORT pass_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    pass_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~pass_process();
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

#endif // SPROKIT_PROCESSES_PASS_PROCESS_H
