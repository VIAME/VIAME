/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_SINK_PROCESS_H
#define SPROKIT_PROCESSES_SINK_PROCESS_H

#include "flow-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file sink_process.h
 *
 * \brief Declaration of the sink process.
 */

namespace sprokit
{

/**
 * \class sink_process
 *
 * \brief A process for doing nothing with a data stream.
 *
 * \process Ignores incoming data.
 *
 * \iports
 *
 * \iport{sink} The data to ignore.
 *
 * \reqs
 *
 * \req The \port{sink} port must be connected.
 *
 * \ingroup process_flow
 */
class SPROKIT_PROCESSES_FLOW_NO_EXPORT sink_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    sink_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~sink_process();
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

#endif // SPROKIT_PROCESSES_SINK_PROCESS_H
