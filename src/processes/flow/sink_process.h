/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SINK_PROCESS_H
#define VISTK_PROCESSES_SINK_PROCESS_H

#include "flow-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file sink_process.h
 *
 * \brief Declaration of the sink process.
 */

namespace vistk
{

/**
 * \class sink_process
 *
 * \brief A process which does nothing with incoming data.
 *
 * \process A process which does nothing with incoming data.
 *
 * \iports
 *
 * \iport{sink} The data to ignore.
 *
 * \reqs
 *
 * \req The \port{sink} port must be connected.
 */
class VISTK_PROCESSES_FLOW_NO_EXPORT sink_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    sink_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~sink_process();
  protected:
    /**
     * \brief Ignores data on the incoming edge.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SINK_PROCESS_H
