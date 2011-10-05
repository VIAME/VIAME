/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_SOURCE_PROCESS_H
#define VISTK_PROCESSES_SOURCE_PROCESS_H

#include "flow-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file source_process.h
 *
 * \brief Declaration of the source process.
 */

namespace vistk
{

/**
 * \class source_process
 *
 * \brief A process which provides a consistent stamp color.
 *
 * \process A process which provides a consistent stamp color.
 *
 * \oports
 *
 * \oport{color} The consistently colored port.
 *
 * \reqs
 *
 * \req The \port{color} port must be connected.
 */
class VISTK_PROCESSES_FLOW_NO_EXPORT source_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    source_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~source_process();
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Ignores data on the incoming edge.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SOURCE_PROCESS_H
