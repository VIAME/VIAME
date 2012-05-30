/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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
 * \brief A process for normalizing streams of data into a single stream.
 *
 * \process Normalizes streams of data into a single stream.
 *
 * \iports
 *
 * \iport{src/\portvar{tag}} The input stream for \portvar{tag}.
 *
 * \oports
 *
 * \oport{out/\portvar{tag}} Recolored output stream for \portvar{tag}.
 *
 * \reqs
 *
 * \req All \port{src/\portvar{tag}} and \port{out/\portvar{tag}} ports must be connected.
 *
 * \ingroup process_flow
 */
class VISTK_PROCESSES_FLOW_NO_EXPORT source_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    source_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~source_process();
  protected:
    /**
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Reset the process.
     */
    void _reset();

    /**
     * \brief Step the process.
     */
    void _step();

    /**
     * \brief Input port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an input port.
     */
    port_info_t _input_port_info(port_t const& port);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_SOURCE_PROCESS_H
