/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file data_dependent_process.h
 *
 * \brief Declaration of the data dependent process.
 */

namespace vistk
{

/**
 * \class data_dependent_process
 *
 * \brief A process which has a data dependent port.
 *
 * \process A process with a data dependent port.
 *
 * \configs
 *
 * \config{reject} Whether to reject the set type or not.
 * \config{set_on_configure} Whether to set the type on configure or not.
 *
 * \oports
 *
 * \oport{output} A data dependent output port.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT data_dependent_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    data_dependent_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~data_dependent_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();
    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();
    /**
     * \brief Sets the type for an output port.
     *
     * \param port The name of the port.
     * \param type The type of the connected port.
     *
     * \returns True if the type can work, false otherwise.
     */
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H
