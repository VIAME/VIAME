/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file tagged_flow_dependent_process.h
 *
 * \brief Declaration of the tagged flow dependent process.
 */

namespace vistk
{

/**
 * \class tagged_flow_dependent_process
 *
 * \brief A process which has tagged flow dependent ports.
 *
 * \process A process with flow dependent ports.
 *
 * \iports
 *
 * \iport{untagged_input} An untagged flow dependent input port.
 * \iport{tagged_input} A tagged flow dependent input port.
 *
 * \oports
 *
 * \oport{untagged_output} An untagged flow dependent output port.
 * \oport{tagged_output} A tagged flow dependent output port.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT tagged_flow_dependent_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    tagged_flow_dependent_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~tagged_flow_dependent_process();
  protected:
    /**
     * \brief Resets the process.
     */
    void _reset();

    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();
  private:
    void make_ports();

    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
