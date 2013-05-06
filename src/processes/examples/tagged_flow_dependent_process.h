/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file tagged_flow_dependent_process.h
 *
 * \brief Declaration of the tagged flow dependent process.
 */

namespace sprokit
{

/**
 * \class tagged_flow_dependent_process
 *
 * \brief A process with tagged flow dependent ports.
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
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT tagged_flow_dependent_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    tagged_flow_dependent_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~tagged_flow_dependent_process();
  protected:
    /**
     * \brief Reset the process.
     */
    void _reset();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    void make_ports();

    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
