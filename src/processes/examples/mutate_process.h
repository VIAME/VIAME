/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file mutate_process.h
 *
 * \brief Declaration of the mutate process.
 */

namespace vistk
{

/**
 * \class mutate_process
 *
 * \brief A process which has an input port with the mutate flag.
 *
 * \process A process for testing mutation flags.
 *
 * \iports
 *
 * \iport{mutate} A port with the mutate flag on it.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT mutate_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    mutate_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~mutate_process();
  protected:
    /**
     * \brief Eats data from the input edge.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
