/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file tunable_process.h
 *
 * \brief Declaration of the tunable process.
 */

namespace vistk
{

/**
 * \class tunable_process
 *
 * \brief A tunable process.
 *
 * \process Outputs a tunable result.
 *
 * \oports
 *
 * \oport{tunable} The tunable output.
 * \oport{non_tunable} The non-tunable output.
 *
 * \configs
 *
 * \config{tunable} The tunable value to use.
 * \config{non_tunable} The non-tunable value to use.
 *
 * \reqs
 *
 * \req The \port{tunable} output must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT tunable_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    tunable_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~tunable_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Step the process.
     */
    void _step();

    /**
     * \brief Step the process.
     */
    void _reconfigure(config_t const& conf);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H
