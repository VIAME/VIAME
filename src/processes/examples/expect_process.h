/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H

#include "examples-config.h"

#include <sprokit/pipeline/process.h>

/**
 * \file expect_process.h
 *
 * \brief Declaration of the expect process.
 */

namespace sprokit
{

/**
 * \class expect_process
 *
 * \brief A process which checks values.
 *
 * \process A process which checks values.
 *
 * \oports
 *
 * \oport{dummy} A dummy port.
 *
 * \configs
 *
 * \config{tunable} A tunable parameter.
 * \config{expect} The expected string.
 * \config{expect_key} Whether to expect a key or value.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT expect_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    expect_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~expect_process();
  protected:
    void _configure();
    void _step();
    void _reconfigure(config_t const& conf);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H
