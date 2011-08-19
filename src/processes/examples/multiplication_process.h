/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class multiplication_process
 *
 * \brief A process which multiplies incoming numbers.
 *
 * \process A process for multiplying numbers.
 *
 * \iports
 *
 * \iport{factor1} The first number to multiply.
 * \iport{factor2} The second number to multiply.
 *
 * \oports
 *
 * \oport{product} The number generated for the step.
 *
 * \reqs
 *
 * \req The \port{factor1}, \port{factor2}, and \port{product} ports must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT multiplication_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    multiplication_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~multiplication_process();
  protected:
    /**
     * \brief Multiplies numbers and outputs the result.
     */
    void _step();
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
