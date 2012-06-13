/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file multiplication_process.h
 *
 * \brief Declaration of the multiplication process.
 */

namespace vistk
{

/**
 * \class multiplication_process
 *
 * \brief A process which multiplies numbers.
 *
 * \process Multiplies numbers.
 *
 * \iports
 *
 * \iport{factor1} The first number to multiply.
 * \iport{factor2} The second number to multiply.
 *
 * \oports
 *
 * \oport{product} The product.
 *
 * \reqs
 *
 * \req The \port{factor1}, \port{factor2}, and \port{product} ports must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT multiplication_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    multiplication_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~multiplication_process();
  protected:
    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
