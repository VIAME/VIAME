/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file const_number_process.h
 *
 * \brief Declaration of the constant number process.
 */

namespace vistk
{

/**
 * \class const_number_process
 *
 * \brief Generates constant numbers.
 *
 * \process Generates constant numbers.
 *
 * \oports
 *
 * \oport{number} The number.
 *
 * \configs
 *
 * \config{value} The first number to use.
 *
 * \reqs
 *
 * \req The \port{number} output must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT const_number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    const_number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~const_number_process();
  protected:
    /**
     * \brief Configure the process.
     */
    void _configure();

    /**
     * \brief Step the process.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H
