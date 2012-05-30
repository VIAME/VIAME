/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file take_number_process.h
 *
 * \brief Declaration of the number taking process.
 */

namespace vistk
{

/**
 * \class take_number_process
 *
 * \brief A process which takes numbers.
 *
 * \process Takes numbers.
 *
 * \iports
 *
 * \iport{number} The source of numbers to take.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT take_number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    take_number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~take_number_process();
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

#endif // VISTK_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H
