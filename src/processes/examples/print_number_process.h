/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

namespace vistk
{

/**
 * \class print_number_process
 *
 * \brief A process which prints incoming numbers.
 *
 * \process A process for printing numbers.
 *
 * \iports
 *
 * \iport{number} The source of numbers to print.
 *
 * \configs
 *
 * \config{output} Where to output the numbers.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT print_number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    print_number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~print_number_process();
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Prints numbers to the output stream.
     */
    void _step();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
