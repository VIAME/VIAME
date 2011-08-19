/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class print_string_process
 *
 * \brief A process which prints incoming strings.
 *
 * \process A process for printing strings.
 *
 * \iports
 *
 * \iport{string} The source of strings to print.
 *
 * \configs
 *
 * \config{output} Where to output the strings.
 *
 * \reqs
 *
 * \req The \port{string} port must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT print_string_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    print_string_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~print_string_process();
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
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H
