// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file print_number_process.h
 *
 * \brief Declaration of the number printer process.
 */

namespace sprokit {

/**
 * \class print_number_process
 *
 * \brief A process for printing numbers to a file.
 *
 * \process Prints numbers to a file.
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
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT print_number_process
  : public process
{
public:
  PLUGIN_INFO( "print_number",
               "Print numbers to a file" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  print_number_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~print_number_process();

protected:
  /**
   * \brief Configure the process.
   */
  void _configure() override;

  /**
   * \brief Reset the process.
   */
  void _reset() override;

  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
