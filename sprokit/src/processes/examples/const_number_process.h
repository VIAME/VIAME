// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file const_number_process.h
 *
 * \brief Declaration of the constant number process.
 */

namespace sprokit {

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
class PROCESSES_EXAMPLES_NO_EXPORT const_number_process
  : public process
{
public:
  PLUGIN_INFO( "const_number",
               "Outputs a constant number" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  const_number_process(kwiver::vital::config_block_sptr const &config);
  /**
   * \brief Destructor.
   */
  ~const_number_process();

protected:
  /**
   * \brief Configure the process.
   */
  void _configure() override;

  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};
}

#endif // SPROKIT_PROCESSES_EXAMPLES_CONST_NUMBER_PROCESS_H
