// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

/**
 * \file number_process.h
 *
 * \brief Declaration of the number process.
 */

namespace sprokit
{

/**
 * \class number_process
 *
 * \brief Generates numbers.
 *
 * \process Generates numbers.
 *
 * \oports
 *
 * \oport{number} The generated number.
 *
 * \configs
 *
 * \config{start} The first number to use.
 * \config{end} The last number to use.
 *
 * \reqs
 *
 * \req \key{start} must be less than \key{end}.
 * \req The \port{number} output must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT number_process
  : public process
{
public:
  PLUGIN_INFO( "numbers",
               "Outputs numbers within a range" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  number_process(kwiver::vital::config_block_sptr const &config);
  /**
   * \brief Destructor.
   */
  ~number_process();

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

} // namespace sprokit

#endif // SPROKIT_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
