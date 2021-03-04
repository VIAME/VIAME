// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file take_number_process.h
 *
 * \brief Declaration of the number taking process.
 */

namespace sprokit {

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
class PROCESSES_EXAMPLES_NO_EXPORT take_number_process
  : public process
{
public:
  PLUGIN_INFO( "take_number",
               "Print numbers to a file" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  take_number_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~take_number_process();

protected:
  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_TAKE_NUMBER_PROCESS_H
