// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file take_string_process.h
 *
 * \brief Declaration of the string taking process.
 */

namespace sprokit {

/**
 * \class take_string_process
 *
 * \brief A process which takes strings.
 *
 * \process Takes strings.
 *
 * \iports
 *
 * \iport{string} The source of strings to take.
 *
 * \reqs
 *
 * \req The \port{string} port must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT take_string_process
  : public process
{
public:
  PLUGIN_INFO( "take_string",
               "Print strings to a file" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  take_string_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~take_string_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_TAKE_STRING_PROCESS_H
