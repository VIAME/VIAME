// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file feedback_process.h
 *
 * \brief Declaration of the feedback process.
 */

namespace sprokit
{

/**
 * \class feedback_process
 *
 * \brief A process with its own backwards edge.
 *
 * \process A process with its own backwards edge.
 *
 * \iports
 *
 * \iport{input} The datum generated the previous step.
 *
 * \oports
 *
 * \oport{output} The datum generated for the step.
 *
 * \reqs
 *
 * \req The ports \port{input} and \port{output} must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT feedback_process
  : public process
{
public:
  PLUGIN_INFO( "feedback",
               "A process which feeds data into itself" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  feedback_process(kwiver::vital::config_block_sptr const &config);
  /**
   * \brief Destructor.
   */
  ~feedback_process();

protected:
  /**
   * \brief Flush the process.
   */
  void _flush() override;
  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

} // namespace sprokit

#endif // SPROKIT_PROCESSES_EXAMPLES_FEEDBACK_PROCESS_H
