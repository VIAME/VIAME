// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_SKIP_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_SKIP_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file skip_process.h
 *
 * \brief Declaration of the skip data process.
 */

namespace sprokit
{

/**
 * \class skip_process
 *
 * \brief Generates numbers.
 *
 * \process Generates numbers.
 *
 * \iports
 *
 * \iport{input} A stream with extra data at regular intervals.
 *
 * \oports
 *
 * \oport{output} The input stream sampled at regular intervals.
 *
 * \configs
 *
 * \config{skip} The number of inputs to skip for each output.
 * \config{offset} The offset from the first datum to use for the output.
 *
 * \reqs
 *
 * \req \key{offset} must be less than \key{skip}.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT skip_process
  : public process
{
public:
  PLUGIN_INFO( "skip",
               "A process which skips input data" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  skip_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~skip_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_SKIP_PROCESS_H
