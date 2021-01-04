// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_DUPLICATE_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_DUPLICATE_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file duplicate_process.h
 *
 * \brief Declaration of the duplicate process.
 */

namespace sprokit
{

/**
 * \class duplicate_process
 *
 * \brief Duplicates input to a single output.
 *
 * \process Duplicates input to a single output.
 *
 * \iports
 *
 * \iport{input} Arbitrary input data.
 *
 * \oports
 *
 * \oport{duplicate} Duplicated input data.
 *
 * \configs
 *
 * \config{copies} The number of copies to make per input.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT duplicate_process
  : public process
{
public:
  PLUGIN_INFO( "duplicate",
               "A process which duplicates input" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  duplicate_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~duplicate_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_DUPLICATE_PROCESS_H
