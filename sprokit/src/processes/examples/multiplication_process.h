// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file multiplication_process.h
 *
 * \brief Declaration of the multiplication process.
 */

namespace sprokit {

/**
 * \class multiplication_process
 *
 * \brief A process which multiplies numbers.
 *
 * \process Multiplies numbers.
 *
 * \iports
 *
 * \iport{factor1} The first number to multiply.
 * \iport{factor2} The second number to multiply.
 *
 * \oports
 *
 * \oport{product} The product.
 *
 * \reqs
 *
 * \req The \port{factor1}, \port{factor2}, and \port{product} ports must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT multiplication_process
  : public process
{
public:
  PLUGIN_INFO( "multiplication",
               "Multiplies numbers" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */

  multiplication_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~multiplication_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
