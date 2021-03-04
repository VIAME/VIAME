// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL Exceptions pertaining to iteration.
 */

#ifndef KWIVER_VITAL_EXCEPTIONS_ITERATION_H_
#define KWIVER_VITAL_EXCEPTIONS_ITERATION_H_

#include <vital/exceptions/base.h>
#include <string>

namespace kwiver {
namespace vital {

/// Exception thrown from next value function to signify the end of iteration.
class VITAL_EXCEPTIONS_EXPORT stop_iteration_exception
  : public vital_exception
{
public:
  /// Constructor
  stop_iteration_exception( std::string const& container) noexcept;

  /// Destructor
  virtual ~stop_iteration_exception() noexcept = default;
};

} } //end namespaces

#endif //KWIVER_VITAL_EXCEPTIONS_ITERATION_H_
