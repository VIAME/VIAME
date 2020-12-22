// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL Exceptions pertaining to serialization operations
 */

#ifndef VITAL_CORE_EXCEPTIONS_SERIALIZATION_H_
#define VITAL_CORE_EXCEPTIONS_SERIALIZATION_H_

#include <string>

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
class VITAL_EXCEPTIONS_EXPORT serialization_exception
  : public vital_exception
{
public:
  /// Constructor
  serialization_exception( std::string const& str );

  virtual ~serialization_exception() noexcept;
};

} } // end namespace

#endif // VITAL_CORE_EXCEPTIONS_SERIALIZATION_H_
