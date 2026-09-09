// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief VITAL Exceptions pertaining to metadata operations
///

#ifndef VITAL_CORE_EXCEPTIONS_METADATA_H_
#define VITAL_CORE_EXCEPTIONS_METADATA_H_

#include <string>

#include <viame/algorithm_framework/exceptions/base.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Generic metadata exception
class VITAL_EXCEPTIONS_EXPORT metadata_exception
  : public vital_exception
{
public:
  /// Constructor
  metadata_exception( std::string const& str );

  virtual ~metadata_exception() noexcept;
};

// ----------------------------------------------------------------------------
/// Data buffer is too small to complete read or write operation.
class VITAL_EXCEPTIONS_EXPORT metadata_buffer_overflow
  : public metadata_exception
{
public:
  metadata_buffer_overflow( std::string const& str );

  virtual ~metadata_buffer_overflow() noexcept;
};

// ----------------------------------------------------------------------------
/// Size of value is greater than data type or format allows.
class VITAL_EXCEPTIONS_EXPORT metadata_type_overflow
  : public metadata_exception
{
public:
  metadata_type_overflow( std::string const& str );

  virtual ~metadata_type_overflow() noexcept;
};

} // namespace vital

} // namespace kwiver

#endif
