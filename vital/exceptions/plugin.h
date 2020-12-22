// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for plugin exceptions
 */

#ifndef VITAL_CORE_EXCEPTION_PLUGIN_H
#define VITAL_CORE_EXCEPTION_PLUGIN_H

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic plugin exception
class VITAL_EXCEPTIONS_EXPORT plugin_exception
  : public vital_exception
{
public:
  /// Constructor
  plugin_exception() noexcept;

  /// Destructor
  virtual ~plugin_exception() noexcept;
};

// ------------------------------------------------------------------
/// Requested factory not found.
class VITAL_EXCEPTIONS_EXPORT plugin_factory_not_found
  : public plugin_exception
{
public:
  /// Constructor
  plugin_factory_not_found( std::string const& msg) noexcept;

  /// Destructor
  virtual ~plugin_factory_not_found() noexcept;
};

// ------------------------------------------------------------------
/// Unable to create desired type.
class VITAL_EXCEPTIONS_EXPORT plugin_factory_type_creation_error
  : public plugin_exception
{
public:
  /// Constructor
  plugin_factory_type_creation_error( std::string const& msg) noexcept;

  /// Destructor
  virtual ~plugin_factory_type_creation_error() noexcept;
};

// ------------------------------------------------------------------
/// Plugin already registered
class VITAL_EXCEPTIONS_EXPORT plugin_already_exists
  : public plugin_exception
{
public:
  /// Constructor
  plugin_already_exists( std::string const& msg) noexcept;

  /// Destructor
  virtual ~plugin_already_exists() noexcept;
};

} } // end namespace

#endif /* VITAL_CORE_EXCEPTION_PLUGIN_H */
