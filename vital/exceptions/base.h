// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL base exception interface
 */

#ifndef VITAL_EXCEPTIONS_BASE_H
#define VITAL_EXCEPTIONS_BASE_H

#include <vital/vital_config.h>
#include <vital/exceptions/vital_exceptions_export.h>

#include <string>
#include <exception>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// The base class for all vital exceptions
/**
 * \ingroup exceptions
 */
class VITAL_EXCEPTIONS_EXPORT vital_exception
  : public std::exception
{
public:
  /// Constructor
  vital_exception() noexcept;

  /// Destructor
  virtual ~vital_exception() noexcept;

  /**
   * \brief Description of the exception
   * \returns A string describing what went wrong.
   */
  virtual char const* what() const noexcept;

  /**
   * \brief Description of the exception with source location.
   *
   * This method returns the description of the exception with the
   * source location from whence it is thrown, if available. If the
   * source location is not available, then the return value looks
   * like that from the what() method.
   *
   * \return A string with the origin and description of the exception.
   */
  std::string what_loc() const noexcept;

  /**
   * \brief Set optional location of exception.
   *
   * This method saves the supplied source file and location in the
   * exception so the location where the exception was thrown can be
   * determined. This is not that useful for an end user, but it is
   * very helpful for developers.
   *
   * \param file Name of source file.
   * \param line Line number in file.
   */
  void set_location( std::string const& file, int line );

protected:
  /// descriptive string as to what happened to cause the exception.
  std::string m_what;

  std::string m_file_name;
  int m_line_number;

private:
  mutable std::string m_what_loc;
};

// ------------------------------------------------------------------
/// Exception for incorrect input values
/**
 * \ingroup exceptions
 */
class VITAL_EXCEPTIONS_EXPORT invalid_value
  : public vital_exception
{
public:
  /// Constructor
  invalid_value(std::string reason) noexcept;
  /// Destructor
  virtual ~invalid_value() noexcept;
};

} } // end namespace vital

// ------------------------------------------------------------------
///Exception helper macro.
/**
 * Macro to simplify creating exceptions. The source
 * location of this macro is also recorded in the exception.
 *
 * The number and type of parameters depends on the type of exception
 * being thrown.
 *
 * @param E       Exception type.
 * @param ...     exception parameters
 */
#define VITAL_THROW(E, ...) do {                \
    E except{ __VA_ARGS__ };                    \
    except.set_location( __FILE__, __LINE__ );  \
    throw except;                               \
  } while (0)

#endif // VITAL_CORE_EXCEPTIONS_BASE_H
