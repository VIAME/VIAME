// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL Exceptions pertaining to image operations and manipulation
 */

#ifndef VITAL_CORE_EXCEPTIONS_IMAGE_H
#define VITAL_CORE_EXCEPTIONS_IMAGE_H

#include <string>

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic image exception
class VITAL_EXCEPTIONS_EXPORT image_exception
  : public vital_exception
{
public:
  /// Constructor
  /**
   * \param message     Description of circumstances surrounding error.
   */
  image_exception(
    std::string const& message = "unspecified image exception" ) noexcept;

  /// Destructor
  virtual ~image_exception() noexcept;

protected:
  /// Constructor
  image_exception( std::nullptr_t ) noexcept;
};

// ------------------------------------------------------------------
/// Exception for image type mismatch
/**
 * For when image type equality must be asserted.
 */
class VITAL_EXCEPTIONS_EXPORT image_type_mismatch_exception
  : public image_exception
{
public:
  /// Constructor
  /**
   * \param message     Description of circumstances surrounding error.
   */
  image_type_mismatch_exception( std::string const& message ) noexcept;

  /// Destructor
  virtual ~image_type_mismatch_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for image sizing mismatch
/**
 * For when image shape/size equality must be asserted.
 */
class VITAL_EXCEPTIONS_EXPORT image_size_mismatch_exception
  : public image_exception
{
public:
  /// Constructor
  /**
   * \param message     Description of circumstances surrounding error.
   * \param correct_w   Correct image width
   * \param correct_h   Correct image height
   * \param given_w     Actual image width
   * \param given_h     Actual image height
   */
  image_size_mismatch_exception( std::string const& message,
                                 size_t correct_w, size_t correct_h,
                                 size_t given_w, size_t given_h ) noexcept;
  /// Destructor
  virtual ~image_size_mismatch_exception() noexcept;

  /// Given error message string
  std::string const m_message;
  /// The correct pixel width and height
  size_t const m_correct_w;
  size_t const m_correct_h;
  /// The incorrect, given pixel width and height
  size_t const m_given_w;
  size_t const m_given_h;
};

} } // end namespace

#endif // VITAL_CORE_EXCEPTIONS_IMAGE_H
