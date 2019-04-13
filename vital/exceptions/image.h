/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
/// Exception for image loading
/**
 * For when image fails to load, or is corrupt.
 */
class VITAL_EXCEPTIONS_EXPORT image_load_exception
  : public image_exception
{
public:
  /// Constructor
  /**
   * \param message     Description of circumstances surrounding error.
   */
  image_load_exception(std::string message) noexcept;
  /// Destructor
  virtual ~image_load_exception() noexcept;

  /// Given error message string
  std::string m_message;
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
