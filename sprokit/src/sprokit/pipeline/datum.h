/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_DATUM_H
#define SPROKIT_PIPELINE_DATUM_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "types.h"

#include <vital/any.h>
#include <boost/operators.hpp>

#include <string>

/**
 * \file datum.h
 *
 * \brief Header for a piece of \link sprokit::datum data\endlink in the pipeline.
 */

namespace sprokit {

/**
 * \class datum datum.h <sprokit/pipeline/datum.h>
 *
 * \brief A wrapper for data that passes through an \ref edge in the \ref pipeline.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT datum
  : boost::equality_comparable<sprokit::datum>
{
  public:
    /// Information about an error that occurred within a process.
    typedef std::string error_t;

    /// The type of the datum being passed through the edge.
    /// Order of this enum is important.
    typedef enum
    {
      /// Data is included in the edge.
      data,
      /// No data was computed for the computation.
      empty,
      /// An error occurred when computing the data.
      error,
      /// An invalid type.
      invalid,
      /// The current data stream is complete and a new one will follow.
      flush,
      /// The process is complete and no more data will be available on this edge.
      complete
    } type_t;

    /**
     * \brief Create a datum with the #data type.
     *
     * This method is for bindings to be able to create kwiver::vital::any objects
     * manually.
     *
     * \param dat The data to pass through the edge.
     *
     * \returns A new datum containing a result.
     */
    static datum_t new_datum(kwiver::vital::any const& dat);

    /**
     * \brief Create a datum with the #data type.
     *
     * \param dat The data to pass through the edge.
     *
     * \returns A new datum containing a result.
     */
    template <typename T>
    static datum_t new_datum(T const& dat);

    /**
     * \brief Create a datum with the #empty type.
     *
     * \returns A new datum which indicates that a result could not be computed.
     */
    static datum_t empty_datum();

    /**
     * \brief Create a datum with the #flush type.
     *
     * \returns A new datum which indicates that the current data stream is complete.
     */
    static datum_t flush_datum();

    /**
     * \brief Create a datum with the #complete type.
     *
     * \returns A new datum which indicates that the calculation of results is complete.
     */
    static datum_t complete_datum();

    /**
     * \brief Create a datum with the #error type.
     *
     * \param error Information about the error that occurred.
     *
     * \returns A new datum that indicates that an error occurred.
     */
    static datum_t error_datum(error_t const& error);

    /**
     * \brief Query a datum for the type.
     *
     * This method returns the sprokit type of the datum.
     *
     * \returns The type of the datum.
     */
    type_t type() const;

    /**
     * \brief Query for the error that occurred.
     *
     * This method returns the error code that is associated with an
     * error type datum. The error text is set by the CTOR when an
     * error type datum is created.
     *
     * \returns The error that occurred.
     */
    error_t get_error() const;

    /**
     * \brief Extract a result from a datum.
     *
     * \throws bad_datum_cast_exception Thrown when the data cannot be cast as requested.
     *
     * \returns The result contained within the datum.
     */
    template <typename T>
    T get_datum() const;

    /**
     * \brief Compare two data for equality.
     *
     * \note This returns false for two data packets which point to the same
     * internal data since \c kwiver::vital::any does not give access to it without
     * knowing the type.
     *
     * \param dat The datum to compare to.
     *
     * \returns True if \p dat and \c *this definitely have the same value, false otherwise.
     */
    bool operator == (datum const& dat) const;

  private:
    SPROKIT_PIPELINE_NO_EXPORT datum(type_t ty);
    SPROKIT_PIPELINE_NO_EXPORT datum(error_t const& err);
    SPROKIT_PIPELINE_NO_EXPORT datum(kwiver::vital::any const& dat);

    type_t const m_type;
    error_t const m_error;
    kwiver::vital::any const m_datum;
};

// ----------------------------------------------------------------------------
/**
 * \class datum_exception datum.h <sprokit/pipeline/datum.h>
 *
 * \brief The base class for all exceptions thrown from \ref datum.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT datum_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    datum_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~datum_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class bad_datum_cast_exception datum.h <sprokit/pipeline/datum.h>
 *
 * \brief Thrown when the \ref datum cannot be converted to the requested type.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT bad_datum_cast_exception
  : public datum_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param requested_typeid The type that was requested.
     * \param typeid_ The type that is in the datum.
     * \param type The type of the datum.
     * \param error The type that was requested.
     * \param reason The reason for the bad cast.
     */
    bad_datum_cast_exception(std::string const& requested_typeid, std::string const& typeid_, datum::type_t const& type, datum::error_t const& error, char const* reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~bad_datum_cast_exception() noexcept;

    /// The requested datum type.
    std::string const m_requested_typeid;
    /// The datum type.
    std::string const m_typeid;
    /// The datum type.
    datum::type_t const m_type;
    /// The error string from the datum.
    datum::error_t const m_error;
    /// The reason for the failed cast.
    std::string const m_reason;
};

// ----------------------------------------------------------------------------
template <typename T>
datum_t
datum::new_datum(T const& dat)
{
  return new_datum(kwiver::vital::any(dat));
}

// ----------------------------------------------------------------------------
template <typename T>
T
datum::get_datum() const
{
  try
  {
    return kwiver::vital::any_cast<T>(m_datum);
  }
  catch (kwiver::vital::bad_any_cast const& e)
  {
    std::string const req_type_name = typeid(T).name();
    std::string const type_name = m_datum.type().name();

    VITAL_THROW( bad_datum_cast_exception,
                 req_type_name, type_name, m_type, m_error, e.what());
  }
}

// ----------------------------------------------------------------------------
template <>
inline
kwiver::vital::any
datum::get_datum() const
{
  return m_datum;
}

}

#endif // SPROKIT_PIPELINE_DATUM_H
