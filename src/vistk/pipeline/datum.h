/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_DATUM_H
#define VISTK_PIPELINE_DATUM_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/any.hpp>

#include <string>

/**
 * \file datum.h
 *
 * \brief Header for a piece of \link vistk::datum data\endlink in the pipeline.
 */

namespace vistk
{

/**
 * \class datum datum.h <vistk/pipeline/datum.h>
 *
 * \brief A wrapper for data that passes through an \ref edge in the \ref pipeline.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT datum
{
  public:
    /// Information about an error that occurred within a process.
    typedef std::string error_t;

    /// The type of the datum being passed through the edge.
    typedef enum
    {
      /// An invalid type.
      invalid,
      /// Data is included in the edge.
      data,
      /// No data was computed for the computation.
      empty,
      /// The current data stream is complete and a new one will follow.
      flush,
      /// The process is complete and no more data will be available on this edge.
      complete,
      /// An error occurred when computing the data.
      error
    } type_t;

    /**
     * \brief Create a datum with the #data type.
     *
     * This method is for bindings to be able to create boost::any objects
     * manually.
     *
     * \param dat The data to pass through the edge.
     *
     * \returns A new datum containing a result.
     */
    static datum_t new_datum(boost::any const& dat);
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
     * \returns The type of the datum.
     */
    type_t type() const;

    /**
     * \brief Query for the error that occurred.
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
  private:
    datum(type_t ty);
    datum(error_t const& err);
    datum(boost::any const& dat);

    type_t const m_type;
    error_t const m_error;
    boost::any const m_datum;
};

/**
 * \class datum_exception datum.h <vistk/pipeline/datum.h>
 *
 * \brief The base class for all exceptions thrown from \ref datum.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT datum_exception
  : public pipeline_exception
{
};

/**
 * \class bad_datum_cast_exception datum.h <vistk/pipeline/datum.h>
 *
 * \brief Thrown when the \ref datum cannot be converted to the requested type.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT bad_datum_cast_exception
  : public datum_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type that was requested.
     * \param reason The reason for the bad cast.
     */
    bad_datum_cast_exception(datum::type_t const& type, char const* reason) throw();
    /**
     * \brief Destructor.
     */
    ~bad_datum_cast_exception() throw();

    /// The datum type.
    datum::type_t const m_type;
    /// The reason for the failed cast.
    std::string const m_reason;
};

template <typename T>
datum_t
datum::new_datum(T const& dat)
{
  return new_datum(boost::any(dat));
}

template <typename T>
T
datum::get_datum() const
{
  try
  {
    return boost::any_cast<T>(m_datum);
  }
  catch (boost::bad_any_cast& e)
  {
    throw bad_datum_cast_exception(m_type, e.what());
  }
}

template <>
inline
boost::any
datum::get_datum() const
{
  return m_datum;
}

}

#endif // VISTK_PIPELINE_DATUM_H
