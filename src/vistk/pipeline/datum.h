/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
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
    VISTK_PIPELINE_NO_EXPORT datum(type_t ty);
    VISTK_PIPELINE_NO_EXPORT datum(error_t const& err);
    VISTK_PIPELINE_NO_EXPORT datum(boost::any const& dat);

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
  public:
    /**
     * \brief Constructor.
     */
    datum_exception() throw();
    /**
     * \brief Destructor.
     */
    virtual ~datum_exception() throw();
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
     * \param requested_typeid The type that was requested.
     * \param typeid_ The type that is in the datum.
     * \param type The type of the datum.
     * \param error The type that was requested.
     * \param reason The reason for the bad cast.
     */
    bad_datum_cast_exception(std::string const& requested_typeid, std::string const& typeid_, datum::type_t const& type, datum::error_t const& error, char const* reason) throw();
    /**
     * \brief Destructor.
     */
    ~bad_datum_cast_exception() throw();

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
  catch (boost::bad_any_cast const& e)
  {
    std::string const req_type_name = typeid(T).name();
    std::string const type_name = m_datum.type().name();

    throw bad_datum_cast_exception(req_type_name, type_name, m_type, m_error, e.what());
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
