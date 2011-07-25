/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
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
      /// No data was computed for the computation.
      DATUM_EMPTY,
      /// The process is complete and no more data will be available on this edge.
      DATUM_COMPLETE,
      /// An error occurred when computing the data.
      DATUM_ERROR,
      /// Data is included in the edge.
      DATUM_DATA
    } datum_type_t;

    /**
     * \brief Create a datum with the #DATUM_EMPTY type.
     *
     * \returns A new datum which indicates that a result could not be computed.
     */
    static datum_t empty_datum();
    /**
     * \brief Create a datum with the #DATUM_COMPLETE type.
     *
     * \returns A new datum which indicates that the calculation of results is complete.
     */
    static datum_t complete_datum();
    /**
     * \brief Create a datum with the #DATUM_ERROR type.
     *
     * \param error Information about the error that occurred.
     *
     * \returns A new datum that indicates that an error occurred.
     */
    static datum_t error_datum(error_t const& error);
    /**
     * \brief Create a datum with the #DATUM_DATA type.
     *
     * \param datum The data to pass through the edge.
     *
     * \returns A new datum containing a result.
     */
    template <typename T>
    static datum_t new_datum(T const& datum);

    /**
     * \brief Query a datum for the type.
     *
     * \returns The type of the datum.
     */
    datum_type_t type() const;

    /**
     * \brief Queries for the error that occurred.
     *
     * \returns The error that occurred.
     */
    error_t get_error() const;

    /**
     * \brief Extract a result from a datum.
     *
     * \throws bad_datum_cast Thrown when the data cannot be cast as requested.
     *
     * \returns The result contained within the datum.
     */
    template <typename T>
    T get_datum() const;
  private:
    datum(bool is_complete);
    datum(error_t const& error);
    datum(boost::any const& datum);

    datum_type_t const m_type;
    error_t const m_error;
    boost::any const m_datum;
};

/**
 * \class datum_exception datum.h <vistk/pipeline/datum.h>
 *
 * \brief The base class for all exceptions thrown from \ref datum.
 */
class VISTK_PIPELINE_EXPORT datum_exception
  : public pipeline_exception
{
};

/**
 * \class bad_datum_cast datum.h <vistk/pipeline/datum.h>
 *
 * \brief Thrown when the \ref datum cannot be converted to the requested type.
 */
class VISTK_PIPELINE_EXPORT bad_datum_cast
  : public datum_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type that was requested.
     * \param reason The reason for the bad cast.
     */
    bad_datum_cast(datum::datum_type_t const& type, char const* reason) throw();
    /**
     * \brief Destructor.
     */
    ~bad_datum_cast() throw();

    /// The datum type.
    datum::datum_type_t const m_type;
    /// The reason for the failed cast.
    std::string const m_reason;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

template <typename T>
datum_t
datum::new_datum(T const& dat)
{
  return datum_t(new datum(boost::any(dat)));
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
    throw bad_datum_cast(m_type, e.what());
  }
}

} // end namespace vistk

#endif // VISTK_PIPELINE_DATUM_H
