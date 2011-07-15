/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_DATUM_H
#define VISTK_PIPELINE_DATUM_H

#include "types.h"

#include <boost/any.hpp>

#include <string>

namespace vistk
{

/**
 * \class datum
 *
 * \brief A wrapper for data that passes through an \ref edge in the \ref pipeline.
 *
 * \ingroup base_classes
 */
class datum
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
     */
    static datum_t empty_datum();
    /**
     * \brief Create a datum with the #DATUM_COMPLETE type.
     */
    static datum_t complete_datum();
    /**
     * \brief Create a datum with the #DATUM_ERROR type.
     *
     * \param error Information about the error that occurred.
     */
    static datum_t error_datum(error_t const& error);
    /**
     * \brief Create a datum with the #DATUM_DATA type.
     *
     * \param datum The data to pass through the edge.
     */
    template <typename T>
    static datum_t new_datum(T const& datum);

    /**
     * \brief Returns the type of the datum.
     */
    datum_type_t type() const;

    /**
     * \brief Returns the data in the edge.
     *
     * \throws bad_datum_cast Thrown when the data cannot be cast as requested.
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
 * \class datum_exception
 *
 * \brief The base class for all exceptions thrown from \ref datum.
 */
class datum_exception
  : public pipeline_exception
{
};

/**
 * \class bad_datum_cast
 *
 * \brief Thrown when the \ref datum cannot be converted to the requested type.
 */
class bad_datum_cast
  : public datum_exception
{
  public:
    bad_datum_cast(datum::datum_type_t const& type, char const* reason) throw();
    ~bad_datum_cast() throw();

    /// The datum type.
    datum::datum_type_t const m_type;
    /// The reason for the failed cast.
    std::string const m_reason;

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
