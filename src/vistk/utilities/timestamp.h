/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_TIMESTAMP_H
#define VISTK_UTILITIES_TIMESTAMP_H

#include "utilities-config.h"

#include <boost/cstdint.hpp>
#include <boost/operators.hpp>
#include <boost/optional.hpp>

/**
 * \file timestamp.h
 *
 * \brief A timestamp for a frame.
 */

namespace vistk
{

/**
 * \class timestamp timestamp.h <vistk/utilities/timestamp.h>
 *
 * \brief A class representing a timestamp for an image.
 */
class VISTK_UTILITIES_EXPORT timestamp
  : boost::equality_comparable<vistk::timestamp
  , boost::less_than_comparable<vistk::timestamp
    > >
{
  public:
    /// The type which represents the time on a frame.
    typedef double time_t;
    /// The type which represents the frame number of a frame.
    typedef uint32_t frame_t;

    /**
     * \brief Constructor.
     *
     * \note Both the time and frame number are invalid.
     */
    timestamp();
    /**
     * \brief Constructor.
     *
     * \note The frame number is invalid.
     *
     * \param t The time of the frame.
     */
    explicit timestamp(time_t t);
    /**
     * \brief Constructor.
     *
     * \note The time is invalid.
     *
     * \param f The frame number.
     */
    explicit timestamp(frame_t f);
    /**
     * \brief Constructor.
     *
     * \param t The time of the frame.
     * \param f The frame number.
     */
    timestamp(time_t t, frame_t f);
    /**
     * \brief Destructor.
     */
    ~timestamp();

    /**
     * \brief Queries if the timestamp has a valid time.
     *
     * \returns True if the timestamp has a valid time, false otherwise.
     */
    bool has_time() const;
    /**
     * \brief Queries if the timestamp has a valid frame number.
     *
     * \returns True if the timestamp has a valid frame number, false otherwise.
     */
    bool has_frame() const;

    /**
     * \brief Queries for the time of the timestamp.
     *
     * \returns The time of the timestamp.
     */
    time_t time() const;
    /**
     * \brief Queries for the frame number of the timestamp.
     *
     * \returns The frame number of the timestamp.
     */
    frame_t frame() const;

    /**
     * \brief Sets the time for the timestamp.
     *
     * \param t The time for the timestamp.
     */
    void set_time(time_t t);
    /**
     * \brief Sets the frame number for the timestamp.
     *
     * \param f The frame number for the timestamp.
     */
    void set_frame(frame_t f);

    /**
     * \brief Queries whether the timestamp has any valid data in it.
     *
     * \returns True if the timestamp has valid data, false otherwise.
     */
    bool is_valid() const;

    /**
     * \brief Equality operator for timestamps.
     *
     * \param ts The timestamp to compare to.
     *
     * \returns True if \p ts and \c this are equivalent, false otherwise.
     */
    bool operator == (timestamp const& ts) const;
    /**
     * \brief Comparison operator for timestamps.
     *
     * \note The time takes precedence here. If only one of \p ts and \c this
     *       have a valid time, false is returned. When falling back to frame
     *       number comparison, \c true will only be returned if they are both
     *       valid.
     *
     * \param ts The timestamp to compare to.
     *
     * \returns True if \p ts is after \c this, false otherwise.
     */
    bool operator <  (timestamp const& ts) const;
  private:
    boost::optional<time_t> m_time;
    boost::optional<frame_t> m_frame;
};

}

#endif // VISTK_UTILITIES_TIMESTAMP_H
