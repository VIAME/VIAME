// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_STAMP_H
#define SPROKIT_PIPELINE_STAMP_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/noncopyable.h>

#include "types.h"

#ifdef WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#pragma warning (disable : 4267)
#endif
#include <boost/operators.hpp>
#ifdef WIN32
#pragma warning (pop)
#endif

/**
 * \file stamp.h
 *
 * \brief Header for \link sprokit::stamp stamps\endlink.
 */

namespace sprokit
{

/**
 * \class stamp stamp.h <sprokit/pipeline/stamp.h>
 *
 * \brief A class to timestamp data in a \ref pipeline.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT stamp
  : private boost::equality_comparable<sprokit::stamp
  , boost::less_than_comparable1<sprokit::stamp
  , kwiver::vital::noncopyable
    > >
{
  public:
    /// The type for an increment size.
    typedef uint64_t increment_t;

    /**
     * \brief Create a new stamp.
     *
     * \returns A new stamp with a specific step increment.
     */
    static stamp_t new_stamp(increment_t increment);
    /**
     * \brief Create a new stamp that is has an incremented index.
     *
     * \param st The stamp to increment.
     *
     * \returns A stamp that is greater than \p st.
     */
    static stamp_t incremented_stamp(stamp_t const& st);

    /**
     * \brief Compare two stamps for equality.
     *
     * \param st The stamp to compare to.
     *
     * \returns True if \p st and \c *this have the same value, false otherwise.
     */
    bool operator == (stamp const& st) const;

    /**
     * \brief Compare two stamps for an order.
     *
     * \param st The stamp to compare to.
     *
     * \returns True if \p st has a higher value than \c *this, false otherwise.
     */
    bool operator <  (stamp const& st) const;

  private:
    typedef uint64_t index_t;

    SPROKIT_PIPELINE_NO_EXPORT stamp(increment_t increment, index_t index);

    increment_t const m_increment;
    index_t const m_index;
};

}

#endif // SPROKIT_PIPELINE_STAMP_H
