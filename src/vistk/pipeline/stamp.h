/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_STAMP_H
#define VISTK_PIPELINE_STAMP_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/operators.hpp>

/**
 * \file stamp.h
 *
 * \brief Header for \link vistk::stamp stamps\endlink.
 */

namespace vistk
{

/**
 * \class stamp stamp.h <vistk/pipeline/stamp.h>
 *
 * \brief A class to timestamp data in a \ref pipeline.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT stamp
  : boost::equality_comparable<vistk::stamp
  , boost::less_than_comparable1<vistk::stamp
  , boost::noncopyable
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

    stamp(increment_t increment, index_t index);

    increment_t const m_increment;
    index_t const m_index;
};

}

#endif // VISTK_PIPELINE_STAMP_H
