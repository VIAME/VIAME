/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_STAMP_H
#define VISTK_PIPELINE_STAMP_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/cstdint.hpp>
#include <boost/operators.hpp>
#include <boost/utility.hpp>

namespace vistk
{

/**
 * \class stamp
 *
 * \brief A class to timestamp data in the pipeline.
 *
 * \ingroup base_classes
 *
 * Stamps have a color and an index. Stamps that are not the same color are
 * considered distinct and cannot be compared. This can be used to implement a
 * system where some data is being processed at a different frequency than other
 * data to keep them from being conflated. Stamps can be recolored given a stamp
 * of the desired color, so the separate data streams can eventually be
 * reconciled.
 */
class VISTK_PIPELINE_EXPORT stamp
  : boost::noncopyable
{
  public:
    /**
     * \brief Creates a new stamp.
     *
     * All stamps created with this call have a unique color.
     *
     * \returns A new stamp with a unique coloring.
     */
    static stamp_t new_stamp();
    /**
     * \brief Copies a stamp.
     *
     * Since stamps are not implicitly copyable, this is provided to copy them.
     *
     * \returns A stamp that is equivalent to \p st.
     */
    static stamp_t copied_stamp(stamp_t const& st);
    /**
     * \brief Creates a new stamp that is has an incremented index.
     *
     * \param st The stamp to increment.
     *
     * \returns A stamp that is greater than \p st.
     */
    static stamp_t incremented_stamp(stamp_t const& st);
    /**
     * \brief Creates a recolored stamp.
     *
     * \param st The original stamp.
     * \param st2 The stamp to obtain the new color from.
     *
     * \returns A new stamp with the color of \p st2 and value of \p st.
     */
    static stamp_t recolored_stamp(stamp_t const& st, stamp_t const& st2);

    /**
     * \brief Tests if a given stamp has the same color another stamp.
     *
     * \param st The stamp to compare to.
     *
     * \returns True if \p st is the same color as \c *this, false otherwise.
     */
    bool is_same_color(stamp_t const& st) const;

    /**
     * \brief Compares two stamps for equality.
     *
     * \param st The stamp to compare to.
     *
     * \returns True if \p st and \c *this have the same color and value, false otherwise.
     */
    bool operator == (stamp const& st) const;
    /**
     * \brief Compares two stamps for an order.
     *
     * \note Stamps of different colors will \em always return false with this
     * function.
     *
     * \param st The stamp to compare to.
     *
     * \returns True if \p st and \c *this are the same color and \p st has a higher value, false otherwise.
     */
    bool operator <  (stamp const& st) const;
  private:
    typedef uint64_t color_t;
    typedef uint64_t index_t;

    stamp(color_t color, index_t index);

    color_t const m_color;
    index_t const m_index;

    static color_t m_new_color;
};

} // end namespace vistk

template struct boost::partially_ordered<vistk::stamp>;
template struct boost::equality_comparable<vistk::stamp>;

#endif // VISTK_PIPELINE_STAMP_H
