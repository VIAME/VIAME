/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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
