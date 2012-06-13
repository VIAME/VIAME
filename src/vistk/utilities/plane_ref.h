/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_PLANE_REF_H
#define VISTK_UTILITIES_PLANE_REF_H

#include "utilities-config.h"

#include <boost/cstdint.hpp>
#include <boost/operators.hpp>
#include <boost/shared_ptr.hpp>

/**
 * \file plane_ref.h
 *
 * \brief Declaration of a reference plane.
 */

namespace vistk
{

/**
 * \class plane_ref plane_ref.h <vistk/utilities/plane_ref.h>
 *
 * \brief An arbitrary reference plane.
 */
class VISTK_UTILITIES_EXPORT plane_ref
  : public boost::equality_comparable<vistk::plane_ref>
{
  public:
    /// The type of the reference.
    typedef uint32_t reference_t;

    /**
     * \brief Constructor.
     */
    plane_ref();
    /**
     * \brief A plane for a specific reference.
     *
     * \param ref The plane to reference.
     */
    plane_ref(reference_t ref);
    /**
     * \brief Destructor.
     */
    ~plane_ref();

    /**
     * \brief Query whether the plane is valid or not.
     *
     * \returns True if the plane is valid, false otherwise.
     */
    bool is_valid() const;

    /**
     * \brief Query the reference of the plane.
     *
     * \returns The reference for the plane.
     */
    reference_t reference() const;

    /**
     * \brief Equality operator for reference planes.
     *
     * \param ref The plane to compare to.
     *
     * \returns True if \p ref and \c this are valid and reference the same plane.
     */
    bool operator == (plane_ref const& ref) const;

    /// An invalid reference frame.
    static reference_t const ref_invalid;
    /// A reference for the world plane.
    static reference_t const ref_tracking_world;
  private:
    reference_t const m_ref;
};

/// The type of a reference plane handle.
typedef boost::shared_ptr<plane_ref> plane_ref_t;

}

#endif // VISTK_UTILITIES_PLANE_REF_H
