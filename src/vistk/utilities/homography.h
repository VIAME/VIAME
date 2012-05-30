/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_HOMOGRAPHY_H
#define VISTK_UTILITIES_HOMOGRAPHY_H

#include "utilities-config.h"

#include <boost/operators.hpp>

#include <vgl/algo/vgl_h_matrix_2d.h>

/**
 * \file homography.h
 *
 * \brief Declaration of the homography class.
 */

namespace vistk
{

/**
 * \class homography_base homography.h <vistk/utilities/homography.h>
 *
 * \brief Base class for the homography type.
 */
class VISTK_UTILITIES_EXPORT homography_base
  : boost::equality_comparable<vistk::homography_base>
{
  public:
    /// The type for the actual transformation matrix.
    typedef vgl_h_matrix_2d<double> transform_t;

    /**
     * \brief Copy constructor.
     *
     * \param h The homography to copy.
     */
    homography_base(homography_base const& h);
    /**
     * \brief Destructor.
     */
    virtual ~homography_base();

    /**
     * \brief The transformation matrix of the homography.
     *
     * \returns The transformation matrix.
     */
    transform_t const& transform() const;
    /**
     * \brief Query if the homography is valid.
     *
     * \returns True if the homography is valid, false otherwise.
     */
    virtual bool is_valid() const;
    /**
     * \brief Query if the homography is a new reference.
     *
     * \returns True if the homography is a new reference, false otherwise.
     */
    virtual bool is_new_reference() const;

    /**
     * \brief Set the transformation matrix.
     *
     * \param trans The new transformation matrix.
     */
    void set_transform(transform_t const& trans);
    /**
     * \brief Set the transformation matrix to the identity.
     */
    void set_identity();
    /**
     * \brief Set the validity of the homography.
     *
     * \param valid Whether the homography is valid or not.
     */
    void set_valid(bool valid);
    /**
     * \brief Set whether the homography is a new reference or not.
     *
     * \param new_reference Whether the homography is a new reference or not.
     */
    void set_new_reference(bool new_reference);

    /**
     * \brief Equality operator for homographies.
     *
     * \param h The homography to compare to.
     *
     * \returns True if \p h and \c this are equivalent, false otherwise.
     */
    bool operator == (homography_base const& h) const;
  protected:
    /**
     * \brief Constructor.
     */
    homography_base();
  private:
    transform_t m_transform;
    bool m_valid;
    bool m_new_reference;
};

/**
 * \class homography homography.h <vistk/utilities/homography.h>
 *
 * \brief A homography between two plane types.
 */
template <typename Source, typename Dest>
class VISTK_UTILITIES_EXPORT homography
  : public homography_base
{
  public:
    /// A typedef for the current homography type.
    typedef homography<Source, Dest> self_t;
    /// A typedef for the inverse of the current homography type.
    typedef homography<Dest, Source> inverse_t;
    /// A typedef for the source type.
    typedef Source source_t;
    /// A typedef for the destination type.
    typedef Dest dest_t;

    /**
     * \brief Constructor.
     */
    homography();
    /**
     * \brief Copy constructor.
     *
     * \param h The homography to copy.
     */
    homography(self_t const& h);
    /**
     * \brief Destructor.
     */
    virtual ~homography();

    /**
     * \brief Query for the source plane data.
     *
     * \returns The source plane data.
     */
    source_t source() const;
    /**
     * \brief Query for the destination plane data.
     *
     * \returns The destination plane data.
     */
    dest_t destination() const;

    /**
     * \brief Set the source plane data.
     *
     * \param src The source plane data.
     */
    void set_source(source_t const& src);
    /**
     * \brief Set the destination plane data.
     *
     * \param dest The destination plane data.
     */
    void set_destination(dest_t const& dest);

    /**
     * \brief The inverse of the current homography.
     *
     * \returns The inverse of the current homography.
     */
    inverse_t inverse() const;

    /**
     * \brief Equality operator for homographies.
     *
     * \param h The homography to compare to.
     *
     * \returns True if \p h and \c this are equivalent, false otherwise.
     */
    bool operator == (self_t const& h) const;
  private:
    source_t m_source;
    dest_t m_dest;
};

/**
 * \brief Multiplication for matrices.
 *
 * \param l The left side of the multiply.
 * \param r The right side of the multiply.
 *
 * \returns The resulting homography.
 */
template <typename Source, typename Shared, typename Dest>
homography<Source, Dest> operator * (homography<Shared, Dest> const& l, homography<Source, Shared> const& r);

}

#endif // VISTK_UTILITIES_HOMOGRAPHY_H
