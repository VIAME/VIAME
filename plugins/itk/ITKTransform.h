/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_ITK_ITK_TRANSFORM_H
#define VIAME_ITK_ITK_TRANSFORM_H

#include "viame_itk_export.h"

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>

#include "RegisterOpticalAndThermal.h"

namespace viame
{

namespace itk
{

/// Wraps ITK transforms in KWIVER vital types
class VIAME_ITK_EXPORT ITKTransform : public kwiver::vital::transform_2d
{
public:

  ITKTransform( BaseTransformType::Pointer transform );
  virtual ~ITKTransform();

  /// Create a clone of this transform object, returning as smart pointer
  /**
   * \return A new deep clone of this transformation.
   */
  kwiver::vital::transform_2d_sptr clone() const override;

  /// Return an inverse of this transform object
  /**
   * \throws non_invertible_transform when the transformation is non-invertible.
   * \return A new transform object that is the inverse of this transformation.
   */
  kwiver::vital::transform_2d_sptr inverse_() const override;

  /// Map a 2D double-type point using this transform
  /**
   * \param p Point to map against this transform
   * \return New point in the projected coordinate system.
   */
  kwiver::vital::vector_2d map( kwiver::vital::vector_2d const& p ) const override;

private:

  BaseTransformType::Pointer m_transform;
};


/// A class for using ITK to read and write arbitrary transforms
class VIAME_ITK_EXPORT ITKTransformIO
  : public kwiver::vital::algo::transform_2d_io
{
public:
  /// Constructor
  ITKTransformIO();

  /// Destructor
  virtual ~ITKTransformIO();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const;

};


} // end namespace itk

} // end namespace viame

#endif // VIAME_ITK_ITK_TRANSFORM_H
