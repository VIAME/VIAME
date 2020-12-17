// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for transform_2d_io
 *        \link kwiver::vital::algo::algorithm_def algorithm definition
 *        \endlink.
 */

#ifndef VITAL_ALGO_TRANSFORM_2D_IO_H_
#define VITAL_ALGO_TRANSFORM_2D_IO_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/transform_2d.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for reading and writing transforms
/**
 * This class represents an abstract interface for reading and writing
 * transforms.
 */
class VITAL_ALGO_EXPORT transform_2d_io
  : public kwiver::vital::algorithm_def<transform_2d_io>
{
public:
  virtual ~transform_2d_io() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "transform_2d_io"; }

  /// Load transform from the file
  /**
   * \throws kwiver::vital::path_not_exists
   *   Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file
   *   Thrown when the given path does not point to a file (i.e. it points to a
   *   directory).
   *
   * \param filename the path to the file to load
   * \returns a transform instance referring to the loaded transform
   */
  kwiver::vital::transform_2d_sptr load( std::string const& filename ) const;

  /// Save transform to a file
  /**
   * Transform file format is based on the algorithm instance.
   *
   * \throws kwiver::vital::path_not_exists
   *   Thrown when the expected containing directory of the given path does not
   *   exist.
   *
   * \throws kwiver::vital::path_not_a_directory
   *   Thrown when the expected containing directory of the given path is not
   *   actually a directory.
   *
   * \throws kwiver::vital::invalid_data
   *   Thrown when the algorithm does not recognize the concrete type of the
   *   transformation instance.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  void save( std::string const& filename,
             kwiver::vital::transform_2d_sptr data ) const;

protected:
  transform_2d_io();

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_2d_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const = 0;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of transform_2d_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const = 0;
};

/// Shared pointer type for generic transform_2d_io definition type.
using transform_2d_io_sptr = std::shared_ptr< transform_2d_io >;

} // namespace algo
} // namespace vital
} // namespace kwiver

#endif
