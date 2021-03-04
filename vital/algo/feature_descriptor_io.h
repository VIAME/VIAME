// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for feature_descriptor_io \link kwiver::vital::algo::algorithm_def algorithm
 *        definition \endlink.
 */

#ifndef VITAL_ALGO_FEATURE_DESCRIPTOR_IO_H_
#define VITAL_ALGO_FEATURE_DESCRIPTOR_IO_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for reading and writing feature and desriptor sets
/**
 * This class represents an abstract interface for reading and writing
 * feature and descriptor sets
 */
class VITAL_ALGO_EXPORT feature_descriptor_io
  : public kwiver::vital::algorithm_def<feature_descriptor_io>
{
public:
  virtual ~feature_descriptor_io() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "feature_descriptor_io"; }

  /// Load features and descriptors from a file
  /**
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   *
   * \param filename the path to the file the load
   * \param feat the set of features to load from the file
   * \param desc the set of descriptors to load from the file
   */
  void load(std::string const& filename,
            feature_set_sptr& feat,
            descriptor_set_sptr& desc) const;

  /// Save features and descriptors to a file
  /**
   * Saves features and/or descriptors to a file.  Either \p feat or \p desc
   * may be Null, but not both.  If both \p feat and \p desc are provided then
   * the must be of the same size.
   *
   * \throws kwiver::vital::path_not_exists Thrown when the expected
   *    containing directory of the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_directory Thrown when the expected
   *    containing directory of the given path is not actually a
   *    directory.
   *
   * \param filename the path to the file to save
   * \param feat the set of features to write to the file
   * \param desc the set of descriptors to write to the file
   */
  void save(std::string const& filename,
            feature_set_sptr feat,
            descriptor_set_sptr desc) const;

protected:
  feature_descriptor_io();

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of feature_descriptor_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \param feat the set of features to load from the file
   * \param desc the set of descriptors to load from the file
   */
  virtual void load_(std::string const& filename,
                     feature_set_sptr& feat,
                     descriptor_set_sptr& desc) const = 0;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of feature_descriptor_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param feat the set of features to write to the file
   * \param desc the set of descriptors to write to the file
   */
  virtual void save_(std::string const& filename,
                     feature_set_sptr feat,
                     descriptor_set_sptr desc) const = 0;
};

/// Shared pointer type for generic feature_descriptor_io definition type.
typedef std::shared_ptr<feature_descriptor_io> feature_descriptor_io_sptr;

} } } // end namespace

#endif // VITAL_ALGO_FEATURE_DESCRIPTOR_IO_H_
