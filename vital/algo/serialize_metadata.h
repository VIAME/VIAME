// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for serialize_metadata \link kwiver::vital::algo::algorithm_def algorithm
 *        definition \endlink.
 */

#ifndef VITAL_ALGO_SERIALIZE_METADATA_H_
#define VITAL_ALGO_SERIALIZE_METADATA_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/algorithm_capabilities.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_map.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for reading and writing meatadata maps
/**
 * This class represents an abstract interface for reading and writing
 * video metadata.
 */
class VITAL_ALGO_EXPORT serialize_metadata
  : public kwiver::vital::algorithm_def< serialize_metadata >
{
public:
  virtual ~serialize_metadata() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "serialize_metadata"; }

  /// Load metadata from the file
  /**
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not
   *    exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   *
   * \param filename the path to the file to load
   * \returns a metadata_map_sptr pointing to the data
   */
  kwiver::vital::metadata_map_sptr load( std::string const& filename ) const;

  /// Save metadata to a file
  /**
   * Save data for all frames in a map
   *
   * \throws kwiver::vital::path_not_exists Thrown when the expected
   *    containing directory of the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_directory Thrown when the expected
   *    containing directory of the given path is not actually a
   *    directory.
   *
   * \param filename the path to the file to save
   * \param data pointer to the metadata to write
   */
  void save( std::string const& filename,
             kwiver::vital::metadata_map_sptr data ) const;

  /**
   * \brief Return capabilities of concrete implementation.
   *
   * This method returns the capabilities for the current image reader/writer.
   *
   * \return Reference to supported image capabilities.
   */
  algorithm_capabilities const& get_implementation_capabilities() const;

  void set_configuration( vital::config_block_sptr config );

  /// Check that the algorithm's current configuration is valid
  bool check_configuration( vital::config_block_sptr config ) const;

protected:
  serialize_metadata();

  void set_capability( algorithm_capabilities::capability_name_t const& name,
                       bool val );

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of serialize_metadata class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns an metadata_map_sptr pointing to the loaded metadata
   */
  virtual kwiver::vital::metadata_map_sptr load_( std::string const& filename ) const = 0;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of serialize_metadata class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the metadata_map_sptr pointing to the metadata
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::metadata_map_sptr data ) const = 0;

  algorithm_capabilities m_capabilities;
};

/// Shared pointer type for generic serialize_metadata definition type.
typedef std::shared_ptr< serialize_metadata > serialize_metadata_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_SERIALIZE_METADATA_H_
