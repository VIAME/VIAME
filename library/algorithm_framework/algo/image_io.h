// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for image_io \link kwiver::vital::algo::algorithm_def
/// algorithm
///        definition \endlink.

#ifndef VITAL_ALGO_IMAGE_IO_H_
#define VITAL_ALGO_IMAGE_IO_H_

#include <viame/algorithm_framework/vital_config.h>

#include <string>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algorithm_capabilities.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/metadata.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for reading and writing images
///
/// This class represents an abstract interface for reading and writing
/// images.
///
/// A note about the basic capabilities:
///
/// HAS_TIME - This capability is set to true if the image metadata
///     supplies a timestamp. If a timestamp is supplied, it is made
///     available in the metadata for the image. If the timestamp
///     is not supplied, then the metadata will not have the timestamp set.
class VITAL_ALGO_EXPORT image_io
  : public kwiver::vital::algorithm
{
public:
  // Common capabilities
  // -- basic capabilities --
  static const algorithm_capabilities::capability_name_t HAS_TIME;

  image_io();
  PLUGGABLE_INTERFACE( image_io );
  /// Load image from the file
  ///
  /// \throws kwiver::vital::path_not_exists Thrown when the given path does not
  /// exist.
  ///
  /// \throws kwiver::vital::path_not_a_file Thrown when the given path does
  ///    not point to a file (i.e. it points to a directory).
  ///
  /// \param filename the path to the file to load
  /// \returns an image container refering to the loaded image
  kwiver::vital::image_container_sptr load(
    std::string const& filename ) const;

  /// Save image to a file
  ///
  /// Image file format is based on file extension.
  ///
  /// \throws kwiver::vital::path_not_exists Thrown when the expected
  ///    containing directory of the given path does not exist.
  ///
  /// \throws kwiver::vital::path_not_a_directory Thrown when the expected
  ///    containing directory of the given path is not actually a
  ///    directory.
  ///
  /// \param filename the path to the file to save
  /// \param data the image container refering to the image to write
  void save(
    std::string const& filename,
    kwiver::vital::image_container_sptr data ) const;

  /// Get the image metadata
  ///
  /// \throws kwiver::vital::path_not_exists Thrown when the given path does not
  /// exist.
  ///
  /// \throws kwiver::vital::path_not_a_file Thrown when the given path does
  ///    not point to a file (i.e. it points to a directory).
  ///
  /// \param filename the path to the file to read
  /// \returns pointer to the loaded metadata
  kwiver::vital::metadata_sptr load_metadata( std::string const& filename )
  const;

  /// \brief Return capabilities of concrete implementation.
  ///
  /// This method returns the capabilities for the current image reader/writer.
  ///
  /// \return Reference to supported image capabilities.
  algorithm_capabilities const& get_implementation_capabilities() const;

protected:
  void set_capability(
    algorithm_capabilities::capability_name_t const& name,
    bool val );

private:
  /// Implementation specific load functionality.
  ///
  /// Concrete implementations of image_io class must provide an
  /// implementation for this method.
  ///
  /// \param filename the path to the file the load
  /// \returns an image container refering to the loaded image
  virtual kwiver::vital::image_container_sptr load_(
    std::string const& filename ) const = 0;

  /// Implementation specific save functionality.
  ///
  /// Concrete implementations of image_io class must provide an
  /// implementation for this method.
  ///
  /// \param filename the path to the file to save
  /// \param data the image container refering to the image to write
  virtual void save_(
    std::string const& filename,
    kwiver::vital::image_container_sptr data ) const = 0;

  /// Implementation specific metadata functionality.
  ///
  /// If a concrete implementation provides metadata, it must be provided
  /// in both load() and load_metadata(), and it must be the same metadata.
  /// To provide it in one but not the other, or to provide different metadata
  /// in each, is an error.
  ///
  /// \param filename the path to the file to read
  /// \returns pointer to the loaded metadata
  virtual kwiver::vital::metadata_sptr load_metadata_(
    std::string const& filename ) const;

  /// Return \c true if the implementation will handle its own filepath
  /// validation for \param filename.
  ///
  /// Some implementations may be capable of loading and saving from
  /// non-filesystem paths, e.g. network URLs. Returning \c true here
  /// indicates that no error should be thrown even if a path is not a
  /// valid image filepath on disk.
  virtual bool skip_path_validation_() const;

  algorithm_capabilities m_capabilities;
};

/// Shared pointer type for generic image_io definition type.
typedef std::shared_ptr< image_io > image_io_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_IMAGE_IO_H_
