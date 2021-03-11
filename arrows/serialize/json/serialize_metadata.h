// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for serialize_metadata \link
 * kwiver::vital::algo::algorithm_def algorithm
 *        definition \endlink.
 */

#ifndef VITAL_ARROWS_SERIALIZATION_JSON_SERIALIZE_METADATA_H_
#define VITAL_ARROWS_SERIALIZATION_JSON_SERIALIZE_METADATA_H_

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/algo/serialize_metadata.h>
#include <vital/vital_config.h>

#include <string>

namespace kwiver {

namespace arrows {

namespace serialize {

namespace json {

/// An implementation class for reading and writing metadata maps

/**
 * This class is a concrete implementation for reading and writing
 * video metadata.
 */
// KWIVER_ALGO_SERIALIZE_EXPORT
class KWIVER_SERIALIZE_JSON_EXPORT serialize_metadata
  : public vital::algo::serialize_metadata
{
public:
  /// Constructor
  serialize_metadata();

  /// Destructor
  ~serialize_metadata();

  /// Implementation specific load functionality.

  /**
   * \param filename the path to the file the load
   * \returns an image container refering to the loaded image
   */
  virtual kwiver::vital::metadata_map_sptr load_( std::string const& filename )
  const;

  /// Implementation specific save functionality.

  /**
   * \param filename the path to the file to save
   * \param data the image container refering to the image to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::metadata_map_sptr data ) const;

private:
  class priv;

  std::unique_ptr< priv > d_;
};

} // namespace json

} // namespace serialize

} // namespace arrows

} // end namespace

#endif
