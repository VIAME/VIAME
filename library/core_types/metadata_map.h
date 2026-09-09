// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header file for a map from frame IDs to metadata vectors

#ifndef KWIVER_VITAL_METADATA_MAP_H_
#define KWIVER_VITAL_METADATA_MAP_H_

#include <viame/core_types/metadata.h>

#include <viame/core_types/geo_point.h>
#include <viame/core_types/metadata_traits.h>
#include <viame/algorithm_framework/vital_config.h>
#include <viame/core_types/vital_types.h>

#include <map>
#include <memory>
#include <set>

namespace kwiver {

namespace vital {

/// An abstract mapping between frame IDs and metadata vectors
//  \note a vector of metadata objects is used because each frame could
//  have multiple metadata blocks.  For example, metadata may come from
//  multiple sources on a given frame or a metadata may be provided at
//  a higher sampling rate than the video sampling rate.
class metadata_map
{
public:
  /// typedef for std::map from integer frame IDs to metadata vectors
  typedef std::map< frame_id_t, metadata_vector > map_metadata_t;

  /// Destructor
  virtual ~metadata_map() = default;

  /// Return the number of frames in the map
  virtual size_t size() const = 0;

  /// Return a map from integer frame IDs to metadata vectors
  virtual map_metadata_t metadata() const = 0;

  /// Check if metadata is present in the map for given tag and frame id
  ///
  /// \param tag the metadata tag
  /// \param fid the frame id
  /// \returns true if the metadata item is present for the given tag and frame
  /// id
  virtual bool has_item( vital_metadata_tag tag, frame_id_t fid ) const = 0;

  /// Get a metadata item from the map according to its tag and the frame
  ///
  /// \param tag the metadata tag
  /// \parma fid the frame id
  /// \returns the metadata item for the requested tag and frame id
  virtual metadata_item const&
  get_item( vital_metadata_tag tag, frame_id_t fid ) const = 0;

  /// Get a vector of all metadata available at a given frame id
  virtual metadata_vector
  get_vector( frame_id_t fid ) const = 0;

  /// Templated version of has_item to match get method.
  ///
  /// \param tag the metadata tag
  /// \param fid the frame id
  /// \returns true if the metadata item is present for the given tag and frame
  /// id
  template < vital_metadata_tag tag >
  bool
  has( frame_id_t fid )
  {
    return has_item( tag, fid );
  }

  /// Get value for a metadata item from the map for given tag and frame id
  ///
  /// \param tag the metadata tag
  /// \param fid the frame id
  /// \returns the metadata value
  template < vital_metadata_tag tag >
  type_of_tag< tag >
  get( frame_id_t fid ) const
  {
    return this->get_item( tag, fid ).get< type_of_tag< tag > >();
  }

  /// Returns the frame ids that have associated metadata
  virtual std::set< frame_id_t > frames() const = 0;
};

/// typedef for a metadata shared pointer
typedef std::shared_ptr< metadata_map > metadata_map_sptr;

/// A concrete metadata_map that simply wraps a std::map.
class simple_metadata_map
  : public metadata_map
{
public:
  /// Default Constructor
  simple_metadata_map() {}

  /// Constructor from a std::map of metadata
  explicit simple_metadata_map( map_metadata_t const& metadata )
    : data_( metadata ) {}

  /// Return the number of metadata in the map
  virtual size_t
  size() const { return data_.size(); }

  /// Return a map from integer IDs to metadata shared pointers
  virtual map_metadata_t
  metadata() const { return data_; }

  /// Returns the frame ids that have associated metadata
  virtual std::set< frame_id_t >
  frames() const
  {
    std::set< frame_id_t > fids;
    for( auto& m : data_ )
    {
      fids.insert( m.first );
    }
    return fids;
  }

  /// get a metadata item from the map according to its tag and the frame
  virtual metadata_item const&
  get_item( vital_metadata_tag tag, frame_id_t fid ) const
  {
    static metadata_item unknown_item{ VITAL_META_UNKNOWN, 0 };

    auto d_it = data_.find( fid );
    if( d_it == data_.end() )
    {
      return unknown_item;
    }

    auto& mdv = d_it->second;
    for( auto md : mdv )
    {
      if( auto const& item = md->find( tag ) )
      {
        return item;
      }
    }

    return unknown_item;
  }

  /// Get a vector of all metadata available at a given frame id
  virtual metadata_vector
  get_vector( frame_id_t fid ) const
  {
    auto const d_it = data_.find( fid );
    if( d_it == data_.end() )
    {
      return {};
    }
    return d_it->second;
  }

  /// check if metadata item is in map for given tag and frame id
  virtual bool
  has_item( vital_metadata_tag tag, frame_id_t fid ) const
  {
    auto d_it = data_.find( fid );
    if( d_it == data_.end() )
    {
      return false;
    }

    auto& mdv = d_it->second;
    for( auto md : mdv )
    {
      if( md->has( tag ) )
      {
        return true;
      }
    }

    return false;
  }

protected:
  /// The map from integer IDs to metadata shared pointers
  map_metadata_t data_;
};

} // end namespace vital

} // end namespace kwiver

#endif
