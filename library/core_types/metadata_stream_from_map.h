// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Map-based implementations of the metadata_stream interfaces.

#ifndef KWIVER_VITAL_METADATA_STREAM_FROM_MAP_H_
#define KWIVER_VITAL_METADATA_STREAM_FROM_MAP_H_

#include <viame/core_types/metadata_stream.h>

#include <map>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Stream that reads from an in-memory map that it does not own.
class VITAL_TYPES_EXPORT metadata_istream_from_map : public metadata_istream
{
public:
  using map_t = std::map< frame_id_t, metadata_vector >;
  using iterator_t = map_t::const_iterator;

  /// \param map Map object to read from sequentially.
  ///
  /// \warning
  ///   The caller is responsible for managing the lifetime of \p map, which
  ///   this object only stores a pointer to.
  explicit metadata_istream_from_map( map_t const& map );

  virtual ~metadata_istream_from_map();

  map_t const& map() const;
  iterator_t iterator() const;

  frame_id_t frame_number() const override;
  metadata_vector metadata() override;
  bool next_frame() override;
  bool at_end() const override;

private:
  map_t const* m_map;
  iterator_t m_it;
};

// ----------------------------------------------------------------------------
/// Stream that writes to an in-memory map that it does not own.
class VITAL_TYPES_EXPORT metadata_ostream_from_map : public metadata_ostream
{
public:
  using map_t = std::map< frame_id_t, metadata_vector >;
  using iterator_t = map_t::iterator;

  /// \param map Map object to write to sequentially.
  ///
  /// \warning
  ///   The caller is responsible for managing the lifetime of \p map, which
  ///   this object only stores a pointer to.
  explicit metadata_ostream_from_map( map_t& map );

  virtual ~metadata_ostream_from_map();

  map_t& map() const;

  bool write_frame(
    frame_id_t frame_number, metadata_vector const& metadata ) override;
  void write_end() override;
  bool at_end() const override;

private:
  map_t* m_map;
  bool m_at_end;
};

} // namespace vital

} // namespace kwiver

#endif
