// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Declaration of the metadata stream classes.

#ifndef KWIVER_VITAL_METADATA_STREAM_H_
#define KWIVER_VITAL_METADATA_STREAM_H_

#include <viame/core_types/metadata.h>
#include <viame/core_types/vital_types_export.h>

#include <map>
#include <string>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Base class for reading or writing metadata.
class VITAL_TYPES_EXPORT metadata_stream
{
public:
  metadata_stream();
  virtual ~metadata_stream();

  /// Return the URI of the metadata stream. May be empty.
  virtual std::string uri() const;

  /// Return the configuration used when creating this stream. May be null.
  virtual config_block_sptr config() const;
};

// ----------------------------------------------------------------------------
/// Interface for reading sequential frames of metadata from somewhere.
class VITAL_TYPES_EXPORT metadata_istream : public metadata_stream
{
public:
  metadata_istream();
  virtual ~metadata_istream();

  /// Return the current frame number.
  ///
  /// \throw std::invalid_argument If \c at_end() returns \c true.
  virtual frame_id_t frame_number() const = 0;

  /// Return the metadata associated with the current frame.
  ///
  /// \throw std::invalid_argument If \c at_end() returns \c true.
  virtual metadata_vector metadata() = 0;

  /// Proceed to the next metadata frame, returning \c true on success.
  ///
  /// If this function returns \c false and \c at_end() also returns \c false,
  /// more frames are possible but not currently available, due to e.g.
  /// network lag or buffering.
  virtual bool next_frame() = 0;

  /// Return \c true if no more frames may be read from the stream.
  ///
  /// A return value of \c true may be due to a true \c EOF, or to some
  /// implementation-specific error (e.g. file corruption).
  virtual bool at_end() const = 0;
};

// ----------------------------------------------------------------------------
/// Interface for writing sequential frames of metadata to somewhere.
class VITAL_TYPES_EXPORT metadata_ostream : public metadata_stream
{
public:
  metadata_ostream();
  virtual ~metadata_ostream();

  /// Write \p metadata to the stream, returning \c true if further metadata
  /// can be written.
  ///
  /// If this function returns \c false and \c at_end() also returns \c false,
  /// it may be possible to write more metadata at some point in the future,
  /// but that is not currently possible due to e.g. a full output buffer.
  ///
  /// Any issues with \p metadata itself should be dealt with only via logging,
  /// i.e. an invalid or unsupported \p metadata object should be ignored, no
  /// exceptions should be thrown, and this function should return \c true as
  /// long as future valid metadata can still be written.
  ///
  /// \throw std::invalid_argument If \c at_end() returns \c true.
  virtual bool write_frame(
    frame_id_t frame_number, metadata_vector const& metadata ) = 0;

  /// Signal that no more metadata will be written to the stream.
  virtual void write_end() = 0;

  /// Return \c true if no more metadata can be written to the stream.
  ///
  /// A return value of \c true may be due to \c write_end() being called, or
  /// to some implementation-specific error.
  virtual bool at_end() const = 0;
};

} // namespace vital

} // namespace kwiver

#endif
