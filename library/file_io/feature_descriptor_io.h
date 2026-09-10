// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Core feature_descriptor_io interface

#ifndef KWIVER_ARROWS_CORE_FEATURE_DESCRIPTOR_IO_H_
#define KWIVER_ARROWS_CORE_FEATURE_DESCRIPTOR_IO_H_

#include "viame_file_io_export.h"

#include <viame/algorithm_framework/algo/feature_descriptor_io.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

namespace kwiver {

namespace arrows {

namespace core {

/// A class for reading and writing feature and desriptor sets
class VIAME_FILE_IO_EXPORT feature_descriptor_io
  : public vital::algo::feature_descriptor_io
{
public:
  PLUGGABLE_IMPL(
    feature_descriptor_io,
    "Read and write features and descriptor"
    " to binary files using Cereal serialization.",
    PARAM_DEFAULT(
      write_float_features, bool,
      "Convert features to use single precision floats "
      "instead of doubles when writing to save space",
      false )
  )

  /// Destructor
  virtual ~feature_descriptor_io();

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

private:
  /// Implementation specific load functionality.
  ///
  /// Concrete implementations of feature_descriptor_io class must provide an
  /// implementation for this method.
  ///
  /// \param filename the path to the file the load
  /// \param feat the set of features to load from the file
  /// \param desc the set of descriptors to load from the file
  virtual void load_(
    std::string const& filename,
    vital::feature_set_sptr& feat,
    vital::descriptor_set_sptr& desc ) const;

  /// Implementation specific save functionality.
  ///
  /// Concrete implementations of feature_descriptor_io class must provide an
  /// implementation for this method.
  ///
  /// \param filename the path to the file to save
  /// \param feat the set of features to write to the file
  /// \param desc the set of descriptors to write to the file
  virtual void save_(
    std::string const& filename,
    vital::feature_set_sptr feat,
    vital::descriptor_set_sptr desc ) const;

  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif
