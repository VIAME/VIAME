// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Core feature_descriptor_io interface
 */

#ifndef KWIVER_ARROWS_CORE_FEATURE_DESCRIPTOR_IO_H_
#define KWIVER_ARROWS_CORE_FEATURE_DESCRIPTOR_IO_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/feature_descriptor_io.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for reading and writing feature and desriptor sets
class KWIVER_ALGO_CORE_EXPORT feature_descriptor_io
  : public vital::algo::feature_descriptor_io
{
public:
  PLUGIN_INFO( "core",
               "Read and write features and descriptor"
               " to binary files using Cereal serialization." )

  /// Constructor
  feature_descriptor_io();

  /// Destructor
  virtual ~feature_descriptor_io();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

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
                     vital::feature_set_sptr& feat,
                     vital::descriptor_set_sptr& desc) const;

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
                     vital::feature_set_sptr feat,
                     vital::descriptor_set_sptr desc) const;

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
