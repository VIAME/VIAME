/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
  : public vital::algorithm_impl<feature_descriptor_io,
                                 vital::algo::feature_descriptor_io>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "core";

  /// Description of the algorithm
  static constexpr char const* description =
    "Read and write features and descriptor"
    " to binary files using Cereal serialization.";

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

#endif // KWIVER_ARROWS_CORE_FEATURE_DESCRIPTOR_IO_H_
