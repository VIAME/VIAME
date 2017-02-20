/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef DYNAMIC_CONFIGURATION_H
#define DYNAMIC_CONFIGURATION_H


#include <vital/algo/algorithm.h>

namespace kwiver {
namespace vital {
namespace algo {


/// Abstract algorithm for getting dynamic configuration values from
/// an external source.
/**
 * This class represents an interface to an external source of
 * configuration values. A typical application would be an external
 * U.I. control that is desired to control the performance of an
 * algorithm by varying some of its configuration values.
 */
class VITAL_ALGO_EXPORT dynamic_configuration :
    public kwiver::vital::algorithm_def< dynamic_configuration >
{
public:
  static std::string static_type_name() { return "dynamic_configuration"; }

  virtual void set_configuration( config_block_sptr config ) = 0;
  virtual bool check_configuration( config_block_sptr config ) const = 0;

  /// Return dynamic configuration values
  /**
   * This method returns dynamic configuration values. a valid config
   * block is returned even if there are not values being returned.
   */
  virtual config_block_sptr get_dynamic_configuration() = 0;


protected:
  dynamic_configuration();
};

/// Shared pointer for generic dynamic_configuration definition type.
typedef std::shared_ptr< dynamic_configuration > dynamic_configuration_sptr;

}
}
}     // end namespace

#endif // DYNAMIC_CONFIGURATION_H
