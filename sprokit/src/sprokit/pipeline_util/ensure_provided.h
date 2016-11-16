/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * @file   ensure_provided.h
 * @brief  Interface to ensure_provided class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_ENSURE_PROVIDED_H
#define SPROKIT_PIPELINE_UTIL_ENSURE_PROVIDED_H

#include "bakery_base.h"

#include <vital/config/config_block.h>


namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class ensure_provided
  : public boost::static_visitor<kwiver::vital::config_block_value_t>
{
  public:
    ensure_provided();
    ~ensure_provided();

    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& value) const;
    kwiver::vital::config_block_value_t operator () (bakery_base::provider_request_t const& request) const;
};


} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_ENSURE_PROVIDED_H */
