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
 * \brief This file contains the interface for iterative query refinement feedback.
 */

#ifndef VITAL_IQR_FEEDBACK_H_
#define VITAL_IQR_FEEDBACK_H_

#include "image_container.h"
#include "timestamp.h"
#include "track_descriptor.h"
#include "uid.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <memory>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/// A representation of iterative query refinement feedback.
class VITAL_EXPORT iqr_feedback
{
public:

  iqr_feedback();
  ~iqr_feedback() VITAL_DEFAULT_DTOR

  uid query_id() const;

  std::vector< unsigned > const& positive_ids() const;
  std::vector< unsigned > const& negative_ids() const;

  void set_query_id( uid const& );

  void set_positive_ids( std::vector< unsigned > const& );
  void set_negative_ids( std::vector< unsigned > const& );

protected:

  vital::uid m_query_id;

  std::vector< unsigned > m_positive_ids;
  std::vector< unsigned > m_negative_ids;
};

/// Shared pointer for query plan
typedef std::shared_ptr< iqr_feedback > iqr_feedback_sptr;

} } // end namespace vital

#endif // VITAL_IQR_FEEDBACK_H_
