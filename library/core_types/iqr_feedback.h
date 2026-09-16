// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief This file contains the interface for iterative query refinement
/// feedback.

#ifndef VITAL_IQR_FEEDBACK_H_
#define VITAL_IQR_FEEDBACK_H_

#include "image_container.h"
#include "timestamp.h"
#include "track_descriptor.h"
#include "uid.h"

#include <viame/core_types/viame_core_types_export.h>
#include <viame/algorithm_framework/viame_compiler_config.h>

#include <memory>
#include <string>

namespace viame {

// ----------------------------------------------------------------------------
/// A representation of iterative query refinement feedback.
class VIAME_CORE_TYPES_EXPORT iqr_feedback
{
public:
  iqr_feedback();

  ~iqr_feedback() = default;

  viame::uid query_id() const;

  std::vector< unsigned > const& positive_ids() const;
  std::vector< unsigned > const& negative_ids() const;

  void set_query_id( viame::uid const& );

  void set_positive_ids( std::vector< unsigned > const& );
  void set_negative_ids( std::vector< unsigned > const& );

protected:
  viame::uid m_query_id;

  std::vector< unsigned > m_positive_ids;
  std::vector< unsigned > m_negative_ids;
};

/// Shared pointer for query plan
typedef std::shared_ptr< iqr_feedback > iqr_feedback_sptr;

} // namespace viame

#endif // VITAL_IQR_FEEDBACK_H_
