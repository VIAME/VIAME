/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shared classification averaging utilities
 *
 * Provides compute_average_classification() used by stereo track pairing,
 * CSV track writing, and any other context where detected_object_type
 * scores need to be averaged across multiple detections.
 */

#ifndef VIAME_CORE_UTILITIES_TARGET_CLFR_H
#define VIAME_CORE_UTILITIES_TARGET_CLFR_H

#include "viame_core_export.h"

#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>

#include <string>
#include <vector>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

/**
 * Compute averaged classification scores from a list of detections.
 *
 * Supports three averaging modes:
 *   - simple_average:  each detection has equal weight (1.0).
 *   - weighted_average: weight = detection confidence.
 *   - weighted_scaled_by_conf: weighted average, then result scaled by
 *     (0.1 + 0.9 * average_confidence) so that low-confidence tracks
 *     produce lower overall scores.
 *
 * The optional ignore_class parameter names a class that receives special
 * treatment: detections whose *only* class label is ignore_class are
 * accumulated separately.  When both ignored and non-ignored detections
 * are present, the ignored detections are excluded from the output.
 * When all detections are ignored, the ignored class is output normally.
 *
 * \param detections  Flat list of detections to average.
 * \param weighted    If true, weight by detection confidence.
 * \param scale_by_conf  If true, apply the confidence scaling factor.
 * \param ignore_class   Class name to handle separately (empty to disable).
 * \return Averaged detected_object_type, or nullptr if no valid input.
 */
VIAME_CORE_EXPORT kv::detected_object_type_sptr
compute_average_classification(
  const std::vector< kv::detected_object_sptr >& detections,
  bool weighted = false,
  bool scale_by_conf = false,
  const std::string& ignore_class = "" );

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_UTILITIES_TARGET_CLFR_H
