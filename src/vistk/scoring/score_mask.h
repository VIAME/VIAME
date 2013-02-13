/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_SCORING_SCORE_MASK_H
#define VISTK_SCORING_SCORE_MASK_H

#include "scoring-config.h"

#include "scoring_result.h"

#include <vil/vil_image_view.h>

#include <boost/cstdint.hpp>

/**
 * \file score_mask.h
 *
 * \brief A function for scoring a mask.
 */

namespace vistk
{

/// A typedef for a mask image.
typedef vil_image_view<uint8_t> mask_t;

/**
 * \brief Scores a computed mask against a truth mask.
 *
 * \note The input images are expected to be the same size.
 *
 * \todo Add error handling to the function (invalid sizes, etc.).
 *
 * \param truth The truth mask.
 * \param computed The computed mask.
 *
 * \returns The results of the scoring.
 */
VISTK_SCORING_EXPORT scoring_result_t score_mask(mask_t const& truth_mask, mask_t const& computed_mask);

}

#endif // VISTK_SCORING_SCORE_MASK_H
