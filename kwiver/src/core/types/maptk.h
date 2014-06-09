/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_TYPES_MAPTK_H_
#define _KWIVER_TYPES_MAPTK_H_

#include <sprokit/pipeline/process.h>

/*! \file Type names for maptk types.
 *
 * This file contains the canonical type strings used by sprokit for
 * the types defined in maptk.
 */

static sprokit::process::type_t const maptk_src_to_ref_homography( "maptk_s2r_homography" );
static sprokit::process::type_t const maptk_image_container( "maptk_image_container_sptr" );

#endif /* _KWIVER_TYPES_MAPTK_H_ */
