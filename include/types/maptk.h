/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef _KWIVER_MAPTK_H_
#define _KWIVER_MAPTK_H_

#include <sprokit/pipeline/process.h>

/*! \file Type names for maptk types.
 *
 * This file contains the canonical type strings used by sprokit for
 * the types defined in maptk.
 */

#define DEF_TYPE( T ) static sprokit::process::type_t const maptk_ ## T( "maptk:" # T )

// Concrete types - these might not be used in the pipeline
DEF_TYPE( f2f_homography );
DEF_TYPE( f2w_homography );

// Abstract types / containers
DEF_TYPE( image );
DEF_TYPE( feature_set );
DEF_TYPE( descriptor_set );
DEF_TYPE( match_set );
DEF_TYPE( track_set );
DEF_TYPE( landmark_map );
DEF_TYPE( camera_map );

#undef DEF_TYPE
#endif /* _KWIVER_MAPTK_H_ */
