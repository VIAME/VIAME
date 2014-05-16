/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef _KWIVER_KWIVER_H_
#define _KWIVER_KWIVER_H_

#include <sprokit/pipeline/process.h>

/*! \file KWIVER specific types.
 *
 * This file contains the canonical type names for KWIVER types used
 * in the sprokit pipeline.
 */
#define DEF_TYPE(T) static sprokit::process::type_t const kwiver_ ## T( "kwiver:" # T)

DEF_TYPE( integer );
DEF_TYPE( int_32 );
DEF_TYPE( int_64 );
DEF_TYPE( float );
DEF_TYPE( double );

#undef DEF_TYPE

#endif /* _KWIVER_KWIVER_H_ */
