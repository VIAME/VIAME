// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for MatLab util functions
 */

#include "matlab_util.h"

#include <arrows/ocv/image_container.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdint.h>
#include <memory>

namespace kwiver {
namespace arrows {
namespace matlab {

#define IDX_COLUMN_MAJOR( c, r, p, nrows, ncols ) ( ( c ) + ( ( r ) + ( p ) * ( ncols ) ) * ( nrows ) )
#define IDX_OPENCV( c, r, p, nrows, ncols, nchannels ) ( ( ( ( c ) * ( ncols ) + ( r ) ) * ( nchannels ) ) + ( p ) )

// ------------------------------------------------------------------
MxArraySptr
convert_mx_image( const kwiver::vital::image_container_sptr image )
{
  cv::Mat ocv_image = kwiver::arrows::ocv::image_container::vital_to_ocv(
    image->get_image(),
    kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );

  const size_t planes = ocv_image.channels();
  const size_t rows = ocv_image.rows;
  const size_t cols = ocv_image.cols;

  mwSize dims[3] = { rows, cols, planes };
  MxArray* mx_image = new MxArray( mxCreateNumericArray( 3, dims, mxUINT8_CLASS, mxREAL ) );

  // Copy the pixels
  uint8_t* mx_mem = static_cast< uint8_t* > ( mxGetData( mx_image->get() ) );
  uint8_t* cv_mem = static_cast< uint8_t* > ( ocv_image.data );

  for ( size_t p = 0; p < planes; p++ )
  {
    for ( size_t c = 0; c < cols; c++ )
    {
      for ( size_t r = 0; r < rows; r++ )
      {
        mx_mem[ IDX_COLUMN_MAJOR( r, c, p, rows, cols ) ] =
          cv_mem[ IDX_OPENCV(  r, c, (planes - p - 1), rows, cols, planes ) ];
      }
    }
  }

  return MxArraySptr( mx_image );
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr convert_mx_image( const MxArraySptr mx_image )
{
  std::vector< mwSize > dims = mx_image->dimensions();

  const size_t rows = dims[0];
  const size_t cols = dims[1];
  const size_t planes = dims[2];

  cv::Mat ocv_image(rows, cols, CV_8UC3 );

  // Copy the pixels
  uint8_t* mx_mem = static_cast< uint8_t* > ( mxGetData( mx_image->get() ) );
  uint8_t* cv_mem = static_cast< uint8_t* > ( ocv_image.data );

  for ( size_t p = 0; p < planes; p++ )
  {
    for ( size_t c = 0; c < cols; c++ )
    {
      for ( size_t r = 0; r < rows; r++ )
      {
          cv_mem[ IDX_OPENCV(  r, c, (planes - p - 1), rows, cols, planes ) ] =
            mx_mem[ IDX_COLUMN_MAJOR( r, c, p, rows, cols ) ];
      }
    }
  }

  kwiver::vital::image_container_sptr retval =
    std::make_shared< kwiver::arrows::ocv::image_container >(
      ocv_image,
      kwiver::arrows::ocv::image_container::ColorMode::BGR_COLOR );
  return retval;
}

// ------------------------------------------------------------------
MxArraySptr create_mxByteArray( size_t r, size_t c )
{
  return std::make_shared< MxArray >( mxCreateNumericMatrix( r, c,  mxUINT8_CLASS, mxREAL ) );
}

// ------------------------------------------------------------------
MxArraySptr create_mxIntArray( size_t r, size_t c )
{
  MxArray* mxa = new MxArray( mxCreateNumericMatrix( r, c, mxINT32_CLASS, mxREAL ) );
  return MxArraySptr( mxa );
}

// ------------------------------------------------------------------
MxArraySptr create_mxDoubleArray( size_t r, size_t c )
{
  MxArray* mxa = new MxArray( mxCreateNumericMatrix( r, c, mxDOUBLE_CLASS, mxREAL ) );
  return MxArraySptr( mxa );
}

} } } // end namespace
