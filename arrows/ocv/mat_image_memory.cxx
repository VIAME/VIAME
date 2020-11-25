// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV mat_image_memory implementation
 */

#include "mat_image_memory.h"

namespace kwiver {
namespace arrows {

namespace ocv
{

/// Constructor - allocates n bytes
/**
 * Base on how the cv::Mat constructor that taked ranges is implemented
 * (sub-matrix construction), data is a pointer with value greater than or equal
 * to datastart. Thus, the start of the global data is datastart and the start
 * of the given matrix's window is data.
 */
mat_image_memory
::mat_image_memory(const cv::Mat& m)
: mat_data_( const_cast<unsigned char*>(m.datastart) ),
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  mat_refcount_(m.refcount)
#else
  u_( m.u )
#endif
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  if ( this->mat_refcount_ )
  {
    CV_XADD(this->mat_refcount_, 1);
  }
#else
  if ( u_ )
  {
    CV_XADD(&u_->refcount, 1);
  }
#endif
  this->size_ = static_cast<size_t>( m.rows * m.step );
}

/// Destructor
mat_image_memory
::~mat_image_memory()
{
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  if( this->mat_refcount_ && CV_XADD(this->mat_refcount_, -1) == 1 )
#else
  if( u_ && CV_XADD( &u_->refcount, -1 ) == 1 )
#endif
  {
    cv::fastFree(this->mat_data_);
  }
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
