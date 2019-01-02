/*ckwg +30
 * Copyright 2019 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#ifndef KWIVER_TOOLS_PIPELINE_VIEWER_IMAGEVIEW_H_
#define KWIVER_TOOLS_PIPELINE_VIEWER_IMAGEVIEW_H_

#include <arrows/qt/kq_global.h>

#include <QGraphicsView>

namespace kwiver {

namespace tools {

class ImageViewPrivate;

// ----------------------------------------------------------------------------
class ImageView : public QGraphicsView
{
  Q_OBJECT

public:
  enum ScaleMode
  {
    OriginalSize,
    FitToWindow,
  };

  explicit ImageView( QWidget* parent = nullptr );

  ~ImageView();

  bool updatesSuspended() const;
  void setUpdatesSuspended( bool suspend );

  ScaleMode scaleMode() const;
  void setScaleMode( ScaleMode mode );

public slots:
  void displayImage( QImage const& image );

protected:
  KQ_DECLARE_PRIVATE_RPTR( ImageView );

private:
  KQ_DECLARE_PRIVATE( ImageView );
};

} // namespace tools

} // namespace kwiver

#endif
