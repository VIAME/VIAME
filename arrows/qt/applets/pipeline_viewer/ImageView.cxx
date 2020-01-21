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

#include "ImageView.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QPaintEvent>
#include <QPixmap>

namespace kwiver {

namespace tools {

// ----------------------------------------------------------------------------
class ImageViewPrivate
{
public:
  void updateTransform( QGraphicsView* view, bool forceUpdate = false );

  bool updatesSuspended = false;
  ImageView::ScaleMode scaleMode = ImageView::OriginalSize;

  QGraphicsScene scene;
  QGraphicsPixmapItem imageItem;

  QImage lastImage;
};

KQ_IMPLEMENT_D_FUNC( ImageView )

// ----------------------------------------------------------------------------
void
ImageViewPrivate
::updateTransform( QGraphicsView* view, bool forceUpdate )
{
  if ( this->scaleMode == ImageView::FitToWindow )
  {
    if ( !this->lastImage.isNull() )
    {
      view->fitInView( &this->imageItem, Qt::KeepAspectRatio );
    }
  }
  else if ( forceUpdate )
  {
    view->resetTransform();
  }
}

// ----------------------------------------------------------------------------
ImageView
::ImageView( QWidget* parent )
  : QGraphicsView{ parent }, d_ptr{ new ImageViewPrivate }
{
  KQ_D();

  d->scene.addItem( &d->imageItem );

  this->setBackgroundBrush( Qt::black );
  this->setScene( &d->scene );
}

// ----------------------------------------------------------------------------
ImageView
::~ImageView()
{
}

// ----------------------------------------------------------------------------
bool
ImageView
::updatesSuspended() const
{
  KQ_D();
  return d->updatesSuspended;
}

// ----------------------------------------------------------------------------
void
ImageView
::setUpdatesSuspended( bool suspend )
{
  KQ_D();

  d->updatesSuspended = suspend;

  if ( !suspend )
  {
    displayImage( d->lastImage );
  }
}

// ----------------------------------------------------------------------------
ImageView::ScaleMode
ImageView
::scaleMode() const
{
  KQ_D();
  return d->scaleMode;
}

// ----------------------------------------------------------------------------
void
ImageView
::setScaleMode( ScaleMode mode )
{
  KQ_D();

  d->scaleMode = mode;
  d->updateTransform( this, true );
}

// ----------------------------------------------------------------------------
void
ImageView
::displayImage( QImage const& image )
{
  KQ_D();

  auto const first = ( d->lastImage.isNull() );
  d->lastImage = image;

  if ( first || !d->updatesSuspended )
  {
    d->imageItem.setPixmap( QPixmap::fromImage( image ) );
    d->updateTransform( this );
  }
}

} // namespace tools

} // namespace kwiver
