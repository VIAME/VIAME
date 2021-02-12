// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
  if( this->scaleMode == ImageView::FitToWindow )
  {
    if( !this->lastImage.isNull() )
    {
      view->fitInView( &this->imageItem, Qt::KeepAspectRatio );
    }
  }
  else if( forceUpdate )
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

  if( !suspend )
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

  if( first || !d->updatesSuspended )
  {
    d->imageItem.setPixmap( QPixmap::fromImage( image ) );
    d->updateTransform( this );
  }
}

} // namespace tools

} // namespace kwiver
