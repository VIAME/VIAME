// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
