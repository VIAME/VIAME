// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_PIPELINE_VIEWER_MAINWINDOW_H_
#define KWIVER_TOOLS_PIPELINE_VIEWER_MAINWINDOW_H_

#include <arrows/qt/kq_global.h>

#include <QMainWindow>

class QImage;

namespace kwiver {

namespace tools {

class MainWindowPrivate;

// ----------------------------------------------------------------------------
class MainWindow : public QMainWindow
{
  Q_OBJECT
  Q_PROPERTY( bool fitToWindow READ fitToWindow WRITE setFitToWindow )

public:
  explicit MainWindow( QWidget* parent = nullptr );
  ~MainWindow();

  bool updatesSuspended() const;
  bool fitToWindow() const;

public slots:
  void executePipeline();
  void executePipeline( QString const& path );

  void setUpdatesSuspended( bool suspend );
  void setFitToWindow( bool fit );

protected:
  KQ_DECLARE_PRIVATE_RPTR( MainWindow )

private:
  KQ_DECLARE_PRIVATE( MainWindow )
};

} // namespace tools

} // namespace kwiver

#endif
