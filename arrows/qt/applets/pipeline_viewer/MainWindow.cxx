// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "ImageView.h"
#include "PipelineWorker.h"

#include <QFileDialog>

namespace kwiver {

namespace tools {

// ----------------------------------------------------------------------------
class MainWindowPrivate
{
public:
  Ui::MainWindow ui;
};

KQ_IMPLEMENT_D_FUNC( MainWindow )

// ----------------------------------------------------------------------------
MainWindow
::MainWindow( QWidget* parent )
  : QMainWindow{ parent }, d_ptr{ new MainWindowPrivate }
{
  KQ_D();

  d->ui.setupUi( this );

  connect( d->ui.actionExecutePipeline, &QAction::triggered,
           this, QOverload<>::of( &MainWindow::executePipeline ) );
  connect( d->ui.actionSuspendUpdates, &QAction::toggled,
           this, &MainWindow::setUpdatesSuspended );
  connect( d->ui.actionFitToWindow, &QAction::toggled,
           this, &MainWindow::setFitToWindow );
}

// ----------------------------------------------------------------------------
MainWindow::~MainWindow()
{
}

// ----------------------------------------------------------------------------
void
MainWindow
::executePipeline()
{
  auto const& path = QFileDialog::getOpenFileName(
    this, "Open Pipeline File", {},
    "KWIVER Pipelines (*.pipe);;"
    "All files (*)" );

  if( !path.isEmpty() )
  {
    this->executePipeline( path );
  }
}

// ----------------------------------------------------------------------------
void
MainWindow
::executePipeline( QString const& path )
{
  KQ_D();

  PipelineWorker worker{ this };

  if( worker.initialize( path ) )
  {
    d->ui.actionExecutePipeline->setEnabled( false );
    d->ui.actionSuspendUpdates->setEnabled( true );

    connect( &worker, &PipelineWorker::imageAvailable,
             d->ui.imageView, &ImageView::displayImage );

    worker.execute();

    d->ui.actionExecutePipeline->setEnabled( true );
  }
}

// ----------------------------------------------------------------------------
bool
MainWindow
::updatesSuspended() const
{
  KQ_D();
  return d->ui.actionSuspendUpdates->isChecked();
}

// ----------------------------------------------------------------------------
void
MainWindow
::setUpdatesSuspended( bool suspend )
{
  KQ_D();
  d->ui.imageView->setUpdatesSuspended( suspend );
}

// ----------------------------------------------------------------------------
bool
MainWindow
::fitToWindow() const
{
  KQ_D();
  return d->ui.actionFitToWindow->isChecked();
}

// ----------------------------------------------------------------------------
void
MainWindow
::setFitToWindow( bool fit )
{
  KQ_D();

  d->ui.imageView->setScaleMode(
    fit ? ImageView::FitToWindow : ImageView::OriginalSize );
}

} // namespace tools

} // namespace kwiver
