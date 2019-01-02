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

  if ( !path.isEmpty() )
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

  if ( worker.initialize( path ) )
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
