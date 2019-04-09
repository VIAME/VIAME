/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Qt VXL image_io interface
 */

#ifndef KWIVER_ARROWS_QT_EMBEDDEDPIPELINEWORKER_H_
#define KWIVER_ARROWS_QT_EMBEDDEDPIPELINEWORKER_H_

#include <arrows/qt/kq_global.h>
#include <arrows/qt/kwiver_algo_qt_export.h>

#include <sprokit/processes/adapters/adapter_types.h>

#include <QObject>

#include <memory>

namespace kwiver {

namespace vital {

class object_track_set;

} // namespace vital

class embedded_pipeline;

namespace arrows {

namespace qt {

class EmbeddedPipelineWorkerPrivate;

/// A class for embedding a KWIVER pipeline inside a Qt application.
///
/// EmbeddedPipelineWorker provides a Qt class that encapsulates a KWIVER
/// embedded pipeline in a way that simplifies integrating such pipelines into
/// a Qt application. In particular, this class takes care of setting up and
/// executing the pipeline, and provides virtual methods to allow users to
/// supply and/or receive data.
///
/// The pipeline is driven from a separate "worker" thread, which is managed by
/// the EmbeddedPipelineWorker instance.
class KWIVER_ALGO_QT_EXPORT EmbeddedPipelineWorker : public QObject
{
  Q_OBJECT

public:
  enum RequiredEndcap
  {
    RequiresInput = 1 << 0,
    RequiresOutput = 1 << 1,
    RequiresInputAndOutput = RequiresInput | RequiresOutput,
  };
  Q_DECLARE_FLAGS( RequiredEndcaps, RequiredEndcap )

  EmbeddedPipelineWorker( RequiredEndcaps = RequiresInputAndOutput,
                          QObject* parent = nullptr );
  ~EmbeddedPipelineWorker();

  /// Initialize pipeline from file.
  ///
  /// This method may be overloaded. However, users that do so should ensure
  /// that they call the base class's implementation.
  ///
  /// \param pipelineFile Path to the file which defines the pipeline.
  ///
  /// \return \c true if the pipeline was successfully created,
  ///         otherwise \c false.
  virtual bool initialize( QString const& pipelineFile );

public slots:
  /// Execute the pipeline.
  ///
  /// This starts the pipeline and waits for it to complete. An internal event
  /// loop is created so that UI events will not be blocked, however the call
  /// will not return until the pipeline is no longer running.
  ///
  /// This method may be overloaded. However, users that do so should ensure
  /// that they call the base class's implementation.
  virtual void execute();

protected:
  KQ_DECLARE_PRIVATE_RPTR( EmbeddedPipelineWorker )

  /// Initialize pipeline.
  ///
  /// This method performs any final initialization of the pipeline that needs
  /// to be performed before the pipeline scheduler is started.
  ///
  /// This method executes on the same thread which called execute(). The
  /// default implementation starts the pipeline scheduler. This is usually
  /// appropriate for pipelines that depend on input from the application, but
  /// may be suppressed by not calling the base implementation. In this case,
  /// users must ensure that they start the pipeline themselves, e.g. by
  /// overloading initializeInput().
  ///
  /// \return \c true if initialization was successful, otherwise \c false.
  virtual bool initializePipeline( kwiver::embedded_pipeline& pipeline );

  /// Prepare to send input to the pipeline.
  ///
  /// This method is called when the pipeline is getting ready to send input.
  /// Users may override this to perform any setup which needs to be done
  /// after the pipeline is set up, but before the endcap is running and before
  /// any input has been supplied to the pipeline.
  ///
  /// The default implementation does nothing.
  ///
  /// This method executes on the pipeline worker thread.
  virtual void initializeInput( kwiver::embedded_pipeline& pipeline );

  /// Send all input to the pipeline.
  ///
  /// This method is called to provide input to the pipeline. The default
  /// implementation does nothing. In order to run pipelines that depend on
  /// receiving input from the application, users must override this method.
  ///
  /// This method executes on the pipeline worker thread. This method should
  /// not return until all available input has been sent, or the user has
  /// aborted execution. Users are responsible for implementing their own
  /// cancellation mechanisms.
  virtual void sendInput( kwiver::embedded_pipeline& pipeline );

  /// Process pipeline output.
  ///
  /// This method is called after every pipeline step with the output of the
  /// pipeline endcap, so that users may act on the output produced by the
  /// pipeline. As the default implementation does nothing, most users will
  /// want to override this method.
  ///
  /// This method executes on the pipeline worker thread.
  virtual void processOutput(
    kwiver::adapter::adapter_data_set_t const& output );

  /// Report errors during pipeline initialization.
  ///
  /// This method is called to report any errors that are encountered during
  /// pipeline initialization. The default implementation uses \c qWarning to
  /// report such errors. Users may wish to override this method to report
  /// errors in some other manner. (For example, a GUI application may wish to
  /// report errors using QMessageBox.)
  ///
  /// This method executes on the same thread which called execute().
  /// Overrides do not need to call the base implementation. (In fact, doing so
  /// is usually undesired.)
  ///
  /// \param message Description of the error which occurred.
  /// \param subject Context of the error (e.g. suitable for a dialog title).
  virtual void reportError( QString const& message, QString const& subject );

private:
  KQ_DECLARE_PRIVATE( EmbeddedPipelineWorker )
};

Q_DECLARE_OPERATORS_FOR_FLAGS( EmbeddedPipelineWorker::RequiredEndcaps )

} // namespace qt

} // namespace arrows

} // namespace kwiver

#endif
