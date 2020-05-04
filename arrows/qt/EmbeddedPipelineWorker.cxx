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

#include "EmbeddedPipelineWorker.h"

#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <QDebug>
#include <QEventLoop>
#include <QFileInfo>
#include <QThread>

#include <fstream>

namespace kv = kwiver::vital;
namespace kaq = kwiver::arrows::qt;

using kaq::EmbeddedPipelineWorker;
using kaq::EmbeddedPipelineWorkerPrivate;

using RequiredEndcaps = kaq::EmbeddedPipelineWorker::RequiredEndcaps;

namespace {

// ----------------------------------------------------------------------------
std::string
stdString( QString const& in )
{
  auto const& data = in.toLocal8Bit();
  return std::string{ data.constData(), static_cast< size_t >( data.size() ) };
}

// ----------------------------------------------------------------------------
class EmbeddedPipeline : public kwiver::embedded_pipeline
{
public:
  EmbeddedPipeline( RequiredEndcaps endcaps ) : endcaps_{ endcaps } {}

  bool hasInput() const { return this->inputConnected_; }
  bool hasOutput() const { return this->outputConnected_; }

protected:
  bool connect_input_adapter() override;
  bool connect_output_adapter() override;

  RequiredEndcaps const endcaps_;
  bool inputConnected_ = false;
  bool outputConnected_ = false;
};

// ----------------------------------------------------------------------------
bool
EmbeddedPipeline
::connect_input_adapter()
{
  this->inputConnected_ = embedded_pipeline::connect_input_adapter();
  if ( this->endcaps_.testFlag( EmbeddedPipelineWorker::RequiresInput ) )
  {
    return this->inputConnected_;
  }
  return true;
}

// ----------------------------------------------------------------------------
bool
EmbeddedPipeline
::connect_output_adapter()
{
  this->outputConnected_ = embedded_pipeline::connect_output_adapter();
  if ( this->endcaps_.testFlag( EmbeddedPipelineWorker::RequiresOutput ) )
  {
    return this->outputConnected_;
  }
  return true;
}

// ----------------------------------------------------------------------------
class Endcap : public QThread
{
public:
  Endcap( EmbeddedPipelineWorkerPrivate* q )
    : q_ptr{ q } {}

protected:
  KQ_DECLARE_PUBLIC_PTR( EmbeddedPipelineWorkerPrivate )

  void run() override;

private:
  KQ_DECLARE_PUBLIC( EmbeddedPipelineWorkerPrivate )
};

} // end namespace (anonymous)

// ----------------------------------------------------------------------------
class kaq::EmbeddedPipelineWorkerPrivate : public QThread
{
public:
  EmbeddedPipelineWorkerPrivate( RequiredEndcaps endcaps,
                                 EmbeddedPipelineWorker* q )
    : pipeline{ endcaps }, endcap{ this }, q_ptr{ q } {}

  void processOutput( kwiver::adapter::adapter_data_set_t const& output );

  EmbeddedPipeline pipeline;
  Endcap endcap;

protected:
  KQ_DECLARE_PUBLIC_PTR( EmbeddedPipelineWorker )

  void run() override;

private:
  KQ_DECLARE_PUBLIC( EmbeddedPipelineWorker )
};

KQ_IMPLEMENT_D_FUNC( EmbeddedPipelineWorker )

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorkerPrivate
::run()
{
  KQ_Q();

  if ( this->pipeline.hasInput() )
  {
    q->initializeInput( this->pipeline );
  }

  if ( this->pipeline.hasOutput() )
  {
    this->endcap.start();
  }

  if ( this->pipeline.hasInput() )
  {
    q->sendInput( this->pipeline );
  }

  this->pipeline.wait();

  if ( this->pipeline.hasOutput() )
  {
    this->endcap.wait();
  }
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorkerPrivate
::processOutput(
  kwiver::adapter::adapter_data_set_t const& output )
{
  KQ_Q();
  q->processOutput( output );
}

// ----------------------------------------------------------------------------
void
Endcap
::run()
{
  KQ_Q();

  for ( int currentFrame = 0;; ++currentFrame )
  {
    const auto& ods = q->pipeline.receive();

    if ( ods->is_end_of_data() )
    {
      return;
    }

    q->processOutput( ods );
  }
}

// ----------------------------------------------------------------------------
EmbeddedPipelineWorker
::EmbeddedPipelineWorker( RequiredEndcaps endcaps, QObject* parent )
  : QObject{ parent },
    d_ptr{ new EmbeddedPipelineWorkerPrivate{ endcaps, this } }
{
}

// ----------------------------------------------------------------------------
EmbeddedPipelineWorker::~EmbeddedPipelineWorker()
{
}

// ----------------------------------------------------------------------------
bool
EmbeddedPipelineWorker
::initialize( QString const& pipelineFile )
{
  KQ_D();

  // Set up pipeline
  try
  {
    auto const& pipelineDir = QFileInfo{ pipelineFile }.canonicalPath();

    std::ifstream pipelineStream;
    pipelineStream.open( stdString( pipelineFile ), std::ifstream::in );
    if ( !pipelineStream )
    {
      this->reportError( "Failed to initialize pipeline: pipeline file '" +
                         pipelineFile + "' could not be read",
                         "Pipeline Error" );
      return false;
    }

    d->pipeline.build_pipeline( pipelineStream, stdString( pipelineDir ) );

    if ( !this->initializePipeline( d->pipeline ) )
    {
      return false;
    }
  }
  catch ( std::exception& e )
  {
    this->reportError( "Failed to initialize pipeline: " +
                       QString::fromLocal8Bit( e.what() ),
                       "Pipeline Error" );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
bool
EmbeddedPipelineWorker
::initializePipeline(
  kwiver::embedded_pipeline& pipeline )
{
  pipeline.start();
  return true;
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorker
::execute()
{
  KQ_D();

  QEventLoop loop;

  connect( d, &QThread::finished, &loop, &QEventLoop::quit );

  d->start();
  loop.exec();
  d->wait();
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorker
::initializeInput(
  kwiver::embedded_pipeline& pipeline )
{
  Q_UNUSED( pipeline );
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorker
::sendInput( kwiver::embedded_pipeline& pipeline )
{
  Q_UNUSED( pipeline );
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorker
::processOutput(
  kwiver::adapter::adapter_data_set_t const& output )
{
  Q_UNUSED( output );
}

// ----------------------------------------------------------------------------
void
EmbeddedPipelineWorker
::reportError(
  QString const& message, QString const& subject )
{
  Q_UNUSED( subject );
  qWarning() << message;
}
