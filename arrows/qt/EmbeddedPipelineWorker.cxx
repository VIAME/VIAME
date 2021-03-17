// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "EmbeddedPipelineWorker.h"

#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
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
  EmbeddedPipeline( RequiredEndcaps );

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
EmbeddedPipeline
::EmbeddedPipeline( RequiredEndcaps endcaps )
  : endcaps_{ endcaps }
{
  // Helper to convert to std::string using the system locale (don't assume it
  // is UTF-8 like QString::toStdString does)
  auto ss = []( QString const& qs ){
    auto const& data = qs.toLocal8Bit();
    auto const l = static_cast< size_t >( data.size() );
    return std::string{ data.constData(), l };
  };

  auto const& an = ss( qApp->applicationName() );
  auto const& av = ss( qApp->applicationVersion() );
  auto ap = std::string{};

  auto const appPath = qApp->applicationDirPath();
  if( appPath.endsWith( QStringLiteral( "bin" ) ) )
  {
    ap = ss( QDir{ appPath }.absoluteFilePath( QStringLiteral( ".." ) ) );
  }
  else
  {
    ap = ss( appPath );
  }

  this->set_application_information( an, av, ap );
}

// ----------------------------------------------------------------------------
bool
EmbeddedPipeline
::connect_input_adapter()
{
  this->inputConnected_ = embedded_pipeline::connect_input_adapter();
  if( this->endcaps_.testFlag( EmbeddedPipelineWorker::RequiresInput ) )
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
  if( this->endcaps_.testFlag( EmbeddedPipelineWorker::RequiresOutput ) )
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

  bool atEnd = false;

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

  if( this->pipeline.hasInput() )
  {
    q->initializeInput( this->pipeline );
  }

  if( this->pipeline.hasOutput() )
  {
    this->atEnd = false;
    this->endcap.start();
  }

  if( this->pipeline.hasInput() )
  {
    q->sendInput( this->pipeline );
  }

  this->pipeline.wait();

  if( this->pipeline.hasOutput() )
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

  if( output->is_end_of_data() )
  {
    this->atEnd = true;
    emit q->finished();
    return;
  }

  q->processOutput( output );
}

// ----------------------------------------------------------------------------
void
Endcap
::run()
{
  KQ_Q();

  while( !q->atEnd )
  {
    q->processOutput( q->pipeline.receive() );
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
    if( !pipelineStream )
    {
      this->reportError( "Failed to initialize pipeline: pipeline file '" +
                         pipelineFile + "' could not be read",
                         "Pipeline Error" );
      return false;
    }

    d->pipeline.build_pipeline( pipelineStream, stdString( pipelineDir ) );

    if( !this->initializePipeline( d->pipeline ) )
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
