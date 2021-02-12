// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "PipelineWorker.h"

#include <arrows/qt/image_container.h>

#include <sprokit/processes/adapters/adapter_data_set.h>

#include <QMessageBox>

using super = kwiver::arrows::qt::EmbeddedPipelineWorker;
using qt_image_container = kwiver::arrows::qt::image_container;

namespace kwiver {

namespace tools {

class PipelineWorkerPrivate {};

// ----------------------------------------------------------------------------
PipelineWorker
::PipelineWorker( QWidget* parent ) : super{ RequiresOutput, parent }
{
}

// ----------------------------------------------------------------------------
PipelineWorker
::~PipelineWorker()
{
}

// ----------------------------------------------------------------------------
void
PipelineWorker
::processOutput( adapter::adapter_data_set_t const& output )
{
  auto const& iter = output->find( "image" );

  if( iter != output->end() )
  {
    auto const& image =
      iter->second->get_datum< vital::image_container_sptr >();
    if( image )
    {
      if( auto qi = std::dynamic_pointer_cast< qt_image_container >( image ) )
      {
        emit this->imageAvailable( *qi );
      }
      else
      {
        emit this->imageAvailable(
          qt_image_container::vital_to_qt( image->get_image() ) );
      }
    }
  }
}

// ----------------------------------------------------------------------------
void
PipelineWorker
::reportError( QString const& message, QString const& subject )
{
  auto* const p = qobject_cast< QWidget* >( this->parent() );
  QMessageBox::warning( p, subject, message );
}

} // namespace tools

} // namespace kwiver
