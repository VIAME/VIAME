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
  if ( iter != output->end() )
  {
    auto const& image =
      iter->second->get_datum< vital::image_container_sptr >();
    if ( image )
    {
      if ( auto qi = std::dynamic_pointer_cast< qt_image_container >( image ) )
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
