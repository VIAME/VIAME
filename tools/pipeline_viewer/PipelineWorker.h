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

#ifndef KWIVER_TOOLS_PIPELINE_VIEWER_PIPELINEWORKER_H_
#define KWIVER_TOOLS_PIPELINE_VIEWER_PIPELINEWORKER_H_

#include <arrows/qt/EmbeddedPipelineWorker.h>

#include <QImage>

namespace kwiver {

namespace tools {

class PipelineWorkerPrivate;

// ----------------------------------------------------------------------------
class PipelineWorker : public arrows::qt::EmbeddedPipelineWorker
{
  Q_OBJECT

public:
  explicit PipelineWorker( QWidget* parent = nullptr );

  ~PipelineWorker() override;

signals:
  void imageAvailable( QImage const& );

protected:
  KQ_DECLARE_PRIVATE_RPTR( PipelineWorker );

  virtual void processOutput(
    adapter::adapter_data_set_t const& output ) override;

  virtual void reportError(
    QString const& message, QString const& subject ) override;

private:
  KQ_DECLARE_PRIVATE( PipelineWorker );
};

} // namespace tools

} // namespace kwiver

#endif
