// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

  virtual ~PipelineWorker();

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
