// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_QT_WIDGETS_METADATAVIEW_H_
#define KWIVER_ARROWS_QT_WIDGETS_METADATAVIEW_H_

#include <arrows/qt/widgets/kwiver_algo_qt_widgets_export.h>
#include <vital/types/metadata_map.h>

#include <qtGlobal.h>

#include <QScrollArea>

namespace kwiver {
namespace arrows {
namespace qt {

class MetadataViewPrivate;

class KWIVER_ALGO_QT_WIDGETS_EXPORT MetadataView : public QScrollArea
{
  Q_OBJECT

public:
  explicit MetadataView( QWidget* parent = 0 );
  virtual ~MetadataView();

  bool eventFilter( QObject* sender, QEvent* e ) override;

public slots:
  void updateMetadata(
    std::shared_ptr< kwiver::vital::metadata_map::map_metadata_t > );
  void updateMetadata( kwiver::vital::metadata_vector const& );

protected:
  void changeEvent( QEvent* e ) override;

private:
  QTE_DECLARE_PRIVATE_RPTR( MetadataView )
  QTE_DECLARE_PRIVATE( MetadataView )

  QTE_DISABLE_COPY( MetadataView )
};

} // namespace qt
} // namespace arrows
} // namespace kwiver

#endif
