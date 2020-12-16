// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_QT_WIDGETS_LOGGERVIEW_H_
#define KWIVER_ARROWS_QT_WIDGETS_LOGGERVIEW_H_

#include <arrows/qt/widgets/kwiver_algo_qt_widgets_export.h>

#include <vital/logger/logger.h>

#include <qtGlobal.h>

#include <QWidget>

namespace kwiver {
namespace arrows {
namespace qt {

class LoggerViewPrivate;

class KWIVER_ALGO_QT_WIDGETS_EXPORT LoggerView : public QWidget
{
  Q_OBJECT

public:
  explicit LoggerView(
    QWidget* parent = nullptr, Qt::WindowFlags flags = {} );
  ~LoggerView();

  void logHandler( kwiver::vital::kwiver_logger::log_level_t,
                   std::string const& name, std::string const& msg,
                   kwiver::vital::logger_ns::location_info const& loc );

signals:
  void messageLogged( QString const& msg );

public slots:
  void appendMessage( QString const& msg );

private:
  QTE_DECLARE_PRIVATE_RPTR( LoggerView )
  QTE_DECLARE_PRIVATE( LoggerView )

  QTE_DISABLE_COPY( LoggerView )
};

} // namespace qt
} // namespace arrows
} // namespace kwiver

#endif
