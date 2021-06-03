// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "LoggerView.h"
#include "ui_LoggerView.h"

#include <qtStlUtil.h>

namespace kv = kwiver::vital;

namespace kwiver {
namespace arrows {
namespace qt {

// ----------------------------------------------------------------------------
class LoggerViewPrivate
{
public:
  Ui::LoggerView m_UI;
};

QTE_IMPLEMENT_D_FUNC( LoggerView )

// ----------------------------------------------------------------------------
LoggerView
::LoggerView(
  QWidget* parent, Qt::WindowFlags flags )
  : QWidget{ parent, flags }, d_ptr{ new LoggerViewPrivate }
{
  QTE_D();

  // Set up UI
  d->m_UI.setupUi( this );

  QFont font{ "Monospace", 10 };
  font.setStyleHint( QFont::TypeWriter );
  d->m_UI.loggerText->document()->setDefaultFont( font );

  QObject::connect( this, &LoggerView::messageLogged,
                    this, &LoggerView::appendMessage );
}

// ----------------------------------------------------------------------------
LoggerView::~LoggerView()
{
}

// ----------------------------------------------------------------------------
void
LoggerView
::logHandler( kv::kwiver_logger::log_level_t level,
              std::string const& name,
              std::string const& msg,
              VITAL_UNUSED kv::logger_ns::location_info const& loc )
{
  std::string levelStr = kv::kwiver_logger::get_level_string( level );
  auto html = qtString( "<b><font color=\"red\">" + levelStr + "</font> " +
                        name + "</b>: <pre>" + msg + "</pre>" );
  emit messageLogged( html );
}

// ----------------------------------------------------------------------------
void
LoggerView
::appendMessage( QString const& msg )
{
  QTE_D();
  d->m_UI.loggerText->appendHtml( msg );
}

} // namespace qt
} // namespace arrows
} // namespace kwiver
