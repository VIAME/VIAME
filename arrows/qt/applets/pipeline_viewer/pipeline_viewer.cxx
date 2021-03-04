// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipeline_viewer.h"

#include "MainWindow.h"

#include <vital/range/iota.h>

#include <QApplication>

// ----------------------------------------------------------------------------
int
kwiver::tools::pipeline_viewer
::run()
{
  std::vector< std::string > const& args = applet_args();
  // Create mutable arguments (needed by QApplication)
  auto argc = static_cast< int >( args.size() );
  auto argv = std::unique_ptr< char*[] >{ new char*[ argc ] };

  auto margs = std::vector< std::unique_ptr< char[] > >{};

  for( auto const i : vital::range::iota( args.size() ) )
  {
    auto const& arg = args[ i ];
    auto const l = arg.size() + 1;

    auto marg = std::unique_ptr< char[] >{ new char[ l ] };
    memcpy( marg.get(), arg.c_str(), l );

    argv[ i ] = marg.get();
    margs.push_back( std::move( marg ) );
  }

  // Create QApplication
  QApplication app{ argc, argv.get() };

  // Create and show main window
  kwiver::tools::MainWindow window;
  window.show();

  // Execute event loop
  return app.exec();
}
