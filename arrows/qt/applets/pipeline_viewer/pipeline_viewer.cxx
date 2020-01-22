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
  for ( auto const i : vital::range::iota( args.size() ) )
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
