/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "register_processes.h"
#include <sprokit/pipeline/process_registry.h>

// -- list processes to register --
#include "kw_archive_writer_process.h"


void register_processes()
{
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "kwiver_processes" );

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  if ( registry->is_module_loaded( module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------
  registry->register_process(
    "kw_archive_writer", "A process to write kw archives",
    sprokit::create_process< kwiver::kw_archive_writer_process > );


  registry->mark_module_as_loaded( module_name );
}


