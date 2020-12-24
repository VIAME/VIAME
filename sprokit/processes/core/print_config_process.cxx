// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "print_config_process.h"

#include <vital/config/config_block_formatter.h>
#include <sprokit/processes/kwiver_type_traits.h>

namespace kwiver {

//----------------------------------------------------------------
// Private implementation class
class print_config_process::priv
{
public:
  priv();
  ~priv();

  std::set< sprokit::process::port_t > m_active_ports;

}; // end priv class

// ==================================================================
print_config_process::
print_config_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new print_config_process::priv )
{
}

print_config_process::
~print_config_process()
{
}

// ------------------------------------------------------------------
void
print_config_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr my_config = get_config();
  kwiver::vital::config_block_formatter fmt( my_config );
  fmt.print( std::cout );
}

// ------------------------------------------------------------------
void
print_config_process::
input_port_undefined(port_t const& port)
{
  // If we have not created the port, then make a new one.
  if ( d->m_active_ports.count( port ) == 0 )
  {
    port_flags_t p_flags;

    LOG_TRACE( logger(), "Creating input port: \"" << port << "\" on process \"" << name() << "\"" );

    // create a new port
    declare_input_port( port,     // port name
                        type_any, // port data type expected
                        p_flags,
                        port_description_t("Input for " + port)
      );

    // Add to our list of existing ports
    d->m_active_ports.insert( port );
  }
}

// ------------------------------------------------------------------
void
print_config_process::
_step()
{
  // Take a peek at the first port to see if it is the end of data
  // marker.  If so, push end marker into our output interface queue.
  // The assumption is that if the first port is at end, then they all
  // are.
  auto edat = this->peek_at_port( *d->m_active_ports.begin() );
  if ( edat.datum->type() == sprokit::datum::complete )
  {
    LOG_DEBUG( logger(), "End of data detected." );

    mark_process_as_complete();

    return;
  }

  // The grab call is blocking, so it will wait until data is there.
  for( auto const p : d->m_active_ports )
  {
    auto dtm = this->grab_datum_from_port( p );
  } // end foreach
}

// ================================================================
print_config_process::priv
::priv()
{
}

print_config_process::priv
::~priv()
{
}

} // end namespace kwiver
