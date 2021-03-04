// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file distribute_process.cxx
 *
 * \brief Implementation of the distribute process.
 */

#include "distribute_process.h"

#include <vital/util/string.h>

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/stamp.h>

#include <map>
#include <string>

namespace sprokit {

/**
 * \class distribute_process
 *
 * \brief A process for distributing input data to multiple output edges.
 *
 * \note Edges for a \portvar{tag} may \em only be connected after the
 * \port{status/\portvar{tag}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{tag} must be \port{status/\portvar{tag}}.
 *
 * \process Distribute input data among many output processes.
 *
 * \iports
 *
 * \iport{src/\portvar{tag}} The source input \portvar{tag}.
 *
 * \oports
 *
 * \oport{status/\portvar{tag}} The status of the input \portvar{tag}.
 * \oport{dist/\portvar{tag}/\portvar{group}} A port to distribute the input
 *                                            \portvar{tag} to. Data is
 *                                            distributed in ASCII-betical order.
 *
 * \reqs
 *
 * \req Each input port \port{src/\portvar{tag}} must be connected.
 * \req Each output port \port{status/\portvar{res}} must be connected.
 * \req Each \portvar{res} must have at least two outputs to distribute to.
 *
 * \code
process distrib :: distribute

# connect input port
connect foo.p1_data      to      distrib.src/set1

# status output port
connect distrib.status/set1  to     bar.status

# connect output ports
connect distrib.dist/set1/A   to    bar.data
connect distrib.dist/set1/B   to    bar_1.data

 * \endcode
 *
 * \todo Add configuration to allow forcing a number of outputs for a source.
 * \todo Add configuration to allow same number of outputs for all sources.
 *
 * \ingroup process_flow
 */

class distribute_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t group_t;
    typedef port_t tag_t;

    class tag_info
    {
      public:
        tag_info();
        ~tag_info();

        ports_t ports;
        ports_t::const_iterator cur_port;
    };
    typedef std::map<tag_t, tag_info> tag_data_t;

    tag_data_t tag_data;

    // Port name break down.
    tag_t tag_for_dist_port(port_t const& port) const;

    static port_t const src_sep;
    static port_t const port_src_prefix;
    static port_t const port_status_prefix;
    static port_t const port_dist_prefix;
};

process::port_t const distribute_process::priv::src_sep = port_t("/");
process::port_t const distribute_process::priv::port_src_prefix = port_t("src") + src_sep;
process::port_t const distribute_process::priv::port_status_prefix = port_t("status") + src_sep;
process::port_t const distribute_process::priv::port_dist_prefix = port_t("dist") + src_sep;

/**
 * \internal
 *
 * Ports on the \ref distribute_process are broken down as follows:
 *
 *   \portvar{type}/\portvar{tag}[/\portvar{group}]
 *
 * The port name is broken down as follows:
 *
 * <dl>
 * \term{\portvar{type}}
 *   \termdef{The type of the port. This must be one of \type{src},
 *   \type{status}, or \type{dist}.}
 * \term{\portvar{tag}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{group}}
 *   \termdef{Only required for \type{dist}-type ports. Data from the same
 *   \portvar{tag} stream from its \type{src} port is distributed in sorted
 *   order over all of the \type{dist} ports.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{status}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{src} and \type{dist}
 *   ports for the tag will not cause errors. The stamp on the port represents
 *   the status of the original stream coming in the \type{src} port for the
 *   tag.}
 * \term{\type{src}}
 *   \termdef{This port for the given tag is where the original stream comes
 *   into the process.}
 * \term{\type{dist}}
 *   \termdef{These ports for a given \portvar{tag} receive a subset of the data
 *   from the corresponding \type{src} port. They are used in sorted order of
 *   the \type{group} name.}
 * </dl>
 */

distribute_process
::distribute_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  // This process manages its own inputs.
  set_data_checking_level(check_none);
}

distribute_process
::~distribute_process()
{
}

// ------------------------------------------------------------------
// Post connection processing
void
distribute_process
::_init()
{
  // Loop over all connected tags
  for (priv::tag_data_t::value_type& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    priv::tag_info& info = tag_data.second;
    ports_t const& ports = info.ports;

    // Ensure that the extra process is actually doing work.
    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to distribute "
                                 "to for the \"" + tag + "\" source data";

      VITAL_THROW( invalid_configuration_exception, name(), reason);
    }

    // Do some port frequency magi
    frequency_component_t const ratio = ports.size();
    port_frequency_t const freq = port_frequency_t(1, ratio);

    for (port_t const& port : ports)
    {
      set_output_port_frequency(port, freq);
    }

    info.cur_port = ports.begin();
  }

  process::_init();
}

// ------------------------------------------------------------------
void
distribute_process
::_reset()
{
  for (priv::tag_data_t::value_type const& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const input_port = priv::port_src_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info const& info = tag_data.second;
    ports_t const& ports = info.ports;

    for (port_t const& port : ports)
    {
      remove_output_port(port);
    }

    remove_output_port(status_port);
    remove_input_port(input_port);
  }

  d->tag_data.clear();

  process::_reset();
}

// ------------------------------------------------------------------
void
distribute_process
::_step()
{
  ports_t complete_ports;

  // Loop over all tags to handle all distribution sets.
  for (priv::tag_data_t::value_type& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const input_port = priv::port_src_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info& info = tag_data.second;

    // Check status input port. This will give us information on the
    // upstream process.
    edge_datum_t const src_edat = grab_from_port(input_port);
    datum_t const& src_dat = src_edat.datum;
    stamp_t const& src_stamp = src_edat.stamp;

    datum::type_t const src_type = src_dat->type();

    bool const is_complete = (src_type == datum::complete);

    if (is_complete || (src_type == datum::flush))
    {
      // echo to downstream status port
      push_to_port(status_port, src_edat);

      // echo to all output ports in this set.
      for (port_t const& port : info.ports)
      {
        push_to_port(port, src_edat);
      }

      // If the upstream process is done, then mark this tag as done.
      if (is_complete)
      {
        complete_ports.push_back(tag);

        continue;
      }
    }
    else
    {
      // There is real data on the input port. Send it to the next output port.
      edge_datum_t const status_edat = edge_datum_t(datum::empty_datum(), src_stamp);

      push_to_port(status_port, status_edat);
      push_to_port(*info.cur_port, src_edat);
    }

    // Go to next output port in the set.
    ++info.cur_port;
    if (info.cur_port == info.ports.end())
    {
      info.cur_port = info.ports.begin();
    }
  } // end foreach

  // Process all ports/tags that have completed. When a status port
  // reports complete on a tag, that tag is erased from the local
  // map. When that map is empty, then we are all done and can complete.
  for (port_t const& port : complete_ports)
  {
    d->tag_data.erase(port);
  }

  if (d->tag_data.empty())
  {
    mark_process_as_complete();
  }

  process::_step();
}

// ------------------------------------------------------------------
process::properties_t
distribute_process
::_properties() const
{
  properties_t consts = process::_properties();

  consts.insert(property_unsync_output);

  return consts;
}

// ------------------------------------------------------------------
// Intercept the connect operation so we can create the needed output port
void
distribute_process
::output_port_undefined(port_t const& port)
{
  // Does this port name start with the status prefix "status/"
  if (kwiver::vital::starts_with(port, priv::port_status_prefix))
  {
    // Extract the tag sub-string from the status port name.
    priv::tag_t const tag = port.substr(priv::port_status_prefix.size());

    // If this tag has not been seen before, create internal tag
    // tracking structures.
    if ( ! d->tag_data.count(tag) )
    {
      priv::tag_info info;
      d->tag_data[tag] = info;

      port_flags_t required;
      required.insert(flag_required);

      // Create input port that will be accept data to be distributed
      declare_input_port(
        priv::port_src_prefix + tag,
        type_flow_dependent + tag, // note the tag magic on port type
        required,
        port_description_t("The input port for " + tag + "."));

      // Create the status output port for downstream signaling.
      declare_output_port(
        port,
        type_none,
        required,
        port_description_t("The status for the input " + tag + "."));
    }
  }

  // Get the canonical tag string from a "dist/xx" port name.
  // Note that this name will be empty for "status/xx" port names
  priv::tag_t const tag = d->tag_for_dist_port(port);

  // If the status port has already been created for this "dist/" port.
  if ( ! tag.empty() )
  {
    priv::tag_info& info = d->tag_data[tag];

    // Add output port to the list
    info.ports.push_back(port);

    port_flags_t required;
    required.insert(flag_required);

    // Create the output data port for this item.
    declare_output_port(
      port,
      type_flow_dependent + tag, // note the tag magic on port type
      required,
      port_description_t("An output for the " + tag + " data."));
  }
}

// ------------------------------------------------------------------
distribute_process::priv
::priv()
  : tag_data()
{
}

distribute_process::priv
::~priv()
{
}

// ------------------------------------------------------------------
distribute_process::priv::tag_t
distribute_process::priv
::tag_for_dist_port(port_t const& port) const
{
  if (kwiver::vital::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    for (priv::tag_data_t::value_type const& data : tag_data)
    {
      tag_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::src_sep;

      if (kwiver::vital::starts_with(no_prefix, tag_prefix))
      {
        return tag;
      }
    }
  }

  return tag_t();
}

// ------------------------------------------------------------------
distribute_process::priv::tag_info
::tag_info()
  : ports()
  , cur_port()
{
}

distribute_process::priv::tag_info
::~tag_info()
{
}

} // end namespace
