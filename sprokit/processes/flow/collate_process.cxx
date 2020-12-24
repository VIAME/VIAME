// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file collate_process.cxx
 *
 * \brief Implementation of the collate process.
 */

#include "collate_process.h"

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/stamp.h>

#include <vital/util/string.h>

#include <map>
#include <string>

namespace sprokit
{

/**
 * \class collate_process
 *
 * \brief A process for collating input data from multiple input edges.
 *
 * \note Edges for a \portvar{tag} may \em only be connected after the
 * \port{status/\portvar{tag}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{tag} must be \port{status/\portvar{tag}}.
 *
 * \process Collate incoming data into a single stream.  A collation
 * operation reads input from a set of input ports and serializes that
 * data to a single output port. This collation process can handle
 * multiple collation operations. Each set of collation ports is
 * identified by a unique \b tag.
 *
 * \iports
 *
 * \iport{status/\portvar{tag}} The status of the result \portvar{tag}.
 * \iport{coll/\portvar{tag}/\portvar{item}} A port to collate data for
 *                                            \portvar{tag} from. Data is
 *                                            collated from ports in
 *                                            ASCII-betical order.
 *
 * \oports
 *
 * \oport{res/\portvar{tag}} The collated result \portvar{tag}.
 *
 * \reqs
 *
 * \req Each input port \port{status/\portvar{tag}} must be connected.
 * \req Each \portvar{tag} must have at least two inputs to collate.
 * \req Each output port \port{res/\portvar{tag}} must be connected.
 *
 * The status port is used to signal upstream status about the collate
 * set. Only complete and flush packet types are handled. Regular data
 * packets are ignored on that port.
 *
 * This process automatically makes the input and output types for
 * each \b tag the same based on the type of the port that is first
 * connected.
 *
 * \note
 * It is not immediately apparent how the input ports become sorted in
 * ASCII-betical order on "item" order.
 *
 * \code
 process collate :: collate_process

 # -- Connect collate set "input1"
 # status port
 connect foo.p1_stat  to  collate.status/input1

 # actual data ports
 connect foo_1.out       to  collate.coll/input1/A
 connect foo_2.out       to  collate.coll/input1/B

 connect collate.res/input1  to bar.input

 # -- Connect collate set "input2"
 # status port can feed multiple groups
 connect foo.p1_stat  to  collate.status/input2

 # actual data ports
 connect foo_1.out       to  collate.coll/input2/A
 connect foo_2.out       to  collate.coll/input2/B
 connect foo_3.out       to  collate.coll/input2/C

 connect collate.res/input2  to bar.other

 * \endcode
 *
 * \todo Add configuration to allow forcing a number of inputs for a result.
 * \todo Add configuration to allow same number of sources for all results.
 *
 * \ingroup process_flow
 */

class collate_process::priv
{
public:
  priv();
  ~priv();

  typedef port_t tag_t;

  // This class stores info for each tag.
  class tag_info
  {
  public:
    tag_info();
    ~tag_info();

    ports_t ports; // list of port names
    ports_t::const_iterator cur_port;
  };
  typedef std::map<tag_t, tag_info> tag_data_t;

  tag_data_t tag_data; // tag table

  tag_t tag_for_coll_port(port_t const& port) const;

  static port_t const res_sep;
  static port_t const port_res_prefix;
  static port_t const port_status_prefix;
  static port_t const port_coll_prefix;
};

process::port_t const collate_process::priv::res_sep = port_t("/");
process::port_t const collate_process::priv::port_res_prefix = port_t("res") + res_sep;
process::port_t const collate_process::priv::port_status_prefix = port_t("status") + res_sep;
process::port_t const collate_process::priv::port_coll_prefix = port_t("coll") + res_sep;

/**
 * \internal
 *
 * Ports on the \ref distribute_process are broken down as follows:
 *
 *   \portvar{type}/\portvar{tag}[/\portvar{item}]
 *
 * The port name is broken down as follows:
 *
 * <dl>
 * \term{\portvar{type}}
 *   \termdef{The type of the port. This must be one of \type{res},
 *   \type{status}, or \type{coll}.}
 * \term{\portvar{tag}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{item}}
 *   \termdef{Only required for \type{coll}-type ports. Data from the same
 *   \portvar{tag} stream from its \type{res} port is collected in sorted order
 *   over all of the \type{coll} ports.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{status}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{res} and \type{coll}
 *   ports for the tag will not cause errors.}
 * \term{\type{res}}
 *   \termdef{This port for the given tag is where the data for a stream leaves
 *   the process. The stamp from the \type{status} port for the \portvar{tag} is
 *   applied.}
 * \term{\type{coll}}
 *   \termdef{These ports for a given \portvar{tag} receive data from a set of
 *   sources, likely made by the \ref distribute_process. Data is collected in
 *   sorted ordef of the \type{item} name and sent out the \type{res} port for
 *   the \portvar{tag}.}
 * </dl>
 */

collate_process
::collate_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  // This process manages its own inputs.
  this->set_data_checking_level(check_none);
}

collate_process
::~collate_process()
{
}

// ------------------------------------------------------------------
// Post connection processing
void
collate_process
::_init()
{
  for (priv::tag_data_t::value_type& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    priv::tag_info& info = tag_data.second;
    ports_t const& ports = info.ports; // list of port names

    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to collate "
                                 "to for the \"" + tag + "\" result data";

      VITAL_THROW( invalid_configuration_exception, name(), reason);
    }

    // Now here's some port frequency magic
    frequency_component_t const ratio = ports.size();
    port_frequency_t const freq = port_frequency_t(1, ratio);

    // Set port frequency for all input ports.
    for (port_t const& port : ports)
    {
      set_input_port_frequency(port, freq);
    }

    // Set iterator to start of list.
    info.cur_port = ports.begin();
  }

  process::_init();
}

// ------------------------------------------------------------------
void
collate_process
::_reset()
{
  for (priv::tag_data_t::value_type const& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const output_port = priv::port_res_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info const& info = tag_data.second;
    ports_t const& ports = info.ports;

    for (port_t const& port : ports)
    {
      remove_input_port(port);
    }

    remove_input_port(status_port);
    remove_output_port(output_port);
  }

  d->tag_data.clear();

  process::_reset();
}

// ------------------------------------------------------------------
void
collate_process
::_step()
{
  ports_t complete_ports;

  // Loop over all tags (input groups)
  for (priv::tag_data_t::value_type& tag_data : d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const output_port = priv::port_res_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info& info = tag_data.second;

    // Check status input port. This will give us information on the
    // upstream process.
    edge_datum_t const status_edat = grab_from_port(status_port);
    datum_t const& status_dat = status_edat.datum;
    datum::type_t const status_type = status_dat->type();

    // Test to see if complete.
    bool const is_complete = (status_type == datum::complete);

    if (is_complete || (status_type == datum::flush))
    {
      // echo the input to the output port
      push_to_port(output_port, status_edat);

      // Flush this set of inputs
      for (port_t const& port : info.ports)
      {
        (void)grab_from_port(port);
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
      // There is real data on the input ports. Grab data from the
      // current input port and push to the output.
      edge_datum_t const coll_dat = grab_from_port(*info.cur_port);

      push_to_port(output_port, coll_dat);
    }

    // Advance to next port in the group, and wrap at the end.
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
collate_process
::_properties() const
{
  properties_t consts = process::_properties();

  consts.insert(property_unsync_input);

  return consts;
}

// ------------------------------------------------------------------
// Intercept input port connection so we can create the requested port
void
collate_process
::input_port_undefined(port_t const& port)
{
  // Is this a status port (starts with "status/")
  if (kwiver::vital::starts_with(port, priv::port_status_prefix))
  {
    // Extract TAG sub-string from port name
    priv::tag_t const tag = port.substr(priv::port_status_prefix.size());

    // If TAG does not exist
    if ( ! d->tag_data.count(tag) )
    {
      // This is the first time the status port is being connected to.
      priv::tag_info info;
      d->tag_data[tag] = info;

      port_flags_t required;
      required.insert(flag_required);

      // Create input status port "tag"
      declare_input_port(
        port,
        type_none,
        required,
        port_description_t("The original status for the result " + tag + "."));

      // Create output port "res/tag"
      declare_output_port(
        priv::port_res_prefix + tag,
        type_flow_dependent + tag, // note the tag magic on port type
        required,
        port_description_t("The output port for " + tag + "."));
    }
  } // end status port

  // Get the canonical tag string from a "coll/xx" port name.
  // Note that this name will be empty for "status/xx" port names
  priv::tag_t const tag = d->tag_for_coll_port(port);

  // If the status port has already been created for this "coll/" port.
  if ( ! tag.empty() )
  {
    // Get entry based on the tag string
    priv::tag_info& info = d->tag_data[tag];

    // Add this port to the info list for this tag
    info.ports.push_back(port);

    port_flags_t required;
    required.insert(flag_required);

    // Open an input port for the name
    declare_input_port(
      port,
      type_flow_dependent + tag, // note the tag magic on port type
      required,
      port_description_t("An input for the " + tag + " data."));
  }
}

// ------------------------------------------------------------------
collate_process::priv
::priv()
  : tag_data()
{
}

collate_process::priv
::~priv()
{
}

// ------------------------------------------------------------------
/*
 * @brief Find tag name that corresponds to the port name.
 *
 * This method looks through the list of current tags to see if the
 * supplied port is in that table.
 *
 * @param port Name of the port
 *
 * @return Tag name
 */
collate_process::priv::tag_t
collate_process::priv
::tag_for_coll_port(port_t const& port) const
{
  // Does this port start with "coll/"
  if (kwiver::vital::starts_with(port, priv::port_coll_prefix))
  {
    // Get the part of the port name after the prefix
    // This could be "tag/item"
    port_t const no_prefix = port.substr(priv::port_coll_prefix.size());

    // loop over all tags seen so far
    for (priv::tag_data_t::value_type const& data : tag_data)
    {
      tag_t const& tag = data.first; // tag string
      port_t const tag_prefix = tag + priv::res_sep;

      // If the port name without the prefix is "tag/*" then return
      // base tag string
      if (kwiver::vital::starts_with(no_prefix, tag_prefix))
      {
        return tag;
      }
    }
  }

  return tag_t();
}

// ------------------------------------------------------------------
collate_process::priv::tag_info
::tag_info()
  : ports()
  , cur_port()
{
}

collate_process::priv::tag_info
::~tag_info()
{
}

} // end namespace
