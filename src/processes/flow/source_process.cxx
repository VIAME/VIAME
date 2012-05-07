/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "source_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <algorithm>
#include <string>

/**
 * \file source_process.cxx
 *
 * \brief Implementation of the source process.
 */

namespace vistk
{

class source_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t tag_t;
    typedef std::vector<tag_t> tags_t;

    stamp_t color_stamp;
    tags_t tags;

    static port_t const port_src_prefix;
    static port_t const port_out_prefix;
};

process::port_t const source_process::priv::port_src_prefix = process::port_t("src/");
process::port_t const source_process::priv::port_out_prefix = process::port_t("out/");

source_process
::source_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  ensure_inputs_are_same_color(false);
  ensure_inputs_are_valid(false);
}

source_process
::~source_process()
{
}

void
source_process
::_init()
{
  if (!d->tags.size())
  {
    std::string const reason = "There must be at least one source data port";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_init();
}

void
source_process
::_step()
{
  std::map<port_t, edge_datum_t> data;
  bool complete = false;

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    edge_datum_t const edat = grab_from_port(priv::port_src_prefix + tag);
    datum_t const& dat = edat.get<0>();
    stamp_t const& src_stamp = edat.get<1>();

    if (dat->type() == datum::complete)
    {
      complete = true;
    }

    stamp_t const recolored_stamp = stamp::recolored_stamp(src_stamp, d->color_stamp);

    edge_datum_t const edat_out = edge_datum_t(dat, recolored_stamp);

    data[tag] = edat_out;
  }

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    edge_datum_t edat;
    edge_datum_t const edat_in = data[tag];

    if (complete)
    {
      stamp_t const& st = edat_in.get<1>();

      edat = edge_datum_t(datum::complete_datum(), st);
    }
    else
    {
      edat = edat_in;
    }

    push_to_port(priv::port_out_prefix + tag, edat);
  }

  if (complete)
  {
    mark_process_as_complete();
  }

  process::_step();
}

process::port_info_t
source_process
::_input_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_src_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_src_prefix.size());

    priv::tags_t::const_iterator const i = std::find(d->tags.begin(), d->tags.end(), tag);

    if (i == d->tags.end())
    {
      d->tags.push_back(tag);

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(port, boost::make_shared<port_info>(
        type_flow_dependent + tag,
        required,
        port_description_t("The original data stream for " + tag + ".")));
      declare_output_port(priv::port_out_prefix + tag, boost::make_shared<port_info>(
        type_flow_dependent + tag,
        required,
        port_description_t("The recolored data stream for " + tag + ".")));
    }
  }

  return process::_input_port_info(port);
}

source_process::priv
::priv()
  : color_stamp(stamp::new_stamp())
{
}

source_process::priv
::~priv()
{
}

}
