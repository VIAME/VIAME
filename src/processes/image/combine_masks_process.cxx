/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "combine_masks_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/operators.h>
#include <processes/helpers/image/pixtypes.h>

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <numeric>
#include <string>

/**
 * \file combine_masks_process.cxx
 *
 * \brief Implementation of the mask combination process.
 */

namespace vistk
{

class combine_masks_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t tag_t;
    typedef std::vector<tag_t> tags_t;

    binop_func_t const combine;
    port_type_t const port_type;

    tags_t tags;

    static port_t const port_mask_prefix;
    static port_t const port_mask;
};

process::port_t const combine_masks_process::priv::port_mask_prefix = process::port_t("mask/");
process::port_t const combine_masks_process::priv::port_mask = process::port_t("mask");

combine_masks_process
::combine_masks_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  ensure_inputs_are_valid(false);

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(priv::port_mask, boost::make_shared<port_info>(
    d->port_type,
    required,
    port_description_t("The combined mask.")));
}

combine_masks_process
::~combine_masks_process()
{
}

void
combine_masks_process
::_configure()
{
  if (!d->combine)
  {
    static std::string const reason = "A combine function for the "
                                      "mask pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
combine_masks_process
::_init()
{
  if (!d->tags.size())
  {
    static std::string const reason = "There must be at least one mask to combine";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_init();
}

void
combine_masks_process
::_step()
{
  std::vector<datum_t> data;
  bool flush = false;
  bool complete = false;

  datum_t dat;

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    datum_t const idat = grab_datum_from_port(priv::port_mask_prefix + tag);

    switch (idat->type())
    {
      case datum::data:
        /// \todo Check image sizes.

        data.push_back(idat);
        break;
      case datum::flush:
        flush = true;
        break;
      case datum::complete:
        complete = true;
        break;
      case datum::invalid:
      case datum::error:
      {
        datum::error_t const err_string = datum::error_t("Error on input tag \'" + tag + "\'");

        dat = datum::error_datum(err_string);
        break;
      }
      case datum::empty:
      default:
        break;
    }
  }

  if (complete)
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }

  if (flush)
  {
    dat = datum::flush_datum();
  }

  if (!dat)
  {
    dat = std::accumulate(data.begin(), data.end(), datum_t(), d->combine);
  }

  if (!dat)
  {
    dat = datum::empty_datum();
  }

  push_datum_to_port(priv::port_mask, dat);

  process::_step();
}

process::port_info_t
combine_masks_process
::_input_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_mask_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_mask_prefix.size());

    priv::tags_t::const_iterator const i = std::find(d->tags.begin(), d->tags.end(), tag);

    if (i == d->tags.end())
    {
      d->tags.push_back(tag);

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(port, boost::make_shared<port_info>(
        d->port_type,
        required,
        port_description_t("The \'" + tag + "\' mask.")));
    }
  }

  return process::_input_port_info(port);
}

combine_masks_process::priv
::priv()
  : combine(or_for_pixtype(pixtypes::pixtype_byte()))
  , port_type(port_type_for_pixtype(pixtypes::pixtype_byte(), pixfmts::pixfmt_mask()))
{
}

combine_masks_process::priv
::~priv()
{
}

}
