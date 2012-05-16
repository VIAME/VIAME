/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "mask_scoring_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/pixtypes.h>

#include <vistk/pipeline/datum.h>

#include <vistk/scoring/score_mask.h>
#include <vistk/scoring/scoring_result.h>

#include <boost/make_shared.hpp>

/**
 * \file mask_scoring_process.cxx
 *
 * \brief Implementation of the image reader process.
 */

namespace vistk
{

class mask_scoring_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_computed_mask;
    static port_t const port_truth_mask;
    static port_t const port_result;
};

process::port_t const mask_scoring_process::priv::port_computed_mask = process::port_t("computed_mask");
process::port_t const mask_scoring_process::priv::port_truth_mask = process::port_t("truth_mask");
process::port_t const mask_scoring_process::priv::port_result = process::port_t("result");

mask_scoring_process
::mask_scoring_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t required;

  required.insert(flag_required);

  port_type_t const mask_port_type = port_type_for_pixtype(pixtypes::pixtype_byte(), pixfmts::pixfmt_mask());

  declare_input_port(priv::port_computed_mask, boost::make_shared<port_info>(
    mask_port_type,
    required,
    port_description_t("The computed mask.")));
  declare_input_port(priv::port_truth_mask, boost::make_shared<port_info>(
    mask_port_type,
    required,
    port_description_t("The truth mask.")));

  declare_output_port(priv::port_result, boost::make_shared<port_info>(
    "score",
    required,
    port_description_t("The result of the scoring.")));
}

mask_scoring_process
::~mask_scoring_process()
{
}

void
mask_scoring_process
::_step()
{
  mask_t const truth = grab_from_port_as<mask_t>(priv::port_truth_mask);
  mask_t const computed = grab_from_port_as<mask_t>(priv::port_computed_mask);

  scoring_result_t const score = score_mask(truth, computed);

  if (score)
  {
    push_to_port_as(priv::port_result, score);
  }
  else
  {
    static datum::error_t const reason = datum::error_t("The scoring failed");
    datum_t const err = datum::error_datum(reason);

    push_datum_to_port(priv::port_result, err);
  }

  process::_step();
}

mask_scoring_process::priv
::priv()
{
}

mask_scoring_process::priv
::~priv()
{
}

}
