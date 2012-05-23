/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "score_aggregation_process.h"

#include <vistk/pipeline/datum.h>

#include <vistk/scoring/scoring_result.h>
#include <vistk/scoring/scoring_statistics.h>

#include <boost/make_shared.hpp>

#include <numeric>

/**
 * \file score_aggregation_process.cxx
 *
 * \brief Implementation of the score aggregation process.
 */

namespace vistk
{

class score_aggregation_process::priv
{
  public:
    priv();
    ~priv();

    void reset();

    scoring_results_t results;
    scoring_statistics_t statistics;

    static port_t const port_score;
    static port_t const port_aggregate;
    static port_t const port_statistics;
};

process::port_t const score_aggregation_process::priv::port_score = process::port_t("score");
process::port_t const score_aggregation_process::priv::port_aggregate = process::port_t("aggregate");
process::port_t const score_aggregation_process::priv::port_statistics = process::port_t("statistics");

score_aggregation_process
::score_aggregation_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  // We only calculate on 'complete' datum.
  ensure_inputs_are_valid(false);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_score, boost::make_shared<port_info>(
    "score",
    required,
    port_description_t("The scores to aggregate.")));

  declare_output_port(priv::port_aggregate, boost::make_shared<port_info>(
    "score",
    required,
    port_description_t("The aggregate scores.")));
  declare_output_port(priv::port_statistics, boost::make_shared<port_info>(
    "statistics/score",
    port_flags_t(),
    port_description_t("Statistics on the aggregate scores.")));
}

score_aggregation_process
::~score_aggregation_process()
{
}

void
score_aggregation_process
::_step()
{
  datum_t const dat = grab_datum_from_port(priv::port_score);

  bool complete = false;

  switch (dat->type())
  {
    case datum::complete:
      complete = true;
    case datum::flush:
    {
      scoring_result_t const base = boost::make_shared<scoring_result>(0, 0, 0, 0);
      scoring_result_t const overall = std::accumulate(d->results.begin(), d->results.end(), base);

      push_to_port_as<scoring_result_t>(priv::port_aggregate, overall);
      push_to_port_as<scoring_statistics_t>(priv::port_aggregate, d->statistics);

      d->reset();

      break;
    }
    case datum::data:
    {
      scoring_result_t const result = dat->get_datum<scoring_result_t>();

      d->results.push_back(result);
      d->statistics->add_score(result);

      break;
    }
    case datum::invalid:
    case datum::empty:
    case datum::error:
    default:
      break;
  }

  if (complete)
  {
    push_datum_to_port(priv::port_aggregate, datum::complete_datum());

    mark_process_as_complete();
  }

  process::_step();
}

score_aggregation_process::priv
::priv()
{
  reset();
}

score_aggregation_process::priv
::~priv()
{
}

void
score_aggregation_process::priv
::reset()
{
  results.clear();
  statistics = scoring_statistics_t(new scoring_statistics);
}

}
