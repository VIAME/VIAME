/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "component_score_json_writer_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <vistk/scoring/scoring_result.h>
#include <vistk/scoring/scoring_statistics.h>
#include <vistk/scoring/statistics.h>

#include <vistk/utilities/path.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
#include <string>

/**
 * \file component_score_json_writer_process.cxx
 *
 * \brief Implementation of a process which writes out component scores to a file in JSON.
 */

namespace vistk
{

class component_score_json_writer_process::priv
{
  public:
    typedef port_t tag_t;
    typedef std::vector<tag_t> tags_t;

    typedef std::map<tag_t, bool> tag_stat_map_t;

    priv();
    priv(path_t const& output_path, tags_t const& tags_);
    ~priv();

    path_t const path;

    std::ofstream fout;

    tags_t tags;
    tag_stat_map_t tag_stats;

    static config::key_t const config_path;
    static port_t const port_score_prefix;
    static port_t const port_stats_prefix;
};

config::key_t const component_score_json_writer_process::priv::config_path = "path";
process::port_t const component_score_json_writer_process::priv::port_score_prefix = process::port_t("score/");
process::port_t const component_score_json_writer_process::priv::port_stats_prefix = process::port_t("stats/");

component_score_json_writer_process
::component_score_json_writer_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The path to output scores to.")));
}

component_score_json_writer_process
::~component_score_json_writer_process()
{
}

void
component_score_json_writer_process
::_configure()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path, d->tags));
  }

  path_t::string_type const path = d->path.native();

  if (path.empty())
  {
    static std::string const reason = "The path given was empty";
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fout.open(path.c_str());

  if (!d->fout.good())
  {
    std::string const file_path(path.begin(), path.end());
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
component_score_json_writer_process
::_init()
{
  if (!d->tags.size())
  {
    static std::string const reason = "There must be at least one component score to write";

    throw invalid_configuration_exception(name(), reason);
  }

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    port_t const port_stat = priv::port_stats_prefix + tag;

    d->tag_stats[tag] = false;

    if (input_port_edge(port_stat))
    {
      d->tag_stats[tag] = true;
    }
  }

  process::_init();
}

void
component_score_json_writer_process
::_step()
{
#define JSON_KEY(key) \
  ("\"" key "\": ")
#define JSON_ATTR(key, value) \
  JSON_KEY(key) << value
#define JSON_SEP \
  "," << std::endl
#define JSON_OBJECT_BEGIN \
  "{" << std::endl
#define JSON_OBJECT_END \
  "}"

  d->fout << JSON_OBJECT_BEGIN;

  /// \todo Name runs.
  d->fout << JSON_ATTR("name", "\"(unnamed)\"");
  d->fout << JSON_SEP;
  /// \todo Insert date.
  d->fout << JSON_ATTR("date", "\"\"");
  d->fout << JSON_SEP;
  /// \todo Get git hash.
  d->fout << JSON_ATTR("hash", "\"\"");
  d->fout << JSON_SEP;
  d->fout << JSON_ATTR("config", "{}");
  d->fout << JSON_SEP;
  d->fout << JSON_KEY("results") << JSON_OBJECT_BEGIN;

  bool first = true;

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    port_t const port_score = priv::port_score_prefix + tag;

    if (!first)
    {
      d->fout << JSON_SEP;
    }

    first = false;

    d->fout << JSON_KEY(+ tag +);

    d->fout << JSON_OBJECT_BEGIN;

    scoring_result_t const result = grab_from_port_as<scoring_result_t>(port_score);

    d->fout << JSON_ATTR("true-positive", result->true_positives);
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("false-positive", result->false_positives);
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("total-true", result->total_trues);
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("total-possible", result->total_possible);
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("percent-detection", result->percent_detection());
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("precision", result->precision());
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR("specificity", result->specificity());

    if (d->tag_stats[tag])
    {
      port_t const port_stats = priv::port_stats_prefix + tag;

      d->fout << JSON_SEP;

#define OUTPUT_STATISTICS(key, stats)                                        \
  do                                                                         \
  {                                                                          \
    d->fout << JSON_KEY(key);                                                \
    d->fout << JSON_OBJECT_BEGIN;                                            \
    d->fout << JSON_ATTR("count", stats->count());                           \
    d->fout << JSON_SEP;                                                     \
    d->fout << JSON_ATTR("min", stats->minimum());                           \
    d->fout << JSON_SEP;                                                     \
    d->fout << JSON_ATTR("max", stats->maximum());                           \
    d->fout << JSON_SEP;                                                     \
    d->fout << JSON_ATTR("mean", stats->mean());                             \
    d->fout << JSON_SEP;                                                     \
    d->fout << JSON_ATTR("median", stats->median());                         \
    d->fout << JSON_SEP;                                                     \
    d->fout << JSON_ATTR("standard-deviation", stats->standard_deviation()); \
    d->fout << JSON_OBJECT_END;                                              \
  } while (false)

      scoring_statistics_t const sc_stats = grab_from_port_as<scoring_statistics_t>(port_stats);

      statistics_t const pd_stats = sc_stats->percent_detection_stats();
      statistics_t const precision_stats = sc_stats->precision_stats();
      statistics_t const specificity_stats = sc_stats->specificity_stats();

      d->fout << JSON_KEY("statistics");
      d->fout << JSON_OBJECT_BEGIN;
      OUTPUT_STATISTICS("percent-detection", pd_stats);
      d->fout << JSON_SEP;
      OUTPUT_STATISTICS("precision", precision_stats);
      d->fout << JSON_SEP;
      OUTPUT_STATISTICS("specificity", specificity_stats);
      d->fout << JSON_OBJECT_END;

#undef OUTPUT_STATISTICS
    }

    d->fout << JSON_OBJECT_END;
  }

  d->fout << std::endl;
  d->fout << JSON_OBJECT_END;
  d->fout << std::endl;
  d->fout << JSON_OBJECT_END;
  d->fout << std::endl;

#undef JSON_OBJECT_END
#undef JSON_OBJECT_BEGIN
#undef JSON_SEP
#undef JSON_ATTR
#undef JSON_KEY

  process::_step();
}

process::port_info_t
component_score_json_writer_process
::_input_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_score_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_score_prefix.size());

    priv::tags_t::const_iterator const i = std::find(d->tags.begin(), d->tags.end(), tag);

    if (i == d->tags.end())
    {
      port_t const port_score = priv::port_score_prefix + tag;
      port_t const port_stats = priv::port_stats_prefix + tag;

      d->tags.push_back(tag);

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(port_score, boost::make_shared<port_info>(
        "score",
        required,
        port_description_t("The \'" + tag + "\' score component.")));
      declare_input_port(port_stats, boost::make_shared<port_info>(
        "statistics/score",
        port_flags_t(),
        port_description_t("The \'" + tag + "\' score statistics component.")));
    }
  }

  return process::_input_port_info(port);
}

component_score_json_writer_process::priv
::priv()
{
}

component_score_json_writer_process::priv
::priv(path_t const& output_path, tags_t const& tags_)
  : path(output_path)
  , tags(tags_)
{
}

component_score_json_writer_process::priv
::~priv()
{
}

}
