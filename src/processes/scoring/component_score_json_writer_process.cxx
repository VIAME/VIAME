/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include <vistk/version.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <string>

#include <cmath>

/**
 * \file component_score_json_writer_process.cxx
 *
 * \brief Implementation of the JSON component score writer process.
 */

using namespace boost::local_time;

#if defined(_WIN32) || defined(_WIN64)
namespace std
{

static bool isnan(double v);

}
#endif

namespace vistk
{

namespace
{

typedef char char_type;

// Use an ISO-formatted date.
static char_type const* const date_format = boost::date_time::time_formats<char_type>::iso_time_format_specifier;

}

class component_score_json_writer_process::priv
{
  public:
    typedef port_t tag_t;
    typedef std::vector<tag_t> tags_t;

    typedef std::map<tag_t, bool> tag_stat_map_t;

    typedef std::string name_t;

    priv();
    priv(name_t const& name_, path_t const& output_path, tags_t const& tags_);
    ~priv();

    void initialize_date();

    name_t const name;
    path_t const path;
    local_date_time const ldt;

    boost::filesystem::ofstream fout;

    tags_t tags;
    tag_stat_map_t tag_stats;

    static config::key_t const config_path;
    static config::key_t const config_name;
    static config::value_t const default_name;
    static port_t const port_score_prefix;
    static port_t const port_stats_prefix;
};

config::key_t const component_score_json_writer_process::priv::config_path = "path";
config::key_t const component_score_json_writer_process::priv::config_name = "name";
config::value_t const component_score_json_writer_process::priv::default_name = "(unnamed)";
process::port_t const component_score_json_writer_process::priv::port_score_prefix = port_t("score/");
process::port_t const component_score_json_writer_process::priv::port_stats_prefix = port_t("stats/");

component_score_json_writer_process
::component_score_json_writer_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  declare_configuration_key(
    priv::config_path,
    config::value_t(),
    config::description_t("The path to output scores to."));
  declare_configuration_key(
    priv::config_name,
    priv::default_name,
    config::description_t("The name of the results."));
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
    priv::name_t const run_name = config_value<priv::name_t>(priv::config_name);
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(run_name, path, d->tags));
  }

  if (d->path.empty())
  {
    static std::string const reason = "The path given was empty";
    config::value_t const value = d->path.string<config::value_t>();

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fout.open(d->path);

  if (!d->fout.good())
  {
    std::string const str = d->path.string<std::string>();
    std::string const reason = "Failed to open the path: " + str;

    throw invalid_configuration_exception(name(), reason);
  }

  d->initialize_date();

  process::_configure();
}

void
component_score_json_writer_process
::_init()
{
  if (d->tags.empty())
  {
    static std::string const reason = "There must be at least one component score to write";

    throw invalid_configuration_exception(name(), reason);
  }

  BOOST_FOREACH (priv::tag_t const& tag, d->tags)
  {
    port_t const port_stat = priv::port_stats_prefix + tag;

    d->tag_stats[tag] = false;

    if (has_input_port_edge(port_stat))
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
#define JSON_ATTR_DOUBLE(key, value) \
  JSON_ATTR(key, (std::isnan(value) ? "\"nan\"" : boost::lexical_cast<std::string>(value)))
#define JSON_SEP \
  "," << std::endl
#define JSON_OBJECT_BEGIN \
  "{" << std::endl
#define JSON_OBJECT_END \
  "}"

  d->fout << JSON_OBJECT_BEGIN;

  d->fout << JSON_ATTR("name", "\"" + d->name + "\"");
  d->fout << JSON_SEP;
  d->fout << JSON_ATTR("date", "\"" << d->ldt << "\"");
  d->fout << JSON_SEP;
  d->fout << JSON_ATTR("hash", "\"" VISTK_GIT_VERSION "\"");
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
    d->fout << JSON_ATTR_DOUBLE("percent-detection", result->percent_detection());
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR_DOUBLE("precision", result->precision());
    d->fout << JSON_SEP;
    d->fout << JSON_ATTR_DOUBLE("specificity", result->specificity());

    if (d->tag_stats[tag])
    {
      port_t const port_stats = priv::port_stats_prefix + tag;

      d->fout << JSON_SEP;

#define OUTPUT_STATISTICS(key, stats)                                                 \
  do                                                                                  \
  {                                                                                   \
    if (stats->count())                                                               \
    {                                                                                 \
      d->fout << JSON_KEY(key);                                                       \
      d->fout << JSON_OBJECT_BEGIN;                                                   \
      d->fout << JSON_ATTR("count", stats->count());                                  \
      d->fout << JSON_SEP;                                                            \
      d->fout << JSON_ATTR("min", stats->minimum());                                  \
      d->fout << JSON_SEP;                                                            \
      d->fout << JSON_ATTR("max", stats->maximum());                                  \
      d->fout << JSON_SEP;                                                            \
      d->fout << JSON_ATTR_DOUBLE("mean", stats->mean());                             \
      d->fout << JSON_SEP;                                                            \
      d->fout << JSON_ATTR_DOUBLE("median", stats->median());                         \
      d->fout << JSON_SEP;                                                            \
      d->fout << JSON_ATTR_DOUBLE("standard-deviation", stats->standard_deviation()); \
      d->fout << JSON_OBJECT_END;                                                     \
    }                                                                                 \
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

    if (!std::count(d->tags.begin(), d->tags.end(), tag))
    {
      port_t const port_score = priv::port_score_prefix + tag;
      port_t const port_stats = priv::port_stats_prefix + tag;

      d->tags.push_back(tag);

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(
        port_score,
        "score",
        required,
        port_description_t("The \'" + tag + "\' score component."));
      declare_input_port(
        port_stats,
        "statistics/score",
        port_flags_t(),
        port_description_t("The \'" + tag + "\' score statistics component."));
    }
  }

  return process::_input_port_info(port);
}

component_score_json_writer_process::priv
::priv()
  : name()
  , path()
  , ldt(local_microsec_clock::local_time(time_zone_ptr()))
  , fout()
  , tags()
  , tag_stats()
{
}

component_score_json_writer_process::priv
::priv(name_t const& name_, path_t const& output_path, tags_t const& tags_)
  : name(name_)
  , path(output_path)
  , ldt(local_microsec_clock::local_time(time_zone_ptr()))
  , fout()
  , tags(tags_)
  , tag_stats()
{
}

component_score_json_writer_process::priv
::~priv()
{
}

void
component_score_json_writer_process::priv
::initialize_date()
{
  local_time_facet* facet = new local_time_facet;

  facet->format(date_format);
  fout.imbue(std::locale(fout.getloc(), facet));
}

}

#if defined(_WIN32) || defined(_WIN64)
namespace std
{

bool
isnan(double v)
{
  return (v != v);
}

}
#endif
