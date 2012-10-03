/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "score_writer_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_exception.h>

#include <vistk/scoring/scoring_result.h>

#include <vistk/utilities/path.h>

#include <boost/filesystem/fstream.hpp>

/**
 * \file score_writer_process.cxx
 *
 * \brief Implementation of the score writer process.
 */

namespace vistk
{

class score_writer_process::priv
{
  public:
    priv(path_t const& output_path);
    ~priv();

    path_t const path;

    boost::filesystem::ofstream fout;

    static config::key_t const config_path;
    static port_t const port_score;
};

config::key_t const score_writer_process::priv::config_path = config::key_t("path");
process::port_t const score_writer_process::priv::port_score = port_t("score");

score_writer_process
::score_writer_process(config_t const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_path,
    config::key_t(),
    config::description_t("The path to write results to."));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_score,
    "score",
    required,
    port_description_t("The scores to write."));

}

score_writer_process
::~score_writer_process()
{
}

void
score_writer_process
::_configure()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  if (d->path.empty())
  {
    static std::string const reason = "The path given was empty";
    path_t::string_type const& path = d->path.native();
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fout.open(d->path);

  if (!d->fout.good())
  {
    path_t::string_type const& path = d->path.native();
    std::string const file_path(path.begin(), path.end());
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
score_writer_process
::_reset()
{
  d->fout.close();

  process::_reset();
}

void
score_writer_process
::_step()
{
  scoring_result_t const result = grab_from_port_as<scoring_result_t>(priv::port_score);

  d->fout << result->true_positives << " "
          << result->false_positives << " "
          << result->total_trues << " "
          << result->total_possible << " "
          << result->percent_detection() << " "
          << result->precision() << " "
          << result->specificity() << std::endl;

  process::_step();
}

score_writer_process::priv
::priv(path_t const& output_path)
  : path(output_path)
  , fout()
{
}

score_writer_process::priv
::~priv()
{
}

}
