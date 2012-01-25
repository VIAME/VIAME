/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "homography_reader_process.h"

#include <vistk/utilities/homography.h>

#include <vistk/pipeline_types/utility_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/filesystem/path.hpp>
#include <boost/make_shared.hpp>

#include <vnl/vnl_double_3x3.h>

#include <fstream>
#include <string>

/**
 * \file homography_reader_process.cxx
 *
 * \brief Implementation of the homography reader process.
 */

namespace vistk
{

namespace
{

typedef boost::filesystem::path path_t;

}

class homography_reader_process::priv
{
  public:
    priv(path_t const& input_path);
    ~priv();

    path_t const path;
    bool read_error;

    std::ifstream fin;

    bool has_color;

    stamp_t output_stamp;

    static config::key_t const config_path;
    static port_t const port_color;
    static port_t const port_output;
};

config::key_t const homography_reader_process::priv::config_path = config::key_t("input");
process::port_t const homography_reader_process::priv::port_color = process::port_t("color");
process::port_t const homography_reader_process::priv::port_output = process::port_t("homography");

homography_reader_process
::homography_reader_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The input file with homographies to read.")));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_color, boost::make_shared<port_info>(
    type_none,
    port_flags_t(),
    port_description_t("If connected, uses the stamp's color for the output.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    utility_types::t_transform,
    required,
    port_description_t("The homographies that are read in.")));
}

homography_reader_process
::~homography_reader_process()
{
}

void
homography_reader_process
::_init()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  boost::filesystem::path::string_type const path = d->path.native();

  if (path.empty())
  {
    config::value_t const file_path = config::value_t(path.begin(), path.end());
    static std::string const reason = "The path given was empty";

    throw invalid_configuration_value_exception(name(), priv::config_path, file_path, reason);
  }

  d->fin.open(path.c_str());

  if (!d->fin.good())
  {
    std::string const file_path(path.begin(), path.end());
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  if (input_port_edge(priv::port_color))
  {
    d->has_color = true;
  }

  d->output_stamp = heartbeat_stamp();
}

void
homography_reader_process
::_step()
{
  datum_t dat;
  bool complete = false;

  if (d->fin.eof())
  {
    complete = true;
  }
  else if (!d->fin.good())
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream.");

    dat = datum::error_datum(err_string);
  }
  else
  {
    typedef vnl_matrix_fixed<double, 3, 3> matrix_t;

    matrix_t read_mat;

    for (size_t i = 0; i < 9; ++i)
    {
      std::istream const& istr = d->fin >> read_mat(i / 3, i % 3);

      if (!istr)
      {
        d->read_error = true;
        break;
      }

      if (d->fin.eof())
      {
        complete = true;
      }
    }

    homography_base::transform_t const mat(read_mat);

    dat = datum::new_datum(mat);
  }

  if (d->read_error)
  {
    static datum::error_t const err_string = datum::error_t("Error reading from the input file.");

    dat = datum::error_datum(err_string);
  }

  d->output_stamp = stamp::incremented_stamp(d->output_stamp);

  if (d->has_color)
  {
    edge_datum_t const color_dat = grab_from_port(priv::port_color);

    switch (color_dat.get<0>()->type())
    {
      case datum::complete:
        complete = true;
      case datum::data:
      case datum::empty:
        break;
      case datum::error:
      {
        static datum::error_t const err_string = datum::error_t("Error on the color edge.");

        dat = datum::error_datum(err_string);
      }
      case datum::invalid:
      default:
      {
        static datum::error_t const err_string = datum::error_t("Unrecognized datum type on the color edge.");

        dat = datum::error_datum(err_string);
      }
    }

    d->output_stamp = stamp::recolored_stamp(d->output_stamp, color_dat.get<1>());
  }

  if (complete)
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }

  edge_datum_t const edat = edge_datum_t(dat, d->output_stamp);

  push_to_port(priv::port_output, edat);

  process::_step();
}

homography_reader_process::priv
::priv(path_t const& input_path)
  : path(input_path)
  , read_error(false)
  , has_color(false)
{
}

homography_reader_process::priv
::~priv()
{
}

}
