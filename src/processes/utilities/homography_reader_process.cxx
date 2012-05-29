/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "homography_reader_process.h"

#include <vistk/utilities/homography.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

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

class homography_reader_process::priv
{
  public:
    priv(path_t const& input_path);
    ~priv();

    path_t const path;

    std::ifstream fin;

    static config::key_t const config_path;
    static port_t const port_output;
};

config::key_t const homography_reader_process::priv::config_path = config::key_t("input");
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

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    "transform",
    required,
    port_description_t("The homographies that are read in.")));
}

homography_reader_process
::~homography_reader_process()
{
}

void
homography_reader_process
::_configure()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  path_t::string_type const path = d->path.native();

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

  process::_configure();
}

void
homography_reader_process
::_step()
{
  datum_t dat;
  bool read_error = false;
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

    for (size_t i = 0; !read_error && (i < 9); ++i)
    {
      std::istream const& istr = d->fin >> read_mat(i / 3, i % 3);

      if (!istr)
      {
        read_error = true;
      }

      if (d->fin.eof())
      {
        complete = true;
        break;
      }
    }

    homography_base::transform_t const mat(read_mat);

    dat = datum::new_datum(mat);
  }

  if (read_error)
  {
    static datum::error_t const err_string = datum::error_t("Error reading from the input file.");

    dat = datum::error_datum(err_string);
  }

  if (complete)
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

homography_reader_process::priv
::priv(path_t const& input_path)
  : path(input_path)
{
}

homography_reader_process::priv
::~priv()
{
}

}
