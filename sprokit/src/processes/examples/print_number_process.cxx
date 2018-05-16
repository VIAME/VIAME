/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "print_number_process.h"

#include <sprokit/pipeline_util/path.h>

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process_exception.h>

#include <boost/filesystem/fstream.hpp>

#include <string>

/**
 * \file print_number_process.cxx
 *
 * \brief Implementation of the number printer process.
 */

namespace sprokit
{

class print_number_process::priv
{
  public:
    typedef int32_t number_t;

    priv(path_t const& output_path);
    ~priv();

    path_t const path;

    boost::filesystem::ofstream fout;

    static kwiver::vital::config_block_key_t const config_path;
    static port_t const port_input;
};

kwiver::vital::config_block_key_t const print_number_process::priv::config_path = kwiver::vital::config_block_key_t("output");
process::port_t const print_number_process::priv::port_input = port_t("number");

print_number_process
::print_number_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_path,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The path of the file to output to."));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    "integer",
    required,
    port_description_t("Where numbers are read from."));
}

print_number_process
::~print_number_process()
{
}

void
print_number_process
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
    kwiver::vital::config_block_value_t const value = d->path.string<kwiver::vital::config_block_value_t>();

    VITAL_THROW( invalid_configuration_value_exception,
                 name(), priv::config_path, value, reason);
  }

  d->fout.open(d->path);

  if (!d->fout.good())
  {
    std::string const file_path = d->path.string<std::string>();
    std::string const reason = "Failed to open the path: " + file_path;

    VITAL_THROW( invalid_configuration_exception,
                 name(), reason);
  }

  process::_configure();
}

void
print_number_process
::_reset()
{
  d->fout.close();

  process::_reset();
}

void
print_number_process
::_step()
{
  priv::number_t const input = grab_from_port_as<priv::number_t>(priv::port_input);

  d->fout << input << std::endl;

  process::_step();
}

print_number_process::priv
::priv(path_t const& output_path)
  : path(output_path)
  , fout()
{
}

print_number_process::priv
::~priv()
{
}

}
