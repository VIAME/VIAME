// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "config_validation.h"

#include <kwiversys/SystemTools.hxx>

#include <fstream>

namespace kwiver {
namespace tools {

namespace kv = ::kwiver::vital;
typedef kwiversys::SystemTools ST;

//=============================================================================
bool
validate_required_input_file(std::string const& name,
                             kv::config_block const& config,
                             kv::logger_handle_t logger)
{
  if (!config.has_value(name) ||
      config.get_value<std::string>(name) == "")
  {
    LOG_ERROR(logger, "Configuration value for " << name
                      << " is missing but required");
    return false;
  }

  return validate_optional_input_file(name, config, logger);
}

//=============================================================================
bool
validate_optional_input_file(std::string const& name,
                             kv::config_block const& config,
                             kv::logger_handle_t logger)
{
  if (config.has_value(name) &&
      config.get_value<std::string>(name) != "")
  {
    std::string path = config.get_value<std::string>(name);
    if (!ST::FileExists(path, true))
    {
      LOG_ERROR(logger, name << " path, " << path
                        << ", does not exist or is not a regular file");
      return false;
    }
  }
  return true;
}

//=============================================================================
bool
validate_required_output_file(std::string const& name,
                              kv::config_block const& config,
                              kv::logger_handle_t logger,
                              bool make_directory,
                              bool test_write)
{
  if (!config.has_value(name) ||
      config.get_value<std::string>(name) == "")
  {
    LOG_ERROR(logger, "Configuration value for " << name
                      << " is missing but required");
    return false;
  }

  return validate_optional_output_file(name, config, logger,
                                       make_directory, test_write);
}

//=============================================================================
bool
validate_optional_output_file(std::string const& name,
                              kv::config_block const& config,
                              kv::logger_handle_t logger,
                              bool make_directory,
                              bool test_write)
{
  if (config.has_value(name) &&
      config.get_value<std::string>(name) != "")
  {
    auto path = config.get_value<std::string>(name);
    auto parent_dir = ST::GetFilenamePath(ST::CollapseFullPath(path));
    if (!ST::FileIsDirectory(parent_dir))
    {
      if (make_directory)
      {
        if (!ST::MakeDirectory(parent_dir))
        {
          LOG_ERROR(logger, "unable to create directory " << parent_dir
                            << " for configuration option " << name);
          return false;
        }
      }
      else
      {
        LOG_ERROR(logger, "directory " << parent_dir << " does not exist"
                          << " for configuration option " << name);
        return false;
      }
    }

    if (test_write)
    {
      std::ofstream ofs(path.c_str());
      if (!ofs)
      {
        LOG_ERROR(logger, "Could not open file " << path << " for writing "
                          << "as required by configuration option " << name);
        return false;
      }
    }
  }
  return true;
}

//=============================================================================
bool
validate_required_input_dir(std::string const& name,
                            kv::config_block const& config,
                            kv::logger_handle_t logger)
{
  // validation for input directories is the same as output directories except
  // that we never create missing directories for inputs
  return validate_required_output_dir(name, config, logger, false);
}

//=============================================================================
bool
validate_optional_input_dir(std::string const& name,
                            kv::config_block const& config,
                            kv::logger_handle_t logger)
{
  // validation for input directories is the same as output directories except
  // that we never create missing directories for inputs
  return validate_optional_output_dir(name, config, logger, false);
}

//=============================================================================
bool
validate_required_output_dir(std::string const& name,
                             kv::config_block const& config,
                             kv::logger_handle_t logger,
                             bool make_directory)
{
  if (!config.has_value(name) ||
      config.get_value<std::string>(name) == "")
  {
    LOG_ERROR(logger, "Configuration value for " << name
                      << " is missing but required");
    return false;
  }
  return validate_optional_output_dir(name, config, logger, make_directory);
}

//=============================================================================
bool
validate_optional_output_dir(std::string const& name,
                             kv::config_block const& config,
                             kv::logger_handle_t logger,
                             bool make_directory)
{
  if (config.has_value(name) &&
      config.get_value<std::string>(name) != "")
  {
    std::string path = config.get_value<std::string>(name);
    if (!ST::FileIsDirectory(path))
    {
      if (ST::FileExists(path))
      {
        LOG_ERROR(logger, name << " is set to " << path
                          << " and is a file, not a valid directory");
        return false;
      }
      if (make_directory)
      {
        if (!ST::MakeDirectory(path))
        {
          LOG_ERROR(logger, "unable to create directory " << path
                            << " for configuration option " << name);
          return false;
        }
        return true;
      }
      LOG_ERROR(logger, path << " is not a valid directory for "
                        << "configuration option " << name);
      return false;
    }
  }
  return true;
}

} // end namespace tools
} // end namespace kwiver
