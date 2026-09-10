// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_block_formatter.h>
#include <viame/algorithm_framework/config/config_block_io.h>
#include <viame/algorithm_framework/config/config_difference.h>
#include <viame/core_types/geo_polygon.h>

#include "module_config_helpers.h"

#include <fstream>

namespace py = pybind11;
namespace kv = kwiver::vital;

PYBIND11_MODULE( _config, m )
{
  py::module::import( "kwiver.vital.types.geo_polygon" );

  m.doc() =
    R"pbdoc(
    Config module for vital
    -----------------------

    .. currentmodule:: config

    .. autosummary::
       :toctree: _generate

    empty_config
    ConfigKeys
    Config
  )pbdoc";

  // ---------------------------------------------------------------------------
  // TODO: Why is this a thing? What is it for?
  //       If we cannot answer this, remove it.
  py::bind_vector< std::vector< std::string > >(
    m, "ConfigKeys",
    R"pbdoc(A collection of keys for a configuration.)pbdoc" );

  // ---------------------------------------------------------------------------
  py::class_< kv::config_block, kv::config_block_sptr >(
    m, "Config",
    R"pbdoc("A key-value store of configuration values)pbdoc" )
    .def(
    "subblock", &kv::config_block::subblock,
    py::arg( "name" )
    // , doc_Config_subblock
    ,
    R"pbdoc(Returns a :class:`kwiver.vital.config.Config` from the
            configuration using the name of the subblock. The object is a copy
            of the block in the configuration.

            :param name: The name of the subblock in a :class:`kwiver.vital.config.Config` object
            :return: a subblock of type :class:`kwiver.vital.config.Config`
            )pbdoc"
    )
    .def(
      "subblock_view", &kv::config_block::subblock_view,
      py::arg( "name" ),
      R"pbdoc(Returns a :class:`kwiver.vital.config.Config`
                from the configuration using the name of the subblock. The object
                is a view rather than the copy of the block in the configuration.
                :param name: The name of the subblock in a :class:`kwiver.vital.config.Config` object
                :return: a subblock of type :class:`kwiver.vital.config.Config`
               )pbdoc" )
    .def(
      "get_value",
      ( kv::config_block_value_t ( kv::config_block::* )(
        kv::config_block_key_t const& ) const ) &
      kv::config_block::get_value< std::string >,
      py::arg( "key" ),
      R"pbdoc(Retrieve a value from the configuration using key.
                    :param key: key in the configuration
                    :return: A string value associated with the key
                   )pbdoc" )
    .def(
      "get_value",
      ( kv::config_block_value_t ( kv::config_block::* )(
        kv::config_block_key_t const&,
        kv::config_block_value_t const& ) const noexcept ) &
      kv::config_block::get_value,
      py::arg( "key" ), py::arg( "default" ),
      R"pbdoc(Retrieve a value from the configuration, using a default in case of failure.
                    :param key: A key in the configuration
                    :param default: A default value for the key
                    :return: A string value associated with the key
                   )pbdoc" )
    .def(
      "get_value_geo_poly",
      ( kv::geo_polygon ( kv::config_block::* )(
        kv::config_block_key_t const& ) const ) &
      kv::config_block::get_value< kv::geo_polygon >,
      py::arg( "key" ),
      R"pbdoc(Retrieve a geo_polygon value from the configuration using key.
                    :param key: key in the configuration
                    :return: A string value associated with the key
                   )pbdoc" )
    .def(
      "get_value_geo_poly",
      ( kv::geo_polygon ( kv::config_block::* )(
        kv::config_block_key_t const&,
        kv::geo_polygon const& ) const ) &
      kv::config_block::get_value< kv::geo_polygon >,
      py::arg( "key" ), py::arg( "default" ),
      R"pbdoc(Retrieve a geo_polygon value from the configuration using key, using a default in case of failure.
                    :param key: key in the configuration
                    :param default: A default geo_polygon value for the key
                    :return: A string value associated with the key
                   )pbdoc" )
    .def(
      "set_value",
      ( void ( kv::config_block::* )(
        kv::config_block_key_t const&,
        kv::config_block_value_t const& ) ) &
      kv::config_block::set_value< kv::config_block_value_t >,
      py::arg( "key" ), py::arg( "value" ),
      R"pbdoc(Set a value in the configuration.
                    :param key: A key in the configuration.
                    :param value: A value in the configuration.
                    :return: None
                   )pbdoc" )
    .def(
      "set_value_geo_poly",
      ( void ( kv::config_block::* )(
        kv::config_block_key_t const&,
        kv::geo_polygon const& ) ) &
      kv::config_block::set_value< kv::geo_polygon >,
      py::arg( "key" ), py::arg( "value" ),
      R"pbdoc(Set a value in the configuration using config_block_set_value_cast<geo_polygon>
                :param key: A key in the configuration.
                :param value: A value in the configuration.
                :return: None
               )pbdoc" )
    .def(
      "unset_value", &kv::config_block::unset_value,
      py::arg( "key" ),
      R"pbdoc(Unset a value in the configuration.
                :param key: A key in the configuration
                :return: None
               )pbdoc" )
    .def(
      "is_read_only", &kv::config_block::is_read_only,
      py::arg( "key" ),
      R"pbdoc(Check if a key is marked as read only.
                :param key: A key in the configuration
                :return: Boolean specifying if the key value pair is read only
               )pbdoc" )
    .def(
      "mark_read_only", &kv::config_block::mark_read_only,
      py::arg( "key" ),
      R"pbdoc(Mark a key as read only.
                :param key: A key in the configuration.
                :return: None
               )pbdoc" )
    .def(
      "merge_config", &kv::config_block::merge_config,
      py::arg( "config" ),
      R"pbdoc(Merge another configuration block into the current one.
                :param config: An object of :class:`vital.config.Config`
                :return: An object of :class:`vital.config.Config` containing the merged configuration
               )pbdoc" )
    .def(
      "available_values", &kv::config_block::available_values,
      R"pbdoc(Retrieves the list of available values in the configuration.
                :return: A list of string with all the keys
               )pbdoc" )
    .def(
      "has_value", &kv::config_block::has_value,
      py::arg( "key" ),
      R"pbdoc(Returns True if the key is set.
                :param key: A key in the configuration
                :return: Boolean specifying if the key is present in the configuration
               )pbdoc" )
    .def_static(
      "block_sep", &kv::config_block::block_sep,
      "The string which separates block names from key names." )
    .def_static(
      "global_value", &kv::config_block::global_value,
      "A special key which is automatically inherited on subblock requests." )
    .def(
      "__len__", &kv::python::config_len,
      R"pbdoc(Magic function that return the length of the configuration block)pbdoc" )
    .def(
      "__contains__", &kv::config_block::has_value,
      R"pbdoc(Magic function to check if an key is in the configuration)pbdoc" )
    .def(
      "__getitem__", &kv::python::config_getitem,
      R"pbdoc(Magic function to get a value)pbdoc" )
    .def(
      "__setitem__", &kv::python::config_setitem,
      R"pbdoc(Magic function to assign a new value to a key)pbdoc" )
    .def(
      "__delitem__", &kv::python::config_delitem,
      R"pbdoc(Magic function to remove a key)pbdoc" )
  ;

  // ---------------------------------------------------------------------------
  m.def(
    "empty_config", &kv::config_block::empty_config,
    R"pbdoc(Returns an empty :class:`kwiver.vital.config.Config` object)pbdoc",
    py::arg( "name" ) = kv::config_block_key_t()
  );

  // ---------------------------------------------------------------------------
  py::class_< kv::config_difference, std::shared_ptr< kv::config_difference > >(
    m, "ConfigDifference",
    "Represents difference between two config blocks" )

    .def(
    py::init< kv::config_block_sptr, kv::config_block_sptr >(),
    py::doc( "Determine difference between config blocks" ) )
    .def(
      py::init< kv::config_block_keys_t, kv::config_block_sptr >(),
      py::doc( "Determine difference between config blocks" ) )

    .def(
      "extra_keys", &kv::config_difference::extra_keys,
      "Return list of config keys that are not in the ref config" )

    .def(
      "unspecified_keys", &kv::config_difference::unspecified_keys,
      "Return list of config keys that are in reference config but not in the other config" )
  ;

  // -------------------------------------------------------------------------
  m.def(
    "read_config_file",
    ( kv::config_block_sptr ( * )(
      kv::config_path_t const&,
      kv::config_path_list_t const&, bool ) ) & kv::read_config_file,
    py::arg( "file_path" ),
    py::arg_v( "search_paths", kv::config_path_list_t(), "[]" ),
    py::arg( "use_system_paths" ) = true,
    // doc-string copied from C++ class
    R"pbdoc(
This method reads the specified config file and returns the
resulting config block. Any files included by config files that are not in
absolute form are resolved using search paths supplied in the environment
variable \c KWIVER_CONFIG_PATH first, and then by using paths supplied in
\c search_paths. If \c no_system_paths is set to \c true, then the contents
of the \c KWIVER_CONFIG_PATH variable is not used, i.e. only the paths given
in \c search_paths are used.

\throws config_file_not_found_exception
   Thrown when the file could not be found on the file system.
\throws config_file_not_read_exception
   Thrown when the file could not be read or parsed for whatever reason.

\param file_path
  The path to the file to read in.
\param search_path
  An optional list of directories to use in locating included files.
\param use_system_paths
  If false, we do not use paths in the KWIVER_CONFIG_PATH environment
  variable or current working directory for searching, otherwise those paths
are
  searched first.

\return A \c config_block object representing the contents of the read-in
  file.
)pbdoc"
  );

  m.def(
    "read_config_file",
    ( kv::config_block_sptr ( * )(
      std::string const&, std::string const&,
      std::string const&, kv::config_path_t const&,
      bool ) ) & kv::read_config_file,
    py::arg( "file_name" ),
    py::arg( "application_name" ),
    py::arg( "application_version" ),
    py::arg( "install_prefix" ) = kv::config_path_t(),
    py::arg( "merge" ) = true,
    // doc-string copied from C++ class
    R"pbdoc(
/**
 * \brief Read in (a) configuration file(s), producing a \c config_block object
 *
 * This function reads one or more configuration files from a search
 * path. The search path is based on environment variables, system
 * defaults, and application defaults. More on this later.
 *
 * The config reader tries to locate the specified config file using
 * the search path. If the file is not found, an exception is
 * thrown. If the file is located and the \c merge parameter is \b
 * true (default value), then the remaining directories in the search
 * path are checked to see if additional versions of the file can be
 * found. If so, then the contents are merged, with values in files earlier in
 * the search order taking precedence, into the resulting config block. If the
 * \c merge parameter is \b false. then reading process stops after the first
 * file is found.
 *
 * A platform specific search path is constructed as follows:
 *
 * ## Windows Platform
 * - .  (the current working directory
 * - ${KWIVER_CONFIG_PATH}          (if set)
 * - $<CSIDL_LOCAL_APPDATA>/<app-name>[/<app-version>]/config
 * - $<CSIDL_APPDATA>/<app-name>[/<app-version>]/config
 * - $<CSIDL_COMMON_APPDATA>/<app-name>[/<app-version>]/config
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 *
 * ## OS/X Apple Platform
 * - .  (the current working directory)
 * - ${KWIVER_CONFIG_PATH}                                    (if set)
 * - ${XDG_CONFIG_HOME}/<app-name>[/<app-version>]/config     (if
 * $XDG_CONFIG_HOME set)
 * - ${HOME}/.config/<app-name>[/<app-version>]/config        (if $HOME set)
 * - /etc/xdg/<app-name>[/<app-version>]/config
 * - /etc/<app-name>[/<app-version>]/config
 * - ${HOME}/Library/Application Support/<app-name>[/<app-version>]/config (if
 * $HOME set)
 * - /Library/Application Support/<app-name>[/<app-version>]/config
 * - /usr/local/share/<app-name>[/<app-version>]/config
 * - /usr/share/<app-name>[/<app-version>]/config
 *
 * If <install-dir> is not `/usr` or `/usr/local`:
 *
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 * - <install-dir>/Resources/config
 *
 * ## Other Posix Platforms (e.g. Linux)
 * - .  (the current working directory
 * - ${KWIVER_CONFIG_PATH}                                    (if set)
 * - ${XDG_CONFIG_HOME}/<app-name>[/<app-version>]/config     (if
 * $XDG_CONFIG_HOME set)
 * - ${HOME}/.config/<app-name>[/<app-version>]/config        (if $HOME set)
 * - /etc/xdg/<app-name>[/<app-version>]/config
 * - /etc/<app-name>[/<app-version>]/config
 * - /usr/local/share/<app-name>[/<app-version>]/config
 * - /usr/share/<app-name>[/<app-version>]/config
 *
 * If <install-dir> is not `/usr` or `/usr/local`:
 *
 * - <install-dir>/share/<app-name>[/<app-version>]/config
 * - <install-dir>/share/config
 * - <install-dir>/config
 *
 * The environment variable \c KWIVER_CONFIG_PATH can be set with a
 * list of one or more directories, in the same manner as the native
 * execution \c PATH variable, to be searched for config files.
 *
 * \throws config_file_not_found_exception
 *    Thrown when the no matching file could be found in the searched paths.
 * \throws config_file_not_read_exception
 *    Thrown when a file could not be read or parsed for whatever reason.
 *
 * \param file_name
 *   The name to the file(s) to read in.
 * \param application_name
 *   The application name, used to build the list of standard locations to be
 *   searched.
 * \param application_version
 *   The application version number, used to build the list of standard
 *   locations to be searched.
 * \param install_prefix
 *   The prefix to which the application is installed (should be one directory
 *   higher than the location of the executing binary).  If not specified
 *   (empty), an attempt to guess the prefix based on the path of the running
 *   executable will be made.
 * \param merge
 *   If \c true, search all locations for matching config files, merging their
 *   contents, with files earlier in the search order taking precedence. If
 *   \c false, read only the first matching file. If this parameter is omitted
 *   the configs are merged.
 *
 * \return
 *   A \c config_block object representing the contents of the read-in file.
 */
)pbdoc" );
  // -------------------------------------------------------------------------
  py::class_< kv::config_block_formatter >( m, "ConfigBlockFormatter" )
    .def( py::init< kv::config_block_sptr >() )
    .def(
      "print", [](kv::config_block_formatter& self, py::object fileHandle){
        std::ofstream fout;
        py::scoped_ostream_redirect stream( fout, fileHandle );
        self.print( fout );
      },
      // doc-string
      "Format config block in simple text format." )
    .def(
      "set_prefix", &kv::config_block_formatter::set_prefix,
      // doc-string
      "Set line prefix for printing." )
    .def(
      "generate_source_loc", &kv::config_block_formatter::generate_source_loc,
      // doc-string
      py::doc(
        "Set option to generate source location.\n"
        "TRUE will generate the source location, FALSE will not."
      )
    );
}
