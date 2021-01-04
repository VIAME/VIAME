// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_TOOLS_PIPELINE_BUILDER_H
#define SPROKIT_TOOLS_PIPELINE_BUILDER_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <vital/vital_types.h>
#include <vital/noncopyable.h>
#include <vital/config/config_block_types.h>

#include <sprokit/pipeline/types.h>
#include <sprokit/pipeline_util/pipe_bakery.h>

#include <istream>
#include <string>
#include <memory>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * \brief Class to build pipeline.
 *
 * This class is the main interface for creating a pipeline. A
 * pipeline builder (object of this class) is created and configured
 * with the pipeline description. Configuration items can be added as
 * needed.
 *
 * After the pipeline builder is configured as desired, it can build
 * the pipeline which is ready to process.
 */
class SPROKIT_PIPELINE_UTIL_EXPORT pipeline_builder
  : kwiver::vital::noncopyable
{
public:
  /**
   * \brief Create default pipeline builder object.
   *
   */
  pipeline_builder();
  virtual ~pipeline_builder() = default;

  /**
   * \brief Load pipeline configuration from stream.
   *
   * This method loads the pipeline configuration into the
   * builder. This can be used to add additional configuration files
   * to the internal pipeline representation.
   *
   * \param istr Stream containing the textual pipeline definition.
   * \param def_file The default file name used when reporting errors
   * from the stream. The directory portion is used when resolving
   * included files and relpath specifiers.
   */
  void load_pipeline(std::istream& istr, kwiver::vital::path_t const& def_file = "" );

  /**
   * \brief Load pipeline configuration from file name.
   *
   * This method loads the pipeline configuration into the
   * builder. This can be used to add additional configuration files
   * to the internal pipeline representation.
   *
   * \param def_file The name of the pipeline file to use.
   */
  void load_pipeline( kwiver::vital::path_t const& def_file );

  /**
   * \brief Load cluster from stream.
   *
   * This method loads the cluster into the builder. This can be used
   * to add additional configuration files to the internal pipeline
   * representation.
   *
   * \param istr Stream containing the textual cluster definition.
   * \param def_file The default file name used when reporting errors
   * from the stream. The directory portion is used when resolving
   * included files and relpath specifiers.
   */
  void load_cluster( std::istream& istr, kwiver::vital::path_t const& def_file = "" );

  /**
   * \brief Load cluster from file name.
   *
   * This method loads the cluster into the builder. This can be used
   * to add additional configuration files to the internal pipeline
   * representation.
   *
   * \param def_file The name of the cluster file to use.
   */
  void load_cluster( kwiver::vital::path_t const& def_file );

  /**
   * \brief Load supplemental data into pipeline description.
   *
   * Adds supplemental block to the internal representation of the pipeline.
   *
   * \param path File to read.
   */
  void load_supplement( kwiver::vital::path_t const& path );

  /**
   * \brief Add single config entry
   *
   * Add a single config entry to the internal pipeline
   * representation. A config entry has for form key=value
   *
   * \param setting String containing a single config setting entry.
   */
  void add_setting(std::string const& setting);

  //@{
  /**
   * \brief Add directory to search path.
   *
   * This method adds a directory to the end of the config file search
   * path. This search path is used to locate all referenced included
   * files only.
   *
   * @param file_path Directory or list to add to end of search path.
   */
  void add_search_path( kwiver::vital::config_path_t const& file_path );
  void add_search_path( kwiver::vital::config_path_list_t const& file_path );
  //@}

  /**
   * \brief Create pipeline from internal representation.
   *
   * This method instantiates a pipeline from the accumulated internal
   * pipeline representation.
   *
   * \return A new pipeline object.
   */
  sprokit::pipeline_t pipeline() const;

  /**
   * \brief Create cluster and return info object.
   *
   * This method bakes the accumulated cluster blocks and returns the
   * resulting info object.
   *
   * \return Cluster info object.
   */
  sprokit::cluster_info_t cluster_info() const;

  /**
   * \brief Extract config block from pipeline.
   *
   * This method extracts the config for the pipeline.
   *
   * \return Block containing the whole pipeline config.
   */
  kwiver::vital::config_block_sptr config() const;

  /**
   * \brief Get internal representation of pipeline.
   *
   *
   * \return List of internal pipeline blocks.
   */
  sprokit::pipe_blocks pipeline_blocks() const;

  /**
   * \brief List of internal cluster blocks
   *
   *
   * \return The list of internal cluster blocks.
   */
  sprokit::cluster_blocks cluster_blocks() const;

protected:
  void process_env(); // get default search path and env path. Add to m_search_path.

private:
  kwiver::vital::logger_handle_t m_logger;

  // List of pipe blocks
  sprokit::pipe_blocks m_blocks;
  sprokit::cluster_blocks m_cluster_blocks;

  // file search path list
  kwiver::vital::config_path_list_t m_search_path;
};

using  pipeline_builder_sptr = std::shared_ptr< pipeline_builder>;

}

#endif // SPROKIT_TOOLS_PIPELINE_BUILDER_H
