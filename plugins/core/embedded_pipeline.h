/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_EMBEDDED_PIPELINE_H
#define VIAME_CORE_EMBEDDED_PIPELINE_H

#include "viame_embedded_pipeline_export.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace kwiver { class embedded_pipeline; }

namespace viame {

/// Options for replacing file readers and writers with memory adapters.
struct embedded_pipeline_options
{
  /// Additional directories for resolving pipeline includes.
  std::vector< std::string > search_paths;
  /// Process names to replace. nullopt auto-detects standard reader/writer
  /// types; an explicitly empty list is an error. Inputs must be sources,
  /// and outputs must be sinks. Other processes keep their behavior.
  std::optional< std::vector< std::string > > inputs;
  std::optional< std::vector< std::string > > outputs;
};

/// Prepared pipeline and original process.port -> adapter port mappings.
///
/// Preparation does not instantiate algorithms or load models. Source files
/// and model assets must remain available while building and running the
/// pipeline. Output ports are synchronized by a single output adapter, so
/// their streams must have compatible rates. Sampling/batching are preserved.
struct VIAME_EMBEDDED_PIPELINE_EXPORT embedded_pipeline_description
{
  std::string pipeline_text;
  std::string source_directory;
  std::vector< std::string > input_names;
  std::map< std::string, std::string > input_ports;
  std::map< std::string, std::string > output_ports;

  /// Build/configure a fresh native pipeline from memory, without starting it.
  /// Native pipeline construction initializes plugins automatically. The
  /// caller manages start/send/receive/end-of-input/wait as with a normal native
  /// embedded pipeline. Every mapped input port needs a correctly typed value.
  void build( kwiver::embedded_pipeline& pipeline ) const;
};

/// Parse a normal .pipe and replace its selected sources/sinks with adapters.
/// Includes, substitutions and relativepath entries use the native parser.
/// No Python interpreter or intermediate .pipe file is required.
/// Selection/topology errors throw std::invalid_argument; native parse and
/// configuration errors propagate from the pipeline framework.
VIAME_EMBEDDED_PIPELINE_EXPORT
embedded_pipeline_description prepare_embedded_pipeline(
  std::string const& filename,
  embedded_pipeline_options const& options = {} );

} // namespace viame

#endif
