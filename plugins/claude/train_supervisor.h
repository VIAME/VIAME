/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CLAUDE_TRAIN_SUPERVISOR_H
#define VIAME_CLAUDE_TRAIN_SUPERVISOR_H

#include "viame_claude_export.h"

#include <string>
#include <vector>

namespace viame {
namespace claude {

/// Options for LLM supervised training, as driven by the train tool's
/// --llm-assist family of command line flags.
struct llm_train_options
{
  /// Resolved claude executable (path or command name known to be runnable)
  std::string claude_cmd;
  /// Optional claude model over-ride (--llm-model)
  std::string model;
  /// Seconds between periodic training checkups
  int poll_seconds = 600;
  /// Maximum number of LLM-initiated restarts of the training process
  int max_restarts = 2;
  /// Do not query the user before applying suggestions
  bool no_query = false;
  /// True when --llm-assist was explicitly "on" (vs "auto")
  bool required = false;

  /// Directory for supervisor state (logs, prompts, settings)
  std::string state_dir;
  /// Dump of the effective training configuration (possibly truncated)
  std::string config_text;
  /// Short description of the input dataset (folders, labels, counts)
  std::string dataset_summary;
  /// Newline-separated list of trainable detector types
  std::string trainable_types;
  /// Newline-separated list of packaged train_*.conf files
  std::string available_configs;
  /// The original command line, for context
  std::string original_cmdline;

  /// Original --config value ("" if --detector was used instead)
  std::string original_config;
  /// Original --detector value ("" if --config was used instead)
  std::string original_detector;
  /// Original --settings-file value, merged into suggested settings
  std::string user_settings_file;

  /// Child argv (minus executable): applet name followed by all original
  /// arguments except --config/--detector/--settings-file/--llm-* options
  std::vector< std::string > child_args_base;
};

/// Environment marker set on the supervised training child. Callers check for
/// it to know they are the child, and so should run training directly instead
/// of starting another supervisor.
VIAME_CLAUDE_EXPORT
extern const char* const child_env_marker;

/// Resolve the claude executable, returning an empty string if not found.
VIAME_CLAUDE_EXPORT
std::string find_claude_binary( const std::string& cmd_override );

/// Run training under claude supervision: query claude for configuration
/// suggestions, then run training as a monitored subprocess, restarting it
/// with revised settings on failure or on claude's recommendation.
///
/// Returns the applet exit code, or -1 if claude turned out to be unusable
/// and the caller should fall back to normal in-process training.
VIAME_CLAUDE_EXPORT
int run_llm_supervised_training( const llm_train_options& options );

} // namespace claude
} // namespace viame

#endif // VIAME_CLAUDE_TRAIN_SUPERVISOR_H
