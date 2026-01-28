/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_MANIPULATE_PIPELINES_H
#define VIAME_CORE_MANIPULATE_PIPELINES_H

#include "viame_core_export.h"

#include <map>
#include <set>
#include <string>

namespace viame {

/// Detect the leading whitespace indent of the line containing a marker
/// in a template string. Returns the indent string (spaces/tabs).
VIAME_CORE_EXPORT
std::string detect_marker_indent(
    const std::string& template_content,
    const std::string& marker );

/// Format a trainer output map as KWIVER .pipe block syntax for use as
/// the [-DETECTOR-IMPL-] replacement in pipeline templates.
///
/// config_entries: non-file key-value pairs (e.g. "type"="netharn",
///                 "netharn:deployed"="trained_detector.zip")
/// copied_filenames: set of filenames being copied (used to decide
///                   which values need "relativepath" syntax)
/// base_indent: leading whitespace for the first line; subsequent
///              lines in the output include this indent
VIAME_CORE_EXPORT
std::string format_output_as_pipe_blocks(
    const std::map< std::string, std::string >& config_entries,
    const std::set< std::string >& copied_filenames,
    const std::string& base_indent );

/// Generate the [-DETECTOR-IMPL-] replacement string from a trainer
/// output map and a pipeline template file.
///
/// Reads the template, separates config entries from file copies,
/// detects the marker indent, and formats the output as .pipe blocks.
/// Returns empty string if the template doesn't exist or doesn't
/// contain [-DETECTOR-IMPL-].
VIAME_CORE_EXPORT
std::string generate_detector_impl_replacement(
    const std::map< std::string, std::string >& output_map,
    const std::string& pipeline_template );

} // end namespace viame

#endif /* VIAME_CORE_MANIPULATE_PIPELINES_H */
