/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_TRAINING_H
#define VIAME_CORE_UTILITIES_TRAINING_H

#include "viame_core_export.h"

#include <vital/types/detected_object_set.h>
#include <vital/types/category_hierarchy.h>

#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Pipeline type alias
// =============================================================================
typedef std::unique_ptr< kwiver::embedded_pipeline > pipeline_t;

// =============================================================================
// Detection set utilities
// =============================================================================

/// Check if a vector of detection sets is empty (all sets null or empty)
///
/// \param sets Vector of detection sets to check
/// \returns true if all sets are null or empty
VIAME_CORE_EXPORT
bool is_detection_set_empty( const std::vector< kv::detected_object_set_sptr >& sets );

/// Correct common issues in manual annotations
///
/// Fixes:
/// - Negative confidence values (set to 1.0)
/// - Flipped bounding boxes (min > max)
/// - Negative type scores (set to 1.0)
///
/// \param dos Detection set to correct (modified in place)
VIAME_CORE_EXPORT
void correct_manual_annotations( kv::detected_object_set_sptr dos );

/// Convert detections to full-frame labels
///
/// If detections are not already full-frame, creates new full-frame detections
/// with one detection per unique label.
///
/// \param dos Input detection set
/// \param width Image width
/// \param height Image height
/// \returns Detection set with full-frame bounding boxes
VIAME_CORE_EXPORT
kv::detected_object_set_sptr
adjust_to_full_frame( const kv::detected_object_set_sptr dos,
                      unsigned width, unsigned height );

// =============================================================================
// Label adjustment utilities
// =============================================================================

/// Adjust detection labels based on category hierarchy
///
/// - Removes detections with classes not in the hierarchy
/// - Applies synonym mapping from the hierarchy
/// - Optionally removes background classes when foreground is present
///
/// \param input Detection set to adjust (modified in place)
/// \param cats_to_use Category hierarchy for filtering/mapping
/// \param background Set of background class names to suppress
/// \returns true if any foreground (non-background) detections remain
VIAME_CORE_EXPORT
bool adjust_labels( kv::detected_object_set_sptr input,
                    kv::category_hierarchy_sptr cats_to_use,
                    const std::unordered_set< std::string >& background );

/// Adjust labels for a vector of detection sets
///
/// \param input Vector of detection sets to adjust (modified in place)
/// \param cats_to_use Category hierarchy for filtering/mapping
/// \param background Set of background class names to suppress
/// \returns Vector of bools indicating if each frame has foreground detections
VIAME_CORE_EXPORT
std::vector< bool >
adjust_labels( std::vector< kv::detected_object_set_sptr >& input,
               kv::category_hierarchy_sptr cats_to_use,
               const std::unordered_set< std::string >& background );

/// Adjust file and detection lists based on foreground mask
///
/// Removes frames that:
/// - Have no detections
/// - Are background-only and should be downsampled
/// - Are within skip count of last foreground frame
///
/// \param input_files File list to adjust (modified in place)
/// \param input_dets Detection list to adjust (modified in place)
/// \param fg_mask Foreground mask from adjust_labels
/// \param background_ds_rate Downsample rate for background frames (0 = no downsampling)
/// \param background_skip_count Skip this many background frames after foreground
VIAME_CORE_EXPORT
void adjust_labels( std::vector< std::string >& input_files,
                    std::vector< kv::detected_object_set_sptr >& input_dets,
                    const std::vector< bool >& fg_mask,
                    unsigned background_ds_rate = 0,
                    unsigned background_skip_count = 0 );

// =============================================================================
// Downsampling utilities
// =============================================================================

/// Remove elements from a vector based on a boolean mask
///
/// \param input Vector to filter (modified in place)
/// \param remove Boolean mask (true = remove element)
template< typename T >
void conditional_remove( std::vector< T >& input, const std::vector< bool >& remove )
{
  std::vector< T > output;
  for( unsigned i = 0; i < input.size(); i++ )
  {
    if( !remove[i] )
    {
      output.push_back( input[i] );
    }
  }
  input = output;
}

/// Downsample file and detection lists
///
/// \param input_files File list to adjust (modified in place)
/// \param input_dets Detection list to adjust (modified in place)
/// \param downsample_factor Downsample factor (e.g., 2 keeps every 2nd frame)
/// \param substr If non-empty, only downsample files containing this substring
VIAME_CORE_EXPORT
void downsample_data( std::vector< std::string >& input_files,
                      std::vector< kv::detected_object_set_sptr >& input_dets,
                      double downsample_factor,
                      const std::string& substr = "" );

// =============================================================================
// Embedded pipeline utilities
// =============================================================================

/// Load and start an embedded pipeline from a file
///
/// \param pipeline_filename Path to pipeline file
/// \returns Unique pointer to started pipeline, or nullptr if filename is empty
/// \throws sprokit::invalid_configuration_exception on pipeline errors
VIAME_CORE_EXPORT
pipeline_t load_embedded_pipeline( const std::string& pipeline_filename );

/// Run an embedded pipeline on a single image
///
/// Sends input_file_name and output_file_name to the pipeline,
/// optionally also output_file_name2 and output_file_name3 if the
/// pipeline expects them.
///
/// \param pipe Pipeline to run
/// \param pipe_file Pipeline file path (used to check for optional ports)
/// \param input_name Input image filename
/// \param output_name Output image filename
/// \returns Success flag from pipeline output
/// \throws std::runtime_error if pipeline terminates unexpectedly
VIAME_CORE_EXPORT
bool run_pipeline_on_image( pipeline_t& pipe,
                            const std::string& pipe_file,
                            const std::string& input_name,
                            const std::string& output_name );

// =============================================================================
// Augmentation utilities
// =============================================================================

/// Generate an augmented filename for caching
///
/// Creates a path like: output_dir/subdir/filename_no_ext.ext
/// If output_dir is empty, uses the system temp directory.
///
/// \param name Original filename
/// \param subdir Subdirectory name for organization
/// \param output_dir Output directory (empty = temp dir)
/// \param ext Output extension (default: ".png")
/// \returns Generated filename path
VIAME_CORE_EXPORT
std::string get_augmented_filename( const std::string& name,
                                    const std::string& subdir,
                                    const std::string& output_dir = "",
                                    const std::string& ext = ".png" );

// =============================================================================
// Video frame extraction
// =============================================================================

/// Extract frames from a video file
///
/// Uses viame with a pipeline to extract frames at the specified rate.
///
/// \param video_filename Path to video file
/// \param pipeline_filename Path to extraction pipeline
/// \param frame_rate Target frame rate
/// \param output_directory Directory to store extracted frames
/// \param skip_extract_if_exists Skip extraction if output directory exists
/// \param max_frame_count Maximum frames to extract (0 = unlimited)
/// \returns Vector of extracted frame file paths
VIAME_CORE_EXPORT
std::vector< std::string >
extract_video_frames( const std::string& video_filename,
                      const std::string& pipeline_filename,
                      double frame_rate,
                      const std::string& output_directory,
                      bool skip_extract_if_exists = false,
                      unsigned max_frame_count = 0 );

} // end namespace viame

#endif /* VIAME_CORE_UTILITIES_TRAINING_H */
