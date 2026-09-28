/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "utilities_model_card.h"
#include "utilities_file.h"

#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace viame {

namespace kv = kwiver::vital;

namespace {

std::map< std::string, unsigned >
count_truth_classes( const std::vector< kv::detected_object_set_sptr >& truth,
                     unsigned& annotated_frames )
{
  std::map< std::string, unsigned > counts;
  annotated_frames = 0;

  for( const auto& set : truth )
  {
    if( !set || set->empty() )
    {
      continue;
    }

    ++annotated_frames;

    for( const auto& det : *set )
    {
      std::string name = "unlabeled";

      if( det && det->type() && det->type()->size() > 0 )
      {
        det->type()->get_most_likely( name );
      }

      counts[ name ] += 1;
    }
  }

  return counts;
}

std::string
markdown_escape( const std::string& text )
{
  std::string out;

  for( char c : text )
  {
    if( c == '|' )
    {
      out += "\\|";
    }
    else
    {
      out += c;
    }
  }

  return out;
}

} // namespace

// Writes MODEL_CARD.md next to the trained pipelines, in the spirit of a
// Hugging Face model card: what was trained, over which categories, and on
// exactly which data. The frame-level split is written alongside it so the
// automatic validation choice is reproducible.
void
write_model_card( const model_card_inputs& in )
{
  const std::string splits_dir = append_path( in.output_directory, "splits" );
  create_folder( splits_dir );

  auto write_frame_list = [&]( const std::string& name,
                               const std::vector< std::string >& frames )
  {
    std::ofstream out( append_path( splits_dir, name ) );

    for( const auto& frame : frames )
    {
      out << frame << std::endl;
    }
  };

  write_frame_list( "train_frames.txt", in.train_frames );
  write_frame_list( "validation_frames.txt", in.validation_frames );

  unsigned train_annotated = 0, validation_annotated = 0;
  const auto train_counts = count_truth_classes( in.train_truth, train_annotated );
  const auto validation_counts = count_truth_classes( in.validation_truth, validation_annotated );

  unsigned train_total = 0, validation_total = 0;
  for( const auto& kv_pair : train_counts ) { train_total += kv_pair.second; }
  for( const auto& kv_pair : validation_counts ) { validation_total += kv_pair.second; }

  std::string model_name = in.output_directory;
  while( !model_name.empty() && ( model_name.back() == '/' || model_name.back() == '\\' ) )
  {
    model_name.pop_back();
  }
  model_name = get_filename_no_path( model_name );
  if( model_name.empty() || model_name == "." || model_name == "trained_model" )
  {
    model_name = "Trained VIAME model";
  }

  std::time_t now = std::time( nullptr );
  char date[ 32 ] = { 0 };
  std::strftime( date, sizeof( date ), "%Y-%m-%d", std::localtime( &now ) );

  std::ofstream card( append_path( in.output_directory, "MODEL_CARD.md" ) );

  if( !card )
  {
    std::cout << "Unable to write MODEL_CARD.md in " << in.output_directory << std::endl;
    return;
  }

  // Front matter
  card << "---" << std::endl
       << "library_name: viame" << std::endl
       << "pipeline_tag: "
       << ( in.detector_types.empty() ? "object-tracking" : "object-detection" ) << std::endl
       << "tags:" << std::endl
       << "- viame" << std::endl;
  // "wrapper (inner)" entries become one tag per name.
  std::set< std::string > tags;
  for( const auto& list : { in.detector_types, in.tracker_types } )
  {
    for( const auto& type : list )
    {
      std::string token;
      for( char c : type + " " )
      {
        if( std::isalnum( static_cast< unsigned char >( c ) ) || c == '_' || c == '-' )
        {
          token += c;
        }
        else if( !token.empty() )
        {
          tags.insert( token );
          token.clear();
        }
      }
    }
  }
  for( const auto& tag : tags ) { card << "- " << tag << std::endl; }
  card << "---" << std::endl << std::endl;

  card << "# " << model_name << std::endl << std::endl
       << "Trained with `viame train` on " << date << "." << std::endl << std::endl;

  // Model details
  card << "## Model details" << std::endl << std::endl
       << "| | |" << std::endl
       << "|---|---|" << std::endl;

  auto join = []( const std::vector< std::string >& v )
  {
    std::string out;
    for( const auto& e : v ) { out += ( out.empty() ? "" : ", " ) + e; }
    return out;
  };

  if( !in.detector_types.empty() )
  {
    card << "| Detector | " << join( in.detector_types ) << " |" << std::endl
         << "| Detector pipeline | `" << get_filename_no_path( in.detector_pipeline ) << "` |" << std::endl;
  }
  if( !in.tracker_types.empty() )
  {
    card << "| Tracker | " << join( in.tracker_types ) << " |" << std::endl
         << "| Tracker pipeline | `" << get_filename_no_path( in.tracker_pipeline ) << "` |" << std::endl;
  }
  if( !in.config_file.empty() )
  {
    card << "| Training config | `" << get_filename_no_path( in.config_file ) << "` |" << std::endl;
  }
  card << "| Seed weights | "
       << ( in.init_weights.empty() ? "none (trained from the configuration's defaults)"
                                    : "`" + get_filename_no_path( in.init_weights ) + "`" )
       << " |" << std::endl
       << "| Frames used | " << ( in.gt_frames_only ? "annotated frames only" : "all frames" )
       << " |" << std::endl << std::endl;

  // Categories
  card << "## Categories" << std::endl << std::endl;

  std::vector< std::string > class_names;
  if( in.labels )
  {
    class_names = in.labels->all_class_names();
  }
  else
  {
    for( const auto& kv_pair : train_counts ) { class_names.push_back( kv_pair.first ); }
    for( const auto& kv_pair : validation_counts )
    {
      if( !train_counts.count( kv_pair.first ) ) { class_names.push_back( kv_pair.first ); }
    }
  }

  card << "The model was trained over " << class_names.size() << " categor"
       << ( class_names.size() == 1 ? "y" : "ies" )
       << ( in.labels ? ", as declared by the labels file"
                      : ", taken from the ground truth as no labels file was given" )
       << "." << std::endl << std::endl
       << "| Category | Also matched as | Train annotations | Validation annotations |" << std::endl
       << "|---|---|---:|---:|" << std::endl;

  auto count_of = []( const std::map< std::string, unsigned >& counts, const std::string& name )
  {
    auto it = counts.find( name );
    return it == counts.end() ? 0u : it->second;
  };

  for( const auto& name : class_names )
  {
    std::vector< std::string > synonyms;
    if( in.labels )
    {
      synonyms = in.labels->get_class_synonyms( name );
    }
    card << "| " << markdown_escape( name ) << " | " << markdown_escape( join( synonyms ) )
         << " | " << count_of( train_counts, name )
         << " | " << count_of( validation_counts, name ) << " |" << std::endl;
  }
  card << "| **Total** | | " << train_total << " | " << validation_total << " |"
       << std::endl << std::endl;

  // Data splits
  card << "## Data splits" << std::endl << std::endl;

  size_t train_items = 0, validation_items = 0;
  for( size_t i = 0; i < in.items.size(); ++i )
  {
    ( in.validation_items.count( i ) ? validation_items : train_items ) += 1;
  }

  card << "| Split | Sequences | Frames | Annotated frames | Annotations |" << std::endl
       << "|---|---:|---:|---:|---:|" << std::endl
       << "| Train | " << train_items << " | " << in.train_frames.size() << " | "
       << train_annotated << " | " << train_total << " |" << std::endl
       << "| Validation | " << ( in.validation_auto_selected ? std::string( "(from train)" )
                                                             : std::to_string( validation_items ) )
       << " | " << in.validation_frames.size() << " | " << validation_annotated << " | "
       << validation_total << " |" << std::endl
       << "| Test | " << in.test_items.size() << " | | | |" << std::endl << std::endl;

  card << "The exact frame lists are in `splits/train_frames.txt` and "
       << "`splits/validation_frames.txt`." << std::endl << std::endl;

  card << "### Train" << std::endl << std::endl;
  for( size_t i = 0; i < in.items.size(); ++i )
  {
    if( in.validation_items.count( i ) ) { continue; }
    card << "- `" << in.items[i] << "`";
    if( i < in.item_frame_counts.size() )
    {
      card << " (" << in.item_frame_counts[i] << " frames)";
    }
    card << std::endl;
  }
  card << std::endl;

  card << "### Validation" << std::endl << std::endl;
  if( in.validation_auto_selected )
  {
    std::ostringstream pct;
    pct << std::fixed << std::setprecision( 0 ) << in.validation_percent * 100.0;
    card << "No validation set was given, so VIAME held out " << in.validation_frames.size()
         << " frame" << ( in.validation_frames.size() == 1 ? "" : "s" )
         << " (target " << pct.str() << "% of the training frames, in bursts) "
         << "from the training sequences above. The frames chosen are listed in "
         << "`splits/validation_frames.txt`." << std::endl;
  }
  else if( in.validation_frames.empty() )
  {
    card << "No validation set was used." << std::endl;
  }
  else
  {
    for( size_t i = 0; i < in.items.size(); ++i )
    {
      if( !in.validation_items.count( i ) ) { continue; }
      card << "- `" << in.items[i] << "`";
      if( i < in.item_frame_counts.size() )
      {
        card << " (" << in.item_frame_counts[i] << " frames)";
      }
      card << std::endl;
    }
  }
  card << std::endl;

  card << "### Test" << std::endl << std::endl;
  if( in.test_items.empty() )
  {
    card << "No test set was given." << std::endl;
  }
  else
  {
    for( const auto& item : in.test_items )
    {
      card << "- `" << item << "`" << std::endl;
    }
  }
  card << std::endl;

  // Evaluation
  card << "## Evaluation" << std::endl << std::endl;

  std::ifstream summary;
  if( !in.test_results_dir.empty() )
  {
    summary.open( append_path( in.test_results_dir, "summary.txt" ) );
  }

  if( summary.is_open() )
  {
    card << "Scores of the trained detector on the test sequences, computed by "
         << "`viame score`. Full metrics, plots, drawn frames and the per-sequence "
         << "detections are in `model_evaluation/test/`, and in "
         << "`model_evaluation/validation/` for the validation sequences when "
         << "those were given." << std::endl << std::endl
         << "```" << std::endl;
    std::string line;
    while( std::getline( summary, line ) )
    {
      card << line << std::endl;
    }
    card << "```" << std::endl;
  }
  else if( !in.test_items.empty() )
  {
    card << "A test set was given but could not be scored; see the training log." << std::endl;
  }
  else
  {
    card << "No held-out test set was scored. Label some sequences as test to have "
         << "the trained model evaluated automatically." << std::endl;
  }
  card << std::endl;

  std::cout << "Wrote model card to "
            << append_path( in.output_directory, "MODEL_CARD.md" ) << std::endl;
}

} // namespace viame
