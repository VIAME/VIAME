/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "utilities_training.h"
#include "utilities_file.h"

#include <kwiversys/SystemTools.hxx>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/processes/adapters/adapter_types.h>

#include <fstream>
#include <iostream>
#include <map>
#include <cstdlib>

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

namespace viame {

// =============================================================================
// Detection set utilities
// =============================================================================

bool is_detection_set_empty( const std::vector< kv::detected_object_set_sptr >& sets )
{
  for( const auto& set : sets )
  {
    if( set && !set->empty() )
    {
      return false;
    }
  }
  return true;
}

void correct_manual_annotations( kv::detected_object_set_sptr dos )
{
  if( !dos )
  {
    return;
  }

  for( kv::detected_object_sptr do_sptr : *dos )
  {
    if( do_sptr->confidence() < 0.0 )
    {
      do_sptr->set_confidence( 1.0 );
    }

    kv::bounding_box_d do_box = do_sptr->bounding_box();

    if( do_box.min_x() > do_box.max_x() )
    {
      do_box = kv::bounding_box_d(
        do_box.max_x(), do_box.min_y(), do_box.min_x(), do_box.max_y() );
    }
    if( do_box.min_y() > do_box.max_y() )
    {
      do_box = kv::bounding_box_d(
        do_box.min_x(), do_box.max_y(), do_box.max_x(), do_box.min_y() );
    }

    do_sptr->set_bounding_box( do_box );

    if( do_sptr->type() )
    {
      kv::detected_object_type_sptr type_sptr = do_sptr->type();

      std::string top_category;
      double top_score;

      type_sptr->get_most_likely( top_category, top_score );

      if( top_score < 0.0 )
      {
        type_sptr->set_score( top_category, 1.0 );
        do_sptr->set_type( type_sptr );
      }
    }
  }
}

kv::detected_object_set_sptr
adjust_to_full_frame( const kv::detected_object_set_sptr dos,
                      unsigned width, unsigned height )
{
  if( !dos )
  {
    return dos;
  }

  bool adj_required = false;

  for( auto det : *dos )
  {
    if( det &&
        ( det->bounding_box().min_x() != 0 ||
          det->bounding_box().min_y() != 0 ||
          det->bounding_box().width() != width ||
          det->bounding_box().height() != height ) )
    {
      adj_required = true;
    }
  }

  if( !adj_required )
  {
    return dos;
  }

  kv::detected_object_set_sptr output =
    std::make_shared< kv::detected_object_set >();

  kv::bounding_box_d ff_box( 0, 0, width, height );

  std::map< std::string, int > obs_labels;

  for( auto det : *dos )
  {
    std::string label;
    double score;

    if( det->type() )
    {
      det->type()->get_most_likely( label, score );
    }

    obs_labels[ label ]++;

    if( obs_labels[ label ] > 1 )
    {
      continue;
    }

    det->set_bounding_box( ff_box );

    output->add( det );
  }

  return output;
}

// =============================================================================
// Label adjustment utilities
// =============================================================================

bool adjust_labels( kv::detected_object_set_sptr input,
                    kv::category_hierarchy_sptr cats_to_use,
                    const std::unordered_set< std::string >& background )
{
  if( !input )
  {
    return false;
  }

  if( cats_to_use )
  {
    std::unordered_set< std::string > frame_cats;

    // Remove detections not present in labels and use synonym table to update labels
    input->filter( [&frame_cats, cats_to_use]( kv::detected_object_sptr& det )
    {
      if( !det || !det->type() )
      {
        return true;
      }

      std::string cat, new_cat;
      double score;

      det->type()->get_most_likely( cat, score );

      if( !cats_to_use->has_class_name( cat ) )
      {
        return true;
      }

      new_cat = cats_to_use->get_class_name( cat );
      if( new_cat != cat )
      {
        det->set_type(
          std::make_shared< kv::detected_object_type >(
            new_cat, score ) );
      }
      frame_cats.insert( new_cat );
      return false;
    } );

    if( !background.empty() )
    {
      // Are any FG more important categories present on frame
      bool has_fg = false;
      for( auto lbl : frame_cats )
      {
        if( background.count( lbl ) == 0 )
        {
          has_fg = true;
          break;
        }
      }

      // Remove background categories
      if( has_fg )
      {
        input->filter( [background]( kv::detected_object_sptr& det )
        {
          std::string cat;
          det->type()->get_most_likely( cat );
          return background.count( cat );
        } );
      }

      frame_cats.clear();

      // Remove duplicates if present
      input->filter( [&frame_cats]( kv::detected_object_sptr& det )
      {
        std::string cat;
        det->type()->get_most_likely( cat );
        if( frame_cats.count( cat ) )
        {
          return true;
        }
        frame_cats.insert( cat );
        return false;
      } );

      return has_fg;
    }
  }
  return !input->empty();
}

std::vector< bool >
adjust_labels( std::vector< kv::detected_object_set_sptr >& input,
               kv::category_hierarchy_sptr cats_to_use,
               const std::unordered_set< std::string >& background )
{
  std::vector< bool > fg_mask;

  for( auto set : input )
  {
    fg_mask.push_back( adjust_labels( set, cats_to_use, background ) );
  }

  return fg_mask;
}

void adjust_labels( std::vector< std::string >& input_files,
                    std::vector< kv::detected_object_set_sptr >& input_dets,
                    const std::vector< bool >& fg_mask,
                    unsigned background_ds_rate,
                    unsigned background_skip_count )
{
  if( !background_ds_rate && !background_skip_count )
  {
    return;
  }

  std::vector< bool > to_remove( fg_mask.size(), false );
  unsigned since_last_fg = 0, bg_counter = 0;

  for( unsigned i = 0; i < input_files.size(); i++ )
  {
    if( fg_mask[i] )
    {
      since_last_fg = 0;
    }
    else if( !input_dets[i] || input_dets[i]->empty() )
    {
      to_remove[i] = true;
    }
    else
    {
      if( ( background_ds_rate && bg_counter % background_ds_rate != 0 ) ||
          ( background_skip_count && since_last_fg < background_skip_count ) )
      {
        to_remove[i] = true;
      }

      bg_counter++;
      since_last_fg++;
    }
  }

  conditional_remove( input_files, to_remove );
  conditional_remove( input_dets, to_remove );
}

// =============================================================================
// Downsampling utilities
// =============================================================================

void downsample_data( std::vector< std::string >& input_files,
                      std::vector< kv::detected_object_set_sptr >& input_dets,
                      double downsample_factor,
                      const std::string& substr )
{
  if( !downsample_factor || downsample_factor == 1.0 || input_files.empty() )
  {
    return;
  }

  if( downsample_factor < 1.0 )
  {
    downsample_factor = 1.0 / downsample_factor;
  }

  double counter = 1.0;

  std::vector< std::string > original_files = input_files;
  std::vector< kv::detected_object_set_sptr > original_dets = input_dets;

  input_files.clear();
  input_dets.clear();

  for( unsigned i = 0; i < original_files.size(); i++ )
  {
    if( !substr.empty() && original_files[i].find( substr ) == std::string::npos )
    {
      input_files.push_back( original_files[i] );
      input_dets.push_back( original_dets[i] );
      continue;
    }

    counter = counter + 1.0;

    if( counter >= downsample_factor )
    {
      counter -= downsample_factor;

      input_files.push_back( original_files[i] );
      input_dets.push_back( original_dets[i] );
    }
  }
}

// =============================================================================
// Embedded pipeline utilities
// =============================================================================

pipeline_t load_embedded_pipeline( const std::string& pipeline_filename )
{
  std::unique_ptr< kwiver::embedded_pipeline > external_pipeline;

  if( !pipeline_filename.empty() )
  {
    auto dir = filesystem::path( pipeline_filename ).parent_path();

    std::unique_ptr< kwiver::embedded_pipeline > new_pipeline =
      std::unique_ptr< kwiver::embedded_pipeline >( new kwiver::embedded_pipeline() );

    std::ifstream pipe_stream;
    pipe_stream.open( pipeline_filename, std::ifstream::in );

    if( !pipe_stream )
    {
      throw sprokit::invalid_configuration_exception( "viame_train_detector",
        "Unable to open pipeline file: " + pipeline_filename );
    }

    try
    {
      new_pipeline->build_pipeline( pipe_stream, dir.string() );
      new_pipeline->start();
    }
    catch( const std::exception& e )
    {
      throw sprokit::invalid_configuration_exception( "viame_train_detector",
                                                      e.what() );
    }

    external_pipeline = std::move( new_pipeline );
    pipe_stream.close();
  }

  return external_pipeline;
}

bool run_pipeline_on_image( pipeline_t& pipe,
                            const std::string& pipe_file,
                            const std::string& input_name,
                            const std::string& output_name )
{
  kwiver::adapter::adapter_data_set_t ids =
    kwiver::adapter::adapter_data_set::create();

  ids->add_value( "input_file_name", input_name );

  ids->add_value( "output_file_name", output_name );

  if( file_contains_string( pipe_file, "output_file_name2" ) )
  {
    ids->add_value( "output_file_name2", add_aux_ext( output_name, 1 ) );
  }

  if( file_contains_string( pipe_file, "output_file_name3" ) )
  {
    ids->add_value( "output_file_name3", add_aux_ext( output_name, 2 ) );
  }

  pipe->send( ids );

  auto const& ods = pipe->receive();

  if( ods->is_end_of_data() )
  {
    throw std::runtime_error( "Pipeline terminated unexpectedly" );
  }

  auto const& success_flag = ods->find( "success_flag" );

  return success_flag->second->get_datum< bool >();
}

// =============================================================================
// Augmentation utilities
// =============================================================================

std::string get_augmented_filename( const std::string& name,
                                    const std::string& subdir,
                                    const std::string& output_dir,
                                    const std::string& ext )
{
  std::string file_name =
    kwiversys::SystemTools::GetFilenameName( name );

  std::size_t last_index = file_name.find_last_of( "." );
  std::string file_name_no_ext = file_name.substr( 0, last_index );

  std::vector< std::string > full_path;

  full_path.push_back( "" );

  if( output_dir.empty() )
  {
    full_path.push_back( filesystem::temp_directory_path().string() );
  }
  else
  {
    full_path.push_back( output_dir );
  }

  full_path.push_back( subdir );
  full_path.push_back( file_name_no_ext + ext );

  std::string mod_path = kwiversys::SystemTools::JoinPath( full_path );
  return mod_path;
}

// =============================================================================
// Video frame extraction
// =============================================================================

std::vector< std::string >
extract_video_frames( const std::string& video_filename,
                      const std::string& pipeline_filename,
                      double frame_rate,
                      const std::string& output_directory,
                      bool skip_extract_if_exists,
                      unsigned max_frame_count )
{
  std::cout << "Extracting frames from " << video_filename
            << " at rate " << frame_rate << std::endl;

  std::vector< std::string > output;

  std::string video_no_path = get_filename_no_path( video_filename );
  std::string output_dir = append_path( output_directory, video_no_path );
  std::string output_path = append_path( output_dir, "frame%06d.png" );
  std::string frame_rate_str = std::to_string( frame_rate );

  if( !skip_extract_if_exists )
  {
    if( does_folder_exist( output_dir ) )
    {
      filesystem::remove_all( output_dir );
    }

    if( !create_folder( output_dir ) )
    {
      std::cout << "Error: Unable to create folder: " << output_dir << std::endl;
      return output;
    }
  }

  std::string cmd = "kwiver";

#ifdef WIN32
  cmd = cmd + ".exe";
#endif

  cmd = cmd + " runner " + add_quotes( pipeline_filename ) + " ";
  cmd = cmd + "-s input:video_filename=" + add_quotes( video_filename ) + " ";
  cmd = cmd + "-s input:video_reader:type=vidl_ffmpeg ";
  cmd = cmd + "-s downsampler:target_frame_rate=" + frame_rate_str + " ";
  cmd = cmd + "-s image_writer:file_name_template=" + add_quotes( output_path ) + " ";

  if( max_frame_count > 0 )
  {
    cmd = cmd + "-s input:video_reader:vidl_ffmpeg:stop_after_frame="
              + std::to_string( max_frame_count );
  }

  if( !skip_extract_if_exists ||
      ( !does_folder_exist( output_dir ) && create_folder( output_dir ) ) ||
      folder_contains_less_than_n_files( output_dir, 3 ) )
  {
    std::system( cmd.c_str() );
  }

  list_files_in_folder( output_dir, output );
  return output;
}

} // end namespace viame
