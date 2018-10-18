#!/usr/bin/env python

import sys
import os
import argparse
import signal
import subprocess

sys.dont_write_bytecode = True

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  if not os.path.isdir( folder ):
    exit_with_error( "Input folder \"" + folder + "\" does not exist" )
  return [
    os.path.join(folder, f) for f in sorted(os.listdir(folder))
    if not f.startswith('.')
  ]

# Create a directory if it doesn't exist
def create_dir( dirname, logging=True ):
  if not os.path.exists( dirname ):
    if logging:
      print( "Creating " + dirname )
    os.makedirs( dirname )

def get_ffmpeg_cmd():
  if os.name == 'nt':
    return ['ffmpeg.exe']
  else:
    return ['ffmpeg']

def exit_with_error( error_str ):
  sys.stdout.write( '\n\nERROR: ' + error_str + '\n\n' )
  sys.stdout.flush()
  sys.exit(0)

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Ingest new videos",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="",
                      help="Input directory to ingest")

  parser.add_argument("-d", dest="input_dir", default="",
                      help="Input directory to ingest")


  args = parser.parse_args()

  # Error checking
  process_data = True

  number_input_args = sum(len(inp_x) > 0 for inp_x in [args.input_video, args.input_dir, args.input_list])
  if number_input_args == 0:
    if not args.build_index and not args.detection_plots:
      exit_with_error( "Either input video or input directory must be specified" )
    else:
      process_data = False

  elif number_input_args > 1:
    exit_with_error( "Only one of input video, directory, or list should be specified, not more" )

  signal.signal( signal.SIGINT, signal_handler )

  # Initialize database
  if args.init_db:
    database_tool.init()

  if process_data:

    # Identify all videos to process
    if len( args.input_list ) > 0:
      video_list = split_image_list(args.input_list, args.gpu_count, args.output_directory)
      is_image_list = True
    elif len( args.input_dir ) > 0:
      video_list = list_files_in_dir( args.input_dir )
      is_image_list = False
    else:
      video_list = [args.input_video]
      is_image_list = False

    if len( video_list ) == 0:
      exit_with_error( "No videos found for ingest in given folder, exiting.\n" )
    elif not is_image_list:
      print( "\nProcessing " + str( len( video_list ) ) + " videos\n" )

    # Get required paths
    pipeline_loc = args.pipeline

    if len( args.output_directory ) > 0:
      create_dir( args.output_directory )
      sys.stdout.write( "\n" )

    if len( args.log_directory ) > 0:
      create_dir( args.output_directory + div + args.log_directory )
      sys.stdout.write( "\n" )

    # Process videos in parallel, one per GPU
    video_queue = queue.Queue()
    for video_name in video_list:
      if os.path.isfile( video_name ):
        video_queue.put( video_name )
      else:
        print( "Skipping " + video_name )

    def process_video_thread( gpu ):
      while True:
        try:
          video_name = video_queue.get_nowait()
        except queue.Empty:
          break
        process_video_kwiver( video_name, args, is_image_list, gpu=gpu )

    threads = [threading.Thread(target=process_video_thread, args=(gpu,))
               for gpu in range(args.gpu_count)]

    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    if is_image_list:
      for image_list in video_list:  # Clean up after split_image_list
        os.unlink(image_list)

    if not video_queue.empty():
      exit_with_error("Some videos were not processed!")

  # Build out final analytics
  if args.detection_plots:
    print( "Generating data plots" )
    generate_detection_plots.aggregate_plot( args.output_directory,
                                    args.objects.split(","),
                                    float( args.plot_threshold ),
                                    float( args.frame_rate ),
                                    int( args.smooth ) )

  # Build index
  if args.build_index:
    print( "\n\nBuilding searchable index\n" )
    if args.ball_tree:
      database_tool.build_balltree_index( remove_quotes( args.install_dir ) )
    else:
      database_tool.build_standard_index( remove_quotes( args.install_dir ) )

  # Output complete message
  print( "\n\nIngest complete, close this window before launching the query GUI.\n" )
