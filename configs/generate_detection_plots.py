#!/usr/bin/env python

import datetime
import os
import os.path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import warnings

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

unordered_ext = ".time_ordered.csv"
ranked_ext = ".max_ordered.csv"

def detection_plot( input_directory, output_directory, objects, threshold, frame_rate,
                    smooth=1, ext=".csv", net_name="all_fish", top_category_only=False ):

  def format_x( x, pos=0, show_ms=False ):
    t = datetime.timedelta( seconds = x )
    split_str = str( t ).split(".")
    if show_ms:
      return split_str[0] + ( "." + split_str[1][0] if len( split_str ) > 1 else ".0" )
    else:
      return split_str[0]

  warnings.filterwarnings( "ignore" )

  if net_name in objects:
    print( "Plotting error, net category id can't be in object list" )
    sys.exit(0)
  else:
    objects = [ net_name ] + objects

  if input_directory == output_directory:
    print( "Plotting error, input and output directories must be different" )
    sys.exit(0)

  video_max_counts = dict()

  for obj in objects:
    video_max_counts[obj] = []

  create_dir( output_directory )

  for filename in os.listdir( input_directory ):

    if not filename.endswith( ext ):
      continue

    filebase = filename.replace( ext, "" )

    # Parse input computed detections file
    video_objects = dict()

    for obj in objects:
      video_objects[obj] = dict()

    with open( os.path.join( input_directory, filename ), "r" ) as f:
      for line in f:
        line = line.rstrip()
        if line[0] != "#":
          columns = line.split(",")
          frame_id = int( columns[2] )
          for obj in objects:
            if frame_id not in video_objects[obj]:
              video_objects[obj][frame_id] = 0

          detection_columns = columns[9:11] if top_category_only else columns[9:]
          name = None
          is_first = True
          for column in detection_columns:
            if name is not None:
              if name in objects:
                value = float(column)
                if value >= threshold:
                  video_objects[name][frame_id] += 1
                  if is_first:
                    video_objects[net_name][frame_id] += 1
                    is_first = False
              name = None
            else:
              name = column

    # Apply smoothing factor
    for obj in objects:
      smoothed_video_frames = dict()
      for frame_id in sorted( video_objects[obj] ):
        lower_bound = frame_id - smooth // 2
        upper_bound = lower_bound + smooth

        max_count = video_objects[obj][frame_id]
        for i in range( lower_bound, upper_bound ):
          try:
            val = video_objects[obj][i]
            if val > max_count:
              max_count = val
          except KeyError:
            pass

        smoothed_video_frames[frame_id] = max_count

      video_objects[obj] = smoothed_video_frames

    # Write out video specific items alongside aggregrate plot
    video_subdir = os.path.join( output_directory, filebase )
    create_dir( video_subdir )

    agr_fig, agr_ax = plt.subplots()
    agr_plot_title = "Aggregate - " + filename
    agr_ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter( format_x ) )
    agr_ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator( integer=True ) )
    agr_ax.set( xlabel="Time", ylabel="Object Count", title=agr_plot_title )
    agr_ax.grid()
    agr_max_y = 0

    for obj in objects:
      sorted_frames = list()
      with open( os.path.join( video_subdir, obj + unordered_ext ), "w" ) as of:
        of.write( "# video_id, time_id, frame_id, detection_count\n" )

        times = list()
        object_counts = list()

        for frame_id in sorted( video_objects[obj] ):
          frame_time = frame_id / frame_rate
          times.append( frame_time )
          object_counts.append( video_objects[obj][frame_id] )

          of.write( filename + "," )
          of.write( format_x(frame_time, show_ms=True) + "," )
          of.write( str(frame_id) + "," )
          of.write( str(video_objects[obj][frame_id]) + "\n" )

          sorted_frames.append( (filename, frame_id, video_objects[obj][frame_id]) )

        x = np.array( times )
        y = np.array( object_counts )

        plot_title = obj + " - " + filename

        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter(format_x) )
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(integer=True) )
        plt.locator_params( axis='x', nbins=6 )
        ax.set( xlabel="Time", ylabel="Object Count", title=plot_title )
        ax.grid()

        ax.plot( x, y )
        ax.set_ylim( ymin = 0 )
        ax.set_xlim( xmin = 0 )
        if np.size( y ) > 0 and np.max( y ) < 5:
          ax.set_ylim( ymax = 5 )
        ax.locator_params( axis='x', nbins = 7 )
        fig.savefig( os.path.join( video_subdir, filename + "." + obj + ".png" ) )

        agr_ax.plot( x, y, label=obj )
        if np.size( y ) > 0:
          agr_max_y = max( np.max( y ), agr_max_y )

        sorted_frames.sort( key=lambda line: line[2], reverse=True )
        if len( sorted_frames ) > 0:
          video_max_counts[obj].append( sorted_frames[0] )
        with open( os.path.join( video_subdir, obj + ranked_ext ), "w" ) as of:
          of.write( "# video_id, time_id, frame_id, detection_count\n" )
          for filename, frame_id, count in sorted_frames:
            frame_time = frame_id / frame_rate
            of.write( filename + "," + format_x(frame_time, show_ms=True) + ",")
            of.write( str(frame_id) + "," + str(count) + "\n" )

    agr_ax.set_ylim( ymin = 0 )
    agr_ax.set_xlim( xmin = 0 )
    if agr_max_y < 5:
      agr_ax.set_ylim( ymax = 5 )
    agr_ax.locator_params( axis='x', nbins = 7 )
    lgd = [ agr_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) ]
    agr_fig.set_size_inches( 10, 7 )
    agr_fig.savefig( os.path.join( output_directory, filename + ".png" ),
                     dpi=100, bbox_inches="tight", additional_artists=lgd )

  # Write out aggregate information across all videos
  for obj in objects:
    with open( os.path.join( output_directory, "max_counts-" + obj + ".csv" ), "w" ) as of:
      of.write( "# video_id, time_id, frame_id, detection_count\n" )
      for filename, frame_id, count in video_max_counts[obj]:
        frame_time = frame_id / frame_rate
        of.write( filename + "," + format_x( frame_time, show_ms=True ) + "," )
        of.write( str( frame_id ) + "," + str( count ) + "\n" )

if __name__ == "__main__":
  try:
    smooth = int( sys.argv[4] )
  except IndexError:
    smooth = 1
  detection_plot( ".", "plots", sys.argv[1].split(","),
                  float( sys.argv[2] ), float( sys.argv[3] ),
                  smooth )
