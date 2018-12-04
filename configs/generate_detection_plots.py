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

def detection_plot( input_directory, output_directory, objects, threshold, frame_rate,
                    smooth=1, ext=".csv", net_category="all_fish", top_category_only=False ):

  def format_x( x, pos ):
    t = datetime.timedelta( seconds = x )
    return str( t )

  warnings.filterwarnings( "ignore" )

  videos = dict()
  video_plots = dict()

  for filename in os.listdir( input_directory ):

    if not filename.endswith( ext ) or filename.endswith( ".output.csv" ):
      continue

    fig, ax = video_plots[filename] = plt.subplots()

    plot_title = "Aggregate - " + filename

    ax.xaxis.set_major_formatter( matplotlib.ticker.FuncFormatter( format_x ) )
    ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator( integer=True ) )
    ax.set( xlabel="Time", ylabel="Object Count", title=plot_title )
    ax.grid()

    video_objects = videos[filename] = dict()
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
          for column in detection_columns:
            if name is not None:
              if name in objects:
                value = float(column)
                if value >= threshold:
                  video_objects[name][frame_id] += 1
              name = None
            else:
              name = column

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

  for obj in objects:
    sorted_frames = list()
    with open( os.path.join( output_directory, obj + ".output.csv" ), "w" ) as outfile:
      outfile.write( "#video_id,frame_id,detection_count\n" )
      for filename in sorted(videos):
        video_objects = videos[filename]
        times = list()
        object_counts = list()
        for frame_id in sorted(video_objects[obj]):
          times.append(frame_id / frame_rate)
          object_counts.append(video_objects[obj][frame_id])
          outfile.write(filename + "," + str(frame_id) + "," + str(video_objects[obj][frame_id]) + "\n")

          sorted_frames.append((filename, frame_id, video_objects[obj][frame_id]))

        x = np.array(times)
        y = np.array(object_counts)

        plot_title = obj + " - " + filename

        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_x))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.locator_params(axis='x', nbins=6)
        ax.set(xlabel="Time", ylabel="Object Count", title=plot_title)
        ax.grid()

        ax.plot( x, y )
        ax.set_ylim( ymin = 0 )
        ax.set_xlim( xmin = 0 )
        if np.size( y ) > 0 and np.max( y ) < 5:
          ax.set_ylim( ymax = 5 )
        ax.locator_params( axis='x', nbins = 7 )
        fig.savefig( os.path.join( output_directory, filename + "." + obj + ".png" ) )

        fig, ax = video_plots[filename]
        ax.plot(x, y, label=obj)

    sorted_frames.sort(key=lambda line: line[2], reverse=True)
    with open( os.path.join( output_directory, obj + ".sorted.output.csv" ), "w" ) as outfile:
      outfile.write("#video_id,frame_id,detection_count\n")
      for filename, frame_id, count in sorted_frames:
        outfile.write(filename + "," + str(frame_id) + "," + str(count) + "\n")

  for filename in video_plots:
    fig, ax = video_plots[filename]
    ax.set_ylim( ymin = 0 )
    ax.set_xlim( xmin = 0 )
    ax.locator_params( axis='x', nbins = 7 )
    ax.legend()
    fig.savefig( os.path.join( output_directory, filename + ".png" ) )

if __name__ == "__main__":
  try:
    smooth = int( sys.argv[4] )
  except IndexError:
    smooth = 1
  detection_plot( ".", sys.argv[1].split(","),
                  float( sys.argv[2] ), float( sys.argv[3] ),
                  smooth )
