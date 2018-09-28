#!/usr/bin/env python

import datetime
import os
import os.path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def fish_aggregate(directory, species, threshold, frame_rate, smooth=1):
    def format_x(x, pos):
        t = datetime.timedelta(seconds=x)
        return str(t)

    videos = dict()
    video_plots = dict()

    for filename in os.listdir(directory):
        if filename.endswith(".csv") and not filename.endswith(".output.csv"):
            fig, ax = video_plots[filename] = plt.subplots()

            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_x))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set(xlabel="Time", ylabel="Fish Count", title="Fish Count (%s)" % filename)
            ax.grid()

            video_species = videos[filename] = dict()
            for s in species:
                video_species[s] = dict()
            with open(os.path.join(directory, filename), "r") as f:
                for line in f:
                    line = line.rstrip()
                    if line[0] != "#":
                        columns = line.split(",")
                        frame_id = int(columns[2])
                        for s in species:
                            if frame_id not in video_species[s]:
                                video_species[s][frame_id] = 0

                        detection_columns = columns[9:]
                        name = None
                        for column in detection_columns:
                            if name is not None:
                                if name in species:
                                    value = float(column)
                                    if value >= threshold:
                                        video_species[name][frame_id] += 1
                                name = None
                            else:
                                name = column

            for s in species:
                smoothed_video_frames = dict()
                for frame_id in sorted(video_species[s]):
                    lower_bound = frame_id - smooth // 2
                    upper_bound = lower_bound + smooth

                    max_count = video_species[s][frame_id]
                    for i in range(lower_bound, upper_bound):
                        try:
                            val = video_species[s][i]
                            if val > max_count:
                                max_count = val
                        except KeyError:
                            pass

                    smoothed_video_frames[frame_id] = max_count

                video_species[s] = smoothed_video_frames

    for s in species:
        sorted_frames = list()
        with open(os.path.join(directory, s + ".output.csv"), "w") as outfile:
            outfile.write("#video_id,frame_id,detection_count\n")
            for filename in sorted(videos):
                video_species = videos[filename]
                times = list()
                fish_counts = list()
                for frame_id in sorted(video_species[s]):
                    times.append(frame_id / frame_rate)
                    fish_counts.append(video_species[s][frame_id])
                    outfile.write(filename + "," + str(frame_id) + "," + str(video_species[s][frame_id]) + "\n")

                    sorted_frames.append((filename, frame_id, video_species[s][frame_id]))

                x = np.array(times)
                y = np.array(fish_counts)

                fig, ax = plt.subplots()
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_x))
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                plt.locator_params(axis='x', nbins=6)
                ax.set(xlabel="Time", ylabel="Fish Count", title="Fish Count (%s) (%s)" % (s, filename))
                ax.grid()

                ax.plot(x, y)
                ax.set_ylim(ymin=0)
                ax.set_xlim(xmin=0)
                ax.locator_params(axis='x', nbins=7)
                fig.savefig(os.path.join(directory, filename + "." + s + ".png"))

                fig, ax = video_plots[filename]
                ax.plot(x, y, label=s)

        sorted_frames.sort(key=lambda line: line[2], reverse=True)
        with open(os.path.join(directory, s + ".sorted.output.csv"), "w") as outfile:
            outfile.write("#video_id,frame_id,detection_count\n")
            for filename, frame_id, count in sorted_frames:
                outfile.write(filename + "," + str(frame_id) + "," + str(count) + "\n")

    for filename in video_plots:
        fig, ax = video_plots[filename]
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        ax.locator_params(axis='x', nbins=7)
        ax.legend()
        fig.savefig(os.path.join(directory, filename + ".png"))


if __name__ == "__main__":
    try:
        smooth = int(sys.argv[4])
    except IndexError:
        smooth = 1
    fish_aggregate(".", sys.argv[1].split(","), float(sys.argv[2]), float(sys.argv[3]), smooth)
