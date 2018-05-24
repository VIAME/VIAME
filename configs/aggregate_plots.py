#!/usr/bin/env python

import datetime
import os
import os.path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def fish_aggregate(directory, species, threshold, frame_rate, smooth=1):
    videos = dict()

    outfiles = { "output.csv", "output_sorted.csv" }

    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename not in outfiles:
            video_frames = videos[filename] = dict()
            with open(os.path.join(directory, filename), "r") as f:
                for line in f:
                    line = line.rstrip()
                    if line[0] != "#":
                        columns = line.split(",")
                        frame_id = int(columns[2])
                        try:
                            count = video_frames[frame_id]
                        except KeyError:
                            count = 0

                        detection_columns = columns[9:]
                        name = None
                        for column in detection_columns:
                            if name is not None:
                                if name == species:
                                    value = float(column)
                                    if value >= threshold:
                                        count += 1
                                name = None
                            else:
                                name = column

                        video_frames[frame_id] = count

            smoothed_video_frames = dict()
            for frame_id in sorted(video_frames):
                lower_bound = frame_id - smooth // 2
                upper_bound = lower_bound + smooth

                max_count = video_frames[frame_id]
                for i in range(lower_bound, upper_bound):
                    try:
                        val = video_frames[i]
                        if val > max_count:
                            max_count = val
                    except KeyError:
                        pass

                smoothed_video_frames[frame_id] = max_count

            videos[filename] = smoothed_video_frames

    def format_x(x, pos):
        t = datetime.timedelta(seconds=x)
        return str(t)

    sorted_frames = list()
    with open(os.path.join(directory, "output.csv"), "w") as outfile:
        outfile.write("#video_id,frame_id,detection_count\n")
        for filename in sorted(videos):
            video_frames = videos[filename]
            times = list()
            fish_counts = list()
            for frame_id in sorted(video_frames):
                times.append(frame_id / frame_rate)
                fish_counts.append(video_frames[frame_id])
                outfile.write(filename + "," + str(frame_id) + "," + str(video_frames[frame_id]) + "\n")

                sorted_frames.append((filename, frame_id, video_frames[frame_id]))

            x = np.array(times)
            y = np.array(fish_counts)

            fig, ax = plt.subplots()
            plt.xticks(rotation=20)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_x))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.plot(x, y)

            ax.set(xlabel="Time", ylabel="Fish Count", title="Fish Count (%s)" % filename)
            ax.grid()

            fig.savefig(os.path.join(directory, filename + ".png"))

    sorted_frames.sort(key=lambda line: line[2], reverse=True)
    with open(os.path.join(directory, "output_sorted.csv"), "w") as outfile:
        outfile.write("#video_id,frame_id,detection_count\n")
        for filename, frame_id, count in sorted_frames:
            outfile.write(filename + "," + str(frame_id) + "," + str(count) + "\n")


if __name__ == "__main__":
    try:
        smooth = int(sys.argv[4])
    except IndexError:
        smooth = 1
    fish_aggregate(".", sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), smooth)
