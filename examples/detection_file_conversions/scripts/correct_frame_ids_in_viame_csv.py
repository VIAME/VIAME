#!/usr/bin/env python

import argparse
import csv
import os
import sys

F_IMAGE_NAME = 1
F_FRAME_NUMBER = 2

field_types = [
    int,
    str,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
]

#------------------------------------------------------------------------------
def warn(msg, *args, **kwargs):
    sys.stderr.write('Warning: ' + msg.format(*args, **kwargs) + '\n')

#------------------------------------------------------------------------------
def read_image_list(path):
    images = {}
    i = 0

    with open(path, 'rt') as f:
        for l in f:
            images[os.path.basename(l.strip())] = i
            i += 1

    return images

#------------------------------------------------------------------------------
def read_records(path):
    records = []

    with open(path, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                record = [t(v) for t, v in zip(field_types, row)]

                classifiers = row[len(field_types):]
                while len(classifiers) >= 2:
                    record += classifiers[:2]
                    classifiers = classifiers[2:]

                if len(classifiers):
                    warn('ignoring unpaired classification {!r}',
                         classifiers[0])

                records.append(record)

            except ValueError:
                warn('ignoring row {!r} with malformatted field(s)', row)

    return records

#------------------------------------------------------------------------------
def write_records(records, out_file):
    writer = csv.writer(out_file, quoting=csv.QUOTE_NONNUMERIC)
    map(writer.writerow, records)

#------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Fix bad frame numbers in a NOAA CSV file '
                    'using an image list')

    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument('-o', '--output', type=str, metavar='OUTPUT',
                              help='Output CSV file')
    output_group.add_argument('-i', '--in-place', action='store_true',
                              help='Rewrite CSV file in place')

    parser.add_argument('images', metavar='IMAGES', type=str,
                        help='Input image list')
    parser.add_argument('csv', metavar='INPUT', type=str,
                        help='Input CSV file')

    args = parser.parse_args()

    # Read input files
    images = read_image_list(args.images)
    records = read_records(args.csv)

    # Fix record frame numbers using image names and image list
    for i in range(len(records)):
        f = records[i][F_IMAGE_NAME]
        if f in images:
            records[i][F_FRAME_NUMBER] = images[f]
        else:
            warn('no match for image name {!r}: frame number not updated', f)

    # Write output
    if args.in_place:
        out = open(args.csv, 'wt')
    elif args.output is not None:
        out = open(args.output, 'wt')
    else:
        out = sys.stdout

    write_records(records, out)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == '__main__':
    main()
