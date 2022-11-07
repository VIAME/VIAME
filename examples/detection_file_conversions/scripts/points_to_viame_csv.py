#!/usr/bin/env python

import sys
import os
import csv
import shutil
import argparse
import glob

# Main Function
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Perform a filtering action on a csv",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", dest="input_file", default="",
                      help="Input file or glob pattern to process")

    parser.add_argument("-width", dest="box_width", default="20",
                      help="Box width to place around points")
    parser.add_argument("-height", dest="box_height", default="20",
                      help="Box height to place around points")

    parser.add_argument("-type", dest="default_type", default="head",
                      help="Default object category to assign")

    args = parser.parse_args()
    input_files = []

    if len( args.input_file ) == 0:
        print( "No valid input files provided, exiting." )
        sys.exit(0)

    if '*' in args.input_file:
        input_files = glob.glob( args.input_file )
    else:
        input_files.append( args.input_file )

    default_width_half = float( args.box_width ) + 0.5
    default_height_half = float( args.box_height ) + 0.5

    for input_file in input_files:

        print( "Processing " + input_file )

        input_no_ext = os.path.splitext( input_file )[0]
        output_file = input_no_ext + ".converted.csv"
 
        fin = open( input_file, "r" )

        annotations = dict()

        for line in fin.readlines():
            parsed_line = line.rstrip().split( ';' )

            if len( parsed_line ) < 4:
                continue
            if parsed_line[0] == '#' or parsed_line[0][0:2] == 'id':
                continue

            image_id = int( line[1] ) - 1

            if image_id not in annotations:
                annotations[ image_id ] = []

            box = [ int( float( parsed_line[2] ) - default_width_half ),
                    int( float( parsed_line[3] ) - default_height_half ),
                    int( float( parsed_line[2] ) + default_width_half ),
                    int( float( parsed_line[3] ) + default_height_half ) ]

            annotations[ image_name ].append( [ args.default_type, box ] )

        fin.close()

        fout = open( output_file, "w" )
        det_counter = 0

        for index, image_id in enumerate( annotations ):
            for annot in annotations[ image ]:
                cat = annot[0]
                if not cat:
                    continue
                if cat[0] == "_":
                    cat = cat[1:]

                fout.write( str( det_counter ) + "," )
                fout.write( "," )
                fout.write( str( image_id ) + "," )
                fout.write( str( annot[1][0] ) + "," )
                fout.write( str( annot[1][1] ) ) + "," )
                fout.write( str( annot[1][2] ) + "," )
                fout.write( str( annot[1][3] ) + "," )
                fout.write( "1.0," )
                fout.write( "-1," )
                fout.write( cat + ",1.0" )
                fout.write( "\n" )

                det_counter = det_counter + 1
        fout.close()
