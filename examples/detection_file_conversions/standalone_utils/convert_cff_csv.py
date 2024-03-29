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

    parser.add_argument("-width", dest="image_width", default="1280",
                      help="Input image width")
    parser.add_argument("-height", dest="image_height", default="1024",
                      help="Input image height")

    parser.add_argument("--ff-only", dest="ff_only", action="store_true",
                      help="Output full frame only annotations")

    parser.add_argument("--fn-only", dest="fn_only", action="store_true",
                      help="Output unique image names only")

    args = parser.parse_args()

    input_files = []

    if len( args.input_file ) == 0:
        print( "No valid input files provided, exiting." )
        sys.exit(0)

    if '*' in args.input_file:
        input_files = glob.glob( args.input_file )
    else:
        input_files.append( args.input_file )

    default_width = int( args.image_width )
    default_height = int( args.image_height )

    default_poly = [ 0, 0, 0, default_height,
      default_width, default_height, default_width, 0 ]

    for input_file in input_files:

        print( "Processing " + input_file )

        input_no_ext = os.path.splitext( input_file )[0]

        output_file = input_no_ext + ".converted.csv"
        output_list = input_no_ext + ".list.txt"

        fin = open( input_file, "r" )

        parsed_csv = csv.reader( fin, quotechar='"', delimiter=',',
          quoting=csv.QUOTE_ALL, skipinitialspace=True )

        annotations = dict()

        for line in list( parsed_csv ):
            if len( line ) > 0 and line[0][0] == '#':
                continue
            if line[0][0:5] == 'Image':
                continue
            if len( line ) < 4:
                continue

            image_name = line[0]

            if image_name not in annotations:
                annotations[ image_name ] = []

            if len( line ) < 11:
                continue

            substrate = line[4].strip().replace( ",", "|" )
            substrate = substrate.replace( " ", "_" )

            if "unreadable" in substrate:
                continue

            biotic = line[6].strip().replace( ",", "|" )
            biotic = biotic.replace( " ", "_" )

            if len( line ) < 12 or line[11] == "NA" or line[11] == "":
                poly = default_poly
            else:
                poly_list = line[11].split( "," )
                if "" in poly_list:
                    poly_list.remove( "" )
                poly = [ int( float( i ) + 0.5 ) for i in poly_list ]

            annotations[ image_name ].append( [ substrate, poly ] )

            if biotic != "NA" and biotic != "":
                annotations[ image_name ].append( [ biotic, poly ] )

        fin.close()

        if args.ff_only:
            for index, image in enumerate( annotations ):
                uid = []
                for cat in annotations[ image ]:
                    uid.extend( cat[0].split("|") )
                res = []
                [ res.append(x) for x in uid if x not in res ]
                annotations[ image ] = []
                for cat in res:
                    annotations[ image ].append( [ cat, default_poly ] )

        if not args.fn_only:
            fout = open( output_file, "w" )

        flist = open( output_list, "w" )

        det_counter = 0
        for index, image in enumerate( annotations ):
            flist.write( image + "\n" )

            if args.fn_only:
                continue

            for poly in annotations[ image ]:
                cat = poly[0]

                if not cat:
                    continue
                if cat[0] == "_":
                    cat = cat[1:]

                x_coord = poly[1][0::2]
                y_coord = poly[1][1::2]

                fout.write( str( det_counter ) + "," )
                fout.write( image + "," )
                fout.write( str( index ) + "," )
                fout.write( str( min( x_coord ) ) + "," )
                fout.write( str( min( y_coord ) ) + "," )
                fout.write( str( max( x_coord ) ) + "," )
                fout.write( str( max( y_coord ) ) + "," )
                fout.write( "1.0," )
                fout.write( "-1," )
                fout.write( poly[0] + ",1.0" )

                if not args.ff_only:
                    fout.write( ",(poly) " + " ".join( [ str( i ) for i in poly[1] ] ) )

                fout.write( "\n" )

                det_counter = det_counter + 1

        if not args.fn_only:
            fout.close()

        flist.close()
