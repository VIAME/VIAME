#!/usr/bin/env python

import sys
import os
import shutil
import argparse
import glob

# Main Function
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Perform a filtering action on a csv",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", dest="input_file", default="",
                      help="Input track csv or glob pattern to process")

    parser.add_argument("--consolidate-ids", dest="consolidate_ids", action="store_true",
                      help="Use a ball tree for the searchable index")

    parser.add_argument("--decrease-fid", dest="decrease_fid", action="store_true",
                      help="Use a ball tree for the searchable index")

    args = parser.parse_args()

    input_files = []

    if len( args.input_file ) == 0:
        print( "NO INPUT FILE, EXITING" )
        sys.exit(0)

    if '*' in args.input_file:
        input_files = glob.glob( args.input_file )
    else:
        input_files.append( args.input_file )

    for input_file in input_files:

        print( "Processing " + input_file + "\n" )

        fin = open( input_file, "r" )
        output = []

        for line in fin:
            if len( line ) > 0 and line[0] == '#' or line[0:9] == 'target_id':
                continue
            parsed_line = line.rstrip().split(',')
            if len( parsed_line ) < 2:
                continue
            if args.consolidate_ids:
                parsed_line[0] = str( 100 * int( int( parsed_line[0] ) / 100 ) )
            if args.decrease_fid:
                parsed_line[2] = str( int( parsed_line[2] ) - 1 )
            output.append( ','.join( parsed_line ) + '\n' )
        fin.close()

        fout = open( input_file, "w" )
        for line in output:
            fout.write( line )
        fout.close()
  
