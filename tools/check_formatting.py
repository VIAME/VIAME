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
      help="Input file or glob pattern to process")

    parser.add_argument("--consolidate-ids", dest="consolidate_ids", action="store_true",
      help="Use a ball tree for the searchable index")

    parser.add_argument("--decrease-fid", dest="decrease_fid", action="store_true",
      help="Use a ball tree for the searchable index")

    parser.add_argument("--increase-fid", dest="increase_fid", action="store_true",
      help="Use a ball tree for the searchable index")

    parser.add_argument("--assign-uid", dest="assign_uid", action="store_true",
      help="Assign unique detection ids to all entries in volume")

    parser.add_argument("--filter-single", dest="filter_single", action="store_true",
      help="Filter single state tracks")

    parser.add_argument("--print-types", dest="print_types", action="store_true",
      help="Print unique list of target types")

    parser.add_argument("--average-box-size", dest="average_box_size", action="store_true",
      help="Print average box size per type")

    args = parser.parse_args()

    input_files = []

    if len( args.input_file ) == 0:
        print( "No valid input files provided, exiting." )
        sys.exit(0)

    if '*' in args.input_file:
        input_files = glob.glob( args.input_file )
    else:
        input_files.append( args.input_file )

    id_counter = 1
    type_counts = dict()
    type_sizes = dict()

    for input_file in input_files:

        print( "Processing " + input_file )

        fin = open( input_file, "r" )
        output = []

        id_mappings = dict()
        id_states = dict()
        has_non_single = False

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
            if args.increase_fid:
                parsed_line[2] = str( int( parsed_line[2] ) + 1 )
            if args.assign_uid:
                if parsed_line[0] in id_mappings:
                    parsed_line[0] = id_mappings[ parsed_line[0] ]
                    has_non_single = True
                else:
                    id_mappings[parsed_line[0]] = str(id_counter)
                    parsed_line[0] = str(id_counter)
                    id_counter = id_counter + 1
            if args.filter_single:
                if parsed_line[0] not in id_states:
                    id_states[ parsed_line[0] ] = 1
                else:
                    id_states[ parsed_line[0] ] = id_states[ parsed_line[0] ] + 1
                    has_non_single = True
            if len( parsed_line ) > 9:
                if args.print_types or args.average_box_size:
                    if parsed_line[9] in type_counts:
                        type_counts[ parsed_line[9] ] += 1
                    else:
                        type_counts[ parsed_line[9] ] = 1
                if args.average_box_size:
                    box_width = float( parsed_line[5] ) - float( parsed_line[3] )
                    box_height = float( parsed_line[6] ) - float( parsed_line[4] )
                    if parsed_line[9] in type_sizes:
                        type_sizes[ parsed_line[9] ] += ( box_width * box_height )
                    else:
                        type_sizes[ parsed_line[9] ] = ( box_width * box_height )
            output.append( ','.join( parsed_line ) + '\n' )
        fin.close()

        if ( args.assign_uid or args.filter_single ) and not has_non_single:
            print( "Sequence " + input_file + " has all single states" )

        if args.filter_single:
            output = [ e for e in output if id_states[ e.split(',')[ 0 ] ] > 1 ]

        if args.filter_single or args.increase_fid or args.decrease_fid \
          or args.assign_uid or args.consolidate_ids or args.filter_single: 
            fout = open( input_file, "w" )
            for line in output:
                fout.write( line )
            fout.close()

    if args.print_types:
        print( ','.join( type_counts.keys() ) )

    if args.average_box_size:
        print( "Type - Average Box Area - Total Count" )
        for i in type_sizes:
            size_str = str( float( type_sizes[ i ] ) / type_counts[ i ] )
            print( i + " " + size_str + " " + str( type_counts[ i ] ) )
  
