#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


import sys
import os
import shutil
import argparse
import glob
import pathlib

def parse_fps( line ):
    if 'fps' not in line:
        return -1
    pos = line.find( 'fps' )
    found = False
    found_period = False
    output = ""
    for i in range( pos, len( line ) ):
        if line[i].isdigit():
            output = output + line[i]
            if not found:
                found = True
        elif found and line[i] == '.' and not found_period:
            output = output + line[i]
            found_period = True
        elif not found:
            continue
        else:
            break
    return float( output )

# Main Function
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Perform a filtering action on a csv",
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", dest="input_file", default="",
      help="Input file or glob pattern to process")

    parser.add_argument("--decrease-fid", dest="decrease_fid", action="store_true",
      help="Decrease frame IDs in files by 1")

    parser.add_argument("--increase-fid", dest="increase_fid", action="store_true",
      help="Increase frame IDs in files by 1")

    parser.add_argument("--assign-uid", dest="assign_uid", action="store_true",
      help="Assign unique detection ids to all entries in volume")

    parser.add_argument("--filter-single", dest="filter_single", action="store_true",
      help="Filter single state tracks")

    parser.add_argument("--print-types", dest="print_types", action="store_true",
      help="Print unique list of target types")

    parser.add_argument("--caps-only", dest="caps_only", action="store_true",
      help="Only print types with capitalized letters in them")

    parser.add_argument("--track-count", dest="track_count", action="store_true",
      help="Print total number of tracks")

    parser.add_argument("--counts-per-frame", dest="counts_per_frame", action="store_true",
      help="Print total number of detections per frame")

    parser.add_argument("--average-box-size", dest="average_box_size", action="store_true",
      help="Print average box size per type")

    parser.add_argument("--conf-threshold", dest="conf_threshold", type=float, default="-1.0",
      help="Confidence threshold")

    parser.add_argument("--type-threshold", dest="type_threshold", type=float, default="-1.0",
      help="Confidence threshold")

    parser.add_argument("--print-filtered", dest="print_filtered", action="store_true",
      help="Print out tracks that were filtered out")

    parser.add_argument("--print-single", dest="print_single", action="store_true",
      help="Print out video sequences only containing single states")

    parser.add_argument("--lower-fid", dest="lower_fid", type=int, default="0",
      help="Lower FID if adjusting FIDs to be within some range")

    parser.add_argument("--upper-fid", dest="upper_fid", type=int, default="0",
      help="Lower FID if adjusting FIDs to be within some range")

    parser.add_argument("--replace-file", dest="replace_file", default="",
      help="If set, replace all types in this file given their synonyms")

    parser.add_argument("--print-fps", dest="print_fps", action="store_true",
      help="Print FPS in input files")

    parser.add_argument("--comp-file", dest="comp_file", default="",
      help="If set, generate a comparison file contrasting types in all inputs")

    args = parser.parse_args()

    input_files = []

    if len( args.input_file ) == 0:
        print( "No valid input files provided, exiting." )
        sys.exit(0)

    if os.path.isdir( args.input_file ):
        for path in pathlib.Path( args.input_file ).rglob( "*.csv" ):
            input_files.append( str( path ) )
    elif '*' in args.input_file:
        input_files = glob.glob( args.input_file )
    else:
        input_files.append( args.input_file )

    if args.caps_only:
        args.print_types = True

    if args.print_single:
        args.track_count = True

    write_output = args.filter_single or args.increase_fid or \
      args.decrease_fid or args.assign_uid or args.replace_file or \
      args.filter_single or args.lower_fid or args.upper_fid

    id_counter = 1
    type_counts = dict()
    type_sizes = dict()
    type_ids = dict()
    repl_dict = dict()
    comp_info = dict()

    track_counter = 0
    state_counter = 0

    if args.replace_file:
        fin = open( args.replace_file, 'r' )
        if not fin:
            print( "Replace file: " + args.replace_file + " does not exist" )
        for line in fin:
            parsed = line.split( ',' )
            if len( line ) > 1:
                repl_dict[ parsed[0].rstrip() ] = parsed[1].rstrip()
            elif len( line.rstrip() ) > 0:
                print( "Error parsing line: " + line )
        fin.close()

    for input_file in input_files:

        if not args.print_single:
            if args.counts_per_frame:
                print( "# " + os.path.basename( input_file ) )
            elif args.print_fps:
                print( input_file, end="," )
            else:
                print( "Processing " + input_file )

        fin = open( input_file, "r" )
        output = []

        id_mappings = dict()
        id_states = dict()
        unique_ids = set()
        printed_ids = set()
        frame_counts = dict()
        seq_ids = dict()

        contains_track = False
        video_fps = 0

        for line in fin:
            if len( line ) > 0 and line[0] == '#' or line[0:9] == 'target_id':
                if args.print_fps and "fps" in line:
                    video_fps = parse_fps( line )
                output.append( line )
                continue
            parsed_line = line.rstrip().split(',')
            if len( parsed_line ) < 2:
                continue
            if args.conf_threshold > 0 and len( parsed_line ) > 7:
                if float( parsed_line[7] ) < args.conf_threshold:
                    if args.print_filtered and parsed_line[0] not in printed_ids:
                        print( "Id: " + parsed_line[0] + " filtered" )
                        printed_ids.add( parsed_line[0] )
                    continue

            if args.track_count:
                state_counter = state_counter + 1
                if parsed_line[0] not in unique_ids:
                    unique_ids.add( parsed_line[0] )
                else:
                    contains_track = True

            if args.decrease_fid:
                parsed_line[2] = str( int( parsed_line[2] ) - 1 )

            if args.increase_fid:
                parsed_line[2] = str( int( parsed_line[2] ) + 1 )

            if args.lower_fid > 0:
                if int( parsed_line[2] ) < args.lower_fid:
                    continue
                parsed_line[2] = str( int( parsed_line[2] ) - args.lower_fid )

            if args.upper_fid > 0:
                if int( parsed_line[2] ) > args.upper_fid - args.lower_fid:
                    continue

            if args.filter_single:
                if parsed_line[0] not in id_states:
                    id_states[ parsed_line[0] ] = 1
                else:
                    id_states[ parsed_line[0] ] = id_states[ parsed_line[0] ] + 1
                    has_non_single = True

            if len( parsed_line ) > 9:
                top_category = ""
                top_score = -100.0
                attr_start = -1

                for i in range( 9, len( parsed_line ), 2 ):
                    if len( parsed_line[i] ) == 0:
                        continue
                    if parsed_line[i][0] == '(':
                        attr_start = i
                        break
                    score = float( parsed_line[i+1] )
                    if score > top_score:
                        top_category = parsed_line[i]
                        top_score = score

                if args.type_threshold > 0:
                    if float( top_score ) < args.type_threshold:
                        if args.print_filtered and parsed_line[0] not in printed_ids:
                            print( "Id: " + parsed_line[0] + " filtered" )
                            printed_ids.add( parsed_line[0] )
                        continue

                if args.print_types or args.average_box_size:
                    if top_category in type_counts:
                        type_counts[ top_category ] += 1
                    else:
                        type_counts[ top_category ] = 1
                    if args.track_count:
                        if top_category in seq_ids:
                            seq_ids[ top_category ].add( parsed_line[0] )
                        else:
                            seq_ids[ top_category ] = { parsed_line[0] }

                if args.counts_per_frame:
                    if parsed_line[1] not in frame_counts:
                        frame_counts[ parsed_line[1] ] = { top_category:1 }
                    elif top_category not in frame_counts[ parsed_line[1] ]:
                        frame_counts[ parsed_line[1] ][ top_category ] = 1
                    else:
                        frame_counts[ parsed_line[1] ][ top_category ] += 1

                if args.average_box_size:
                    box_width = float( parsed_line[5] ) - float( parsed_line[3] )
                    box_height = float( parsed_line[6] ) - float( parsed_line[4] )
                    if top_category in type_sizes:
                        type_sizes[ top_category ] += ( box_width * box_height )
                    else:
                        type_sizes[ top_category ] = ( box_width * box_height )

                if args.replace_file:
                    new_cat = repl_dict[ top_category ] if top_category in repl_dict else top_category
                    new_score = str(1.0)
                    parsed_line[9] = new_cat
                    parsed_line[10] = new_score
                    if attr_start > 0:
                        attr_count = len( parsed_line ) - attr_start
                        for i in range( attr_count ):
                            parsed_line[ i + 11 ] = parsed_line[ i + attr_start ]
                        parsed_line = parsed_line[ :(11+attr_count) ]
                    elif len( parsed_line ) > 11:
                        parsed_line = parsed_line[ :11 ]

            if args.assign_uid:
                if parsed_line[0] in id_mappings:
                    parsed_line[0] = id_mappings[ parsed_line[0] ]
                    has_non_single = True
                else:
                    id_mappings[parsed_line[0]] = str(id_counter)
                    parsed_line[0] = str(id_counter)
                    id_counter = id_counter + 1

            if write_output:
                output.append( ','.join( parsed_line ) + os.linesep )

        fin.close()

        if seq_ids:
            type_ids[ input_file ] = seq_ids

        if args.print_fps:
            if video_fps > 0:
                print( video_fps )
            else:
                print( "unlisted" )

        if args.track_count:
            track_counter = track_counter + len( unique_ids )

        if ( args.assign_uid or args.filter_single ) and not has_non_single:
            print( "Sequence " + input_file + " has all single states" )

        if args.print_single and not contains_track:
            if len( unique_ids ) == 0:
                print( "Sequence " + input_file + " contains no detections" )
            else:
                print( "Sequence " + input_file + " contains only detections" )

        if args.filter_single:
            output = [ e for e in output if id_states[ e.split(',')[ 0 ] ] > 1 ]

        if args.counts_per_frame:
            for frame in frame_counts:
                frame_str = frame
                for cls in frame_counts[ frame ]:
                    frame_str += ", " + cls + "=" + str( frame_counts[ frame ][ cls ] )
                print( frame_str )

        if write_output: 
            fout = open( input_file, "w" )
            for line in output:
                fout.write( line )
            fout.close()

    if args.track_count:
        print( "Track count: " + str(track_counter) + " , states = " + str(state_counter) )

    if args.print_types:
        print( os.linesep + "Types found in files:" + os.linesep )

        def count_type( type_name ):
            output = 0
            for fn in type_ids.keys():
                if type_name in type_ids[fn]:
                    output = output + len( type_ids[fn][type_name] )
            return output

        for itm in sorted( type_counts.keys() ):
            if args.caps_only and not any( char.isupper() for char in itm ):
                continue
            if args.track_count:
                print( itm + " " + str( count_type( itm ) ) )
            else:
                print( itm )

    if args.comp_file:
        fout = open( args.comp_file, "w" )
        fout.write( "file_name" )
        for itm in sorted( type_counts.keys() ):
            fout.write( ", " + itm )
        fout.write( "\n" )
        for fn in sorted( type_ids ):
            fout.write( fn )
            for itm in sorted( type_counts.keys() ):
                if itm in type_ids[fn]:
                    fout.write( ", " + str( len( type_ids[fn][itm] ) ) )
                else:
                    fout.write( ", 0" )
            fout.write( "\n" )
        fout.close()

    if args.average_box_size:
        print( "Type - Average Box Area - Total Count" )
        for i in type_sizes:
            size_str = str( float( type_sizes[ i ] ) / type_counts[ i ] )
            print( i + " " + size_str + " " + str( type_counts[ i ] ) )
