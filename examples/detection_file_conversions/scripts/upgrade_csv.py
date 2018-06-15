
import csv
import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

fout = open( output_file, 'w' )

counter = 0

with open( input_file ) as ifile:
  for line in ifile:
    parsed_line = line.split(',')

    if len( parsed_line ) < 6:
      continue

    counter = counter + 1

    fout.write( str( counter ) )
    fout.write( ',' )
    fout.write( parsed_line[1] )
    fout.write( ',' )
    fout.write( parsed_line[0] )
    fout.write( ',' )
    fout.write( parsed_line[2] )
    fout.write( ',' )
    fout.write( parsed_line[3] )
    fout.write( ',' )
    fout.write( parsed_line[4] )
    fout.write( ',' )
    fout.write( parsed_line[5] )

    fout.write( ',0' )

    fout.write( ',' )
    fout.write( parsed_line[6] )

    for i in range( 7, len( parsed_line ) ):
      fout.write( ',' )
      fout.write( parsed_line[i] )

fout.close()
