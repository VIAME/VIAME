#!/usr/bin/perl
#
# Script to convert WHOI annotations to KWIVER csv detection format
#
# typical input looks like the following:
#
# 201503.20150517.074921957.9593.png 469 201501
# 201503.20150517.074921957.9593.png 527 201501 boundingBox 458.6666666666667 970.4166666666666 521.3333333333334 1021.0833333333334
#
# Output format for kw18
#
# Column(s) 1: Track-id
# Column(s) 2: Track-length (# of detections)
# Column(s) 3: Frame-number (-1 if not available)
# Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)
# Column(s) 6-7: Velocity(x,y)
# Column(s) 8-9: Image-loc(x,y)
# Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left & bottom-right vertices)
# Column(s) 14: Area
# Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when available)
# Column(s) 18: Timesetamp(-1 if not available)
# Column(s) 19: Track-confidence(-1_when_not_available)
#
# Since these are just detections, tracks are one state long.
# Whereas these are indivdual images, the frame number is tied to an image.
#
# output is written to standard out
#
#

use strict;
use List::Util qw[min max];
use Math::Complex;

my $topx = 99999999;
my $topy = 99999999;

# ------------------------------------------------------------------
my $out_file;
my $in_file;

while ( $ARGV[0] =~ /^-/)
{
    &usage                  if $ARGV[0] =~ /-h/;

    if ($ARGV[0] =~ /-topx/) {
        $topx = $ARGV[1];        shift @ARGV;
    }
    if ($ARGV[0] =~ /-topy/) {
        $topy = $ARGV[1];        shift @ARGV;
    }

    shift @ARGV;
}


# ------------------------------------------------------------------
open(my $fhi, "<", $ARGV[0] ) or die "Cound not open file $ARGV[0]";

while (my $buf = <$fhi>)
{
    chomp $buf;
    my @line = split( ' ', $buf );

    next if ( $#line < 4 );
    # print "\n--- processing line: $buf\n"; # test

    my $adj_min_x = $line[9];
    my $adj_min_y = $line[10];
    my $adj_max_x = $line[11];
    my $adj_max_y = $line[12];

    if( $adj_min_x < 0 )
    {
      $adj_min_x = 0;
    }

    if( $adj_min_y < 0 )
    {
      $adj_min_y = 0;
    }

    if( $adj_max_x > $topx )
    {
      $adj_max_x = $topx;
    }

    if( $adj_max_y > $topy )
    {
      $adj_max_y = $topy;
    }

    print"$line[0] $line[1] $line[2] $line[3] $line[4] $line[5] $line[6] ";
    print"$line[7] $line[8] $adj_min_x $adj_min_y $adj_max_x ";
    print"$adj_max_y $line[13] $line[14] $line[15] $line[16] $line[17] ";
    print"$line[18]\n";

} # end while

close($fhi);


# ----------------------------------------------------------------
sub usage {
    print "Usage: habcam_to_kw18.pl [opts] file \n";
    print "  Options:\n";
    print "    --help                     print usage\n";
    print "    --topx number              Maximum x (col) value\n";
    print "    --topy number              Maximum y (row) value\n";
}
