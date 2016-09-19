#!/usr/bin/perl
#
# Script to convert scallop-tk annotations to KWIVER csv detection format
#
# typical input looks like the following:
#
# 1,201503.20150517.074908728.9514.png,1042.6666666666667,495.75,1136,647.75,1.0,Fish,1.0
# 1,201503.20150517.074908728.9514.png,1033.3333333333333,498.4166666666667,1141.3333333333333,643.75,1.0,Sand-eel,1.0
# 2,201503.20150517.074921957.9593.png,458.6666666666667,970.4166666666666,521.3333333333334,1021.0833333333334,1.0,(527),1.0
# 3,201503.20150517.075335165.11105.png,1242.6666666666667,378.4166666666667,1353.3333333333333,534.4166666666666,1.0,Sand-eel,1.0
# 3,201503.20150517.075335165.11105.png,1285.3333333333333,554.4166666666666,1370.6666666666667,773.0833333333334,1.0,(527),1.0
# 4,201503.20150517.075559764.11969.png,724,161.08333333333334,857.3333333333334,243.75,1.0,Sand-eel,1.0
# 4,201503.20150517.075559764.11969.png,569.3333333333334,545.0833333333334,722.6666666666666,606.4166666666666,1.0,Sand-eel,1.0
#
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

my $opt_cache_only = 0;

my $point_delta = 50;

my $track_id = 1;
my $next_frame_index = 1;

my %image_dict;

# ------------------------------------------------------------------
my $out_file;
my $in_file;

while ( $ARGV[0] =~ /^-/)
{
    &usage                  if $ARGV[0] =~ /-h/;
    $opt_cache_only = 1     if $ARGV[0] =~ /-cache/;

    if ($ARGV[0] =~ /-write/) {
        $out_file = $ARGV[1];        shift @ARGV;
    }
    if ($ARGV[0] =~ /-read/) {
        $in_file = $ARGV[1];        shift @ARGV;
    }

    shift @ARGV;
}

if (length($in_file) > 0)
{
    &read_file_index($in_file);
}

# ------------------------------------------------------------------
open(my $fhi, "<", $ARGV[0] ) or die "Cound not open file $ARGV[0]";

while (my $buf = <$fhi>)
{
    chomp $buf;
    my @line = split( ',', $buf );

    # [0] = image index
    # [0] = file name
    # [1, 2, 3, 4] = bounding box
    # [5] = class name
    # [7] = score
    # more class, score pairs

    # print "\n--- processing line: $_\n"; # test

    # get frame index for this image
    my $frame_idx = $image_dict{$line[0]};
    if ( length($frame_idx) == 0)
    {
        next if $opt_cache_only;

        # need to set up a new frame index
        $frame_idx = $next_frame_index;
        $image_dict{$line[0]} = $next_frame_index;
        $next_frame_index++;
    }

    # vpview frames start at 0 not 1
    $frame_idx = $frame_idx - 1;

    # if this is the same bbox, then skip the line
    if ( $line[8] == 1 )
    {
        my $ts = $frame_idx * 0.0333;
        print "$track_id 1 $frame_idx 0 0  0 0  0 0 $line[1] $line[2] $line[3] $line[4] 0  0 0 0  $ts -1\n";
        $track_id++;
    }
} # end while

close($fhi);

if (length($out_file))
{
    &write_file_index($out_file);
}



# ----------------------------------------------------------------
sub write_file_index {
    my ($filename) = @_;

    open( my $fh, ">", $out_file ) or die "Cant open file $out_file";
    # dump image dictionary
    foreach my $name (keys %image_dict)
    {
        print $fh "$name   $image_dict{$name}\n";
    }

    close($fh);
}


# ----------------------------------------------------------------
sub read_file_index {
    my ($filename) = @_;

    open( my $fh, "<", $filename ) or die "Can't open file $filename";
    my $counter = 1;
    while (my $line = <$fh>)
    {
        chomp $line;
        my @parts = split( ' ', $line );
        if( length( $parts[1] ) gt 0 )
        {
          $image_dict{$parts[0]} = $parts[1];
        }
        else
        {
          $image_dict{$parts[0]} = $counter;
        }
        $counter = $counter + 1;
    }

    close($fh);
}


# ----------------------------------------------------------------
sub usage {
    print "Usage: detection_set_to_kw18.pl [opts] file \n";
    print "  Options:\n";
    print "    --help                     print usage\n";
    print "    --out_file   file-name     Write image file/index correspondence to file\n";
    print "    --in_file    file-name     Read image file/index correspondence to file\n";
    print "    --cache-only               With --in-file, does not add process images unless they are already in cache\n";
}
