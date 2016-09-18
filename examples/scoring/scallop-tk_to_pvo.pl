#!/usr/bin/perl
#
# Script to convert scallop-tk annotations to KWIVER csv detection format
#
# typical input looks like the following:
#
# 201503.20150517.074320800.7435.png,664.306,2567.31,26.668,26.668,0,LIVE_SCALLOP,0.453392,1
# 201503.20150517.074320800.7435.png,664.306,2567.31,26.668,26.668,0,DEAD_SCALLOP,0.0062762,2
# 201503.20150517.074739861.8983.png,660.457,829.716,36.1975,35.1703,162.678,LIVE_SCALLOP,0.920905,1
# 201503.20150517.074739861.8983.png,660.457,829.716,36.1975,35.1703,162.678,DEAD_SCALLOP,0.000549866,2
# 201503.20150517.074908728.9514.png,682.994,2354.23,25.9739,25.9739,0,LIVE_SCALLOP,0.719811,1
# 201503.20150517.074908728.9514.png,682.994,2354.23,25.9739,25.9739,0,DEAD_SCALLOP,0.0103365,2
# 201503.20150517.075335165.11105.png,942.777,1112.58,15.8875,15.8875,0,LIVE_SCALLOP,0.975731,1
# 201503.20150517.075335165.11105.png,942.777,1112.58,15.8875,15.8875,0,DEAD_SCALLOP,1.4467e-06,2
# 201503.20150517.075335165.11105.png,142.871,82.6177,29.1586,29.1586,0,LIVE_SCALLOP,0.748576,1
# 201503.20150517.075335165.11105.png,142.871,82.6177,29.1586,29.1586,0,DEAD_SCALLOP,0.00941856,2
#
# Note that there can be multiple entries for same box with different classifications
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

    # [0] = file name
    # [1, 2, 3, 4] = bounding box
    # [5] = ?
    # [6] = class name
    # [7] = score
    # [8] = class rank

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

    # only process top rank class-name
    if ( $line[8] == 1 )
    {
        print "$track_id 0.0 $line[7] 0.0\n";
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
    while (my $line = <$fh>)
    {
        chomp $line;
        my @parts = split( ' ', $line );
        $image_dict{$parts[0]} = $parts[1];
    }

    close($fh);
}


# ----------------------------------------------------------------
sub usage {
    print "Usage: scallop-tk_to_kw18.pl [opts] file \n";
    print "  Options:\n";
    print "    --help                     print usage\n";
    print "    --write_file file-name     Write image file/index correspondence to file\n";
    print "    --read_file  file-name     Read image file/index correspondence to file\n";
    print "    --cache-only               With --in-file, does not add process images unless they are already in cache\n";
}
