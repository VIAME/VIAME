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
    my @line = split( ' ', $buf );

    # get frame index for this image
    my $frame_idx = $image_dict{$line[0]};

    if ( length($frame_idx) == 0)
    {
        next if $opt_cache_only == 1;

        # need to set up a new frame index
        $frame_idx = $next_frame_index;
        $image_dict{$line[0]} = $next_frame_index;
        $next_frame_index++;
    }

    # vpview frames start at 0 not 1
    $frame_idx = $frame_idx - 1;

    next if ( $#line < 4 );
    # print "\n--- processing line: $buf\n"; # test

    my $ts = $frame_idx * 0.0333;

    if ( $line[3] eq "boundingBox" )
    {
        print "$track_id 1 $frame_idx 0 0  0 0  0 0 $line[4] $line[5] $line[6] $line[7] 0  0 0 0  $ts -1\n";
        $track_id++;
    }
    elsif ( $line[3] eq "line" )
    {
        my $min_x = min ($line[4], $line[6]);
        my $max_x = max ($line[4], $line[6]);
        my $min_y = min ($line[5], $line[7]);
        my $max_y = max ($line[5], $line[7]);

        my $cx = ( $min_x + $max_x ) /2;
        my $cy = ( $min_y + $max_y ) /2;
        my $r = sqrt( ($min_x - $cx )**2 + ( $min_y - $cy)**2 );

        $min_x = $cx - $r;
        $max_x = $cx + $r;
        $min_y = $cy - $r;
        $max_y = $cy + $r;

        print"$track_id 1 $frame_idx 0 0  0 0  0 0  $min_x $min_y $max_x $max_y 0  0 0 0 $ts -1\n";
        $track_id++;
    }
    elsif ( $line[3] eq "point" )
    {
        my $min_x = $line[4] - $point_delta;
        my $max_x = $line[4] + $point_delta;
        my $min_y = $line[5] - $point_delta;
        my $max_y = $line[5] + $point_delta;

        print"$track_id 1 $frame_idx 0 0  0 0  0 0  $min_x $min_y $max_x $max_y 0  0 0 0 $ts -1\n";
        $track_id++;
    }
    else
    {
        print "Unrecognized construct: $line[3]\n";
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
    print "Usage: habcam_to_kw18.pl [opts] file \n";
    print "  Options:\n";
    print "    --help                     print usage\n";
    print "    --write-file file-name     Write image file/index correspondence to file\n";
    print "    --read-file  file-name     Read image file/index correspondence to file\n";
    print "    --cache-only               With --in-file, does not add process images unless they are already in cache\n";
}
