#!/usr/bin/perl
#
# Script to convert WHOI annotations to KWIVER csv detection format
#
# typical input looks like the following:
#
# 201503.20150517.074921957.9593.png 469 201501
# 201503.20150517.074921957.9593.png 527 201501 boundingBox 458.6666666666667 970.4166666666666 521.3333333333334 1021.0833333333334
#
# usage:
#
# annotation_converter.pl annotation-file
#
# output is written to standard out
#

use strict;
use List::Util qw[min max];
use Math::Complex;

my $last_file;
my $last_index = 0;
my $point_delta = 50;

my %species_table = (
    185 => "Live Scallop",
    197 => "Live Scallop",
    207 => "Live Scallop",
    211 => "Live Scallop",
    515 => "Live Scallop",
    912 => "Live Scallop",
    919 => "Live Scallop",
    920 => "Live Scallop",
    188 => "Dead Scallop",
    403 => "Sand-eel",
    524 => "Skate",
    533 => "Fish",
    1003 => "Fish",
    1001 => "Fish",
    158 => "Crab",
    258 => "Crab"
    );

# ------------------------------------------------------------------
while (<>)
{
    chop;
    my @line = split( ' ', $_ );

    next if ( $#line < 4 );
    # print "\n--- processing line: $_\n"; # test

    # see if we are still looking at the same image
    if ( $line[0] ne $last_file )
    {
        $last_file = $line[0];
        $last_index++;
    }

    my $class = &species_name( $line[1] );

    if ( $line[3] eq "boundingBox" )
    {
        print "$last_index,$last_file,$line[4],$line[5],$line[6],$line[7],1.0,$class,1.0   # box\n";
    }
    elsif ( $line[3] eq "line" )
    {
        my $min_x = min (&to_num($line[4]), &to_num($line[6]) );
        my $max_x = max (&to_num($line[4]), &to_num($line[6]) );
        my $min_y = min (&to_num($line[5]), &to_num($line[7]) );
        my $max_y = max (&to_num($line[5]), &to_num($line[7]) );

        my $cx = ( $min_x + $max_x ) /2;
        my $cy = ( $min_y + $max_y ) /2;
        my $r = sqrt( ($min_x - $cx )**2 + ( $min_y - $cy)**2 );

        $min_x = $cx - $r;
        $max_x = $cx + $r;
        $min_y = $cy - $r;
        $max_y = $cy + $r;

        print "$last_index,$last_file,$min_x,$min_y,$max_x,$max_y,1.0,$class,1.0   # line (R=$r)\n";
    }
    elsif ( $line[3] eq "point" )
    {
        my $min_x = $line[4] - $point_delta;
        my $max_x = $line[4] + $point_delta;
        my $min_y = $line[5] - $point_delta;
        my $max_y = $line[5] + $point_delta;

        print "$last_index,$last_file,$min_x,$min_y,$max_x,$max_y,1.0,$class,1.0    # point ($point_delta)\n";
    }
    else
    {
        print "Unrecognized construct: $line[3]\n";
    }

} # end while


sub to_num {
    my  ($val) = @_;
    $val = oct($val) if $val =~ /^0/;
    $val += 0;
    return $val;
}

sub species_name {
    my ($id) = @_;

    $id = &to_num($id);
    my $name = $species_table{$id};
    if ( length($name) != 0)
    {
        $name = "$name($id)";
    }
    else
    {
        $name = "($id)";
    }

    return $name;
}
