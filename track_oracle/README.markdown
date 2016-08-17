# track_oracle and the scoring framework

This readme gives a quick roadmap to the code in this directory, which
falls into two categories:

1. The data management and file I/O library, known as "track_oracle"

2. The scoring code built on top of track_oracle

For historical reasons, the scoring code is a subdirectory of the
track_oracle code.  For further information about the scoring code,
see [this file.](scoring_framework/README.markdown)  The rest of this
file briefly describes track_oracle.

## Overview

The track_oracle library loads track and event data from various file
formats (as of this writing, ten) and presents a unified interface to
the data for clients (such as the scoring code.)  The asbtracted data
layer allows clients to code against data scehmae, rather than file
formats, and provides for adapter layers to transparently supply
missing data without requiring changes to the core client code.

## Basic ideas

1. The data model is that semantically distinct track and frame data
element typess are distinct columns in a data "cloud"; instances of
tracks and frames are rows in the cloud using whichever columns are
required to satisfy their particular data formats.  Not every row
populates every column; for example, some rows may be track-level data
such as track ID or activity type, while other rows may be frame-level
information such as bounding boxes or timestamps.

2. Semantic data elements such as "timestamp" and "bounding box" are
identified in the code by two attributes:
    - Their C++ data type (int, double, vgl_box_2d<double>, etc.)
    - A text string naming the element ("timestamp_usecs")

    These attributes are combined in a `track_field` object, for
    example: `track_field<unsigned long long> ts("timestamp_usecs")`
    creates an object, `ts`, which sets and gets data of type
    `unsigned long long` associated with the "timestamp_usecs" label.

    The assumption is that concepts such as "bounding box" are common
    across many file formats, and while these formats may have
    different methods of storing the data, the act of associating them
    with the same data type and string asserts that they are all
    semantically equivalent.

3. These `track_field` objects do not hold any track or frame data.
Instead, they identify columns in the data cloud.

4. Any particular file format is defined as a collection of these
`track_field` objects; an equivalent view is that a file format is a
schema specifying which data columns the format populates.

## The goal

The goal of track_oracle is to allow client code to be written against
data schemas, not file formats.  In particular, the client code may be
written to require a data schema which is not satisfied by any single
file format, as long as the missing data columns can be
algorithmically filled in.

For example, the scoring code aligns tracks by timestamp.  Some file
formats (e.g. XGTF) do not define timestamps.  However, a utility
routine can synthesize timestamps given a base timestamp and and a
frame rate.  This utility routine can insert the timestamps into the
"timestamp_usecs" column for each row populated by the XGTF reader.
Thus:

- The alignment code never knows what format supplied the data;
  all it knows is that it needs timestamps.
- The XGTF reader code never knows about the alignment code; all
  it knows is how to populate the columns defined by the XGTF
  format.
- The code which knows how to add timestamps to XGTF is located
  directly where it is needed and can exist without modifying the
  data contracts promised by the either of the above.

Although currently implemented as an in-memory database, alternate
implementations of the backend using persistent databases are also possible.


## Examples

The examples/ subdirectory contains some short examples exercising
various aspects of track_oracle including simple schema construction
and use, reading files and displaying their schemas, and examples of
schema introspection.
