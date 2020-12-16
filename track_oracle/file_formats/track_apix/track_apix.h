// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_TRACK_APIX_H
#define INCL_TRACK_APIX_H

#include <vital/vital_config.h>
#include <track_oracle/track_apix/track_apix_export.h>

#include <track_oracle/track_base.h>
#include <track_oracle/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

/*
Looking at the files written by APIX 1.70L's "Tracks->Save All Tracks to
Shapefiles" option, we see that the .dbf schema is:

Field 0: Type=N/Double, Title=`Latitude', Width=12, Decimals=7
Field 1: Type=N/Double, Title=`Longitude', Width=12, Decimals=7
Field 2: Type=N/Integer, Title=`DataUTCTim', Width=10, Decimals=0
Field 3: Type=N/Integer, Title=`DataTimeMS', Width=4, Decimals=0
Field 4: Type=C/String, Title=`TimeString', Width=30, Decimals=0
Field 5: Type=C/String, Title=`MGRS', Width=24, Decimals=0
Field 6: Type=N/Integer, Title=`FrameNum', Width=7, Decimals=0
Field 7: Type=N/Integer, Title=`Intrplat', Width=2, Decimals=0

...while the .shp is a list of points, e.g.

Shape:0 (Point)  nVertices=1, nParts=0
  Bounds:(    -101.876,      33.581, 0)
      to (    -101.876,      33.581, 0)
     (    -101.876,      33.581, 0)

...each point apparently a (lat, lon).

The MITRE ground truth schema is:

Field 0: Type=N/Double, Title=`Latitude', Width=12, Decimals=7
Field 1: Type=N/Double, Title=`Longitude', Width=12, Decimals=7
Field 2: Type=N/Integer, Title=`DataUTCTim', Width=10, Decimals=0
Field 3: Type=N/Integer, Title=`DataTimeMS', Width=4, Decimals=0
Field 4: Type=C/String, Title=`TimeString', Width=30, Decimals=0
Field 5: Type=C/String, Title=`MGRS', Width=24, Decimals=0
Field 6: Type=N/Integer, Title=`FrameNum', Width=7, Decimals=0
Field 7: Type=C/String, Title=`SourceImg', Width=128, Decimals=0

...and the .shp files are the same as the APIX output.

In both cases, there are the same number of points in the .shp file as
rows in the .dbf.  Dunno if they are linked by anything other than
order of appearance.

*/

namespace kwiver {
namespace track_oracle {

struct TRACK_APIX_EXPORT track_apix_type: public track_base< track_apix_type >
{
  // track level data
  track_field< dt::tracking::external_id > external_id;

  // frame level data
  track_field< dt::tracking::latitude > lat;
  track_field< dt::tracking::longitude > lon;
  track_field< dt::tracking::time_stamp > utc_timestamp;

  track_apix_type()
  {
    Track.add_field( external_id );
    Frame.add_field( lat );
    Frame.add_field( lon );
    Frame.add_field( utc_timestamp );
  }
};

} // ...track_oracle
} // ...kwiver

#endif
