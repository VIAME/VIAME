#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Standalone CLI for survey metadata extraction.

The logic (GPS/EXIF/flight-log parsing, per-image record building) lives in
viame.core.survey_metadata so the registration and prior-coverage processes
can import it; this is a thin command-line wrapper around it.
"""

from viame.core.survey_metadata import main

if __name__ == '__main__':
    main()
