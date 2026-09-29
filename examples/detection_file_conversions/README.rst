
==========================
Detection File Conversions
==========================

This document corresponds to the `detection file conversions`_ example folder within a
VIAME desktop installation. This folder shows how to convert between textual formats
representing object detections, tracks, results, etc. The formats themselves are described
in `detection file formats <https://viame.github.io/VIAME/sections/detection_file_formats.html>`__. Conversions are
performed with the ``viame convert`` tool, which reads any registered annotation format
and writes any other, one file or a whole folder at a time::

    viame convert annotations.csv annotations.json          # VIAME CSV to COCO
    viame convert annotations.csv annotations.dive.json     # VIAME CSV to DIVE
    viame convert results.json results.csv                  # COCO or DIVE to VIAME CSV
    viame convert training_data output_folder -o coco       # every file under a folder

The input format is recognised from each file's extension and content, and the output
format from the output extension or the ``-o`` flag. When the imagery an annotation file
belongs to sits next to it (images in the same folder, or a video), the tool uses it for
the frame names, frame count and timing of the output; ``--no-images`` converts from the
annotation files alone, and ``--images`` points at imagery kept elsewhere. See the
`Example Conversions`_ section below and the ``bulk_convert`` scripts in this folder.

.. _detection file conversions: https://github.com/VIAME/VIAME/tree/main/examples/detection_file_conversions

*******************
Example Conversions
*******************

The ``viame convert`` tool converts between every registered reader and writer
directly, without a pipeline. Tracks are carried across when both formats hold
them, otherwise per-frame detections. Run ``viame convert --list-formats`` to see
what is available; ``viame convert --help`` lists the options.

Single files::

    viame convert groundtruth.csv groundtruth.json           # to COCO
    viame convert groundtruth.csv groundtruth.dive.json      # to DIVE
    viame convert groundtruth.kw18 groundtruth.csv           # KW18 to VIAME CSV
    viame convert habcam.csv habcam_viame.csv -i habcam      # HabCam CSV to VIAME CSV
    viame convert results.json results.csv -o viame_csv      # COCO or DIVE to VIAME CSV

Folders, mirroring the input layout into the output folder with the new extension::

    viame convert training_data converted -o coco
    viame convert training_data converted -o dive --no-images

Imagery alongside the annotations is used automatically: an image folder gives the
frame names and count (so empty frames are recorded too, as COCO expects), and a
video gives the frame count and timestamps. ``--frame-rate`` sets the rate the
frames of a video are numbered at, matching the ``-frate`` used when the
annotations were produced, and applies timestamps to image sequences::

    viame convert clip.csv clip.json --frame-rate 5          # clip.mp4 found alongside
    viame convert annotations.csv out.json --images frames/  # imagery kept elsewhere

Reader and writer settings are passed as ``-s key=value``, prefixed by ``reader:``
or ``writer:`` when the two share a key::

    viame convert in.csv out.csv -s writer:tot_option=average

``viame run --gt-only`` also converts annotation folders through the same tool, for
batch runs that already use the run applet, and the ``bulk_convert`` scripts in this
folder show both the with-data and annotation-only forms.

The ``standalone_utils`` folder keeps older single-purpose scripts for formats that
are not registered readers (Scallop-TK, PVO and similar).


.. dive-crosslink

******************
DIVE Documentation
******************

`DIVE command line tools`_ covers the separate ``dive convert`` utility, and `DIVE data
formats`_ the formats DIVE converts on import and export.

.. _DIVE command line tools: https://viame.github.io/VIAME/sections/dive/Command-Line-Tools.html
.. _DIVE data formats: https://viame.github.io/VIAME/sections/dive/DataFormats.html


********************
Code and Build Flags
********************

Command line tools:

* tools/convert.cxx -- ``viame convert``
