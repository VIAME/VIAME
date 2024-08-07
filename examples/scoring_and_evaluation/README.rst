==============================
Scoring Detectors and Trackers
==============================

-------
Summary
-------


.. image:: http://www.viametoolkit.org/wp-content/uploads/2018/02/scoring-2.png
   :scale: 30
   :align: center
   :target: https://github.com/VIAME/VIAME/tree/master/examples/scoring_and_evaluation

This document corresponds to the `scoring and evaluation`_ example folder within a
VIAME desktop installation. Contained in this folder are a few options for scoring
either detections, frame-level classifications, or object tracks.

For each script, there are two operating modes: a normal ("across all") option and
a per-class option. The normal method will score the outputs of all categories jointly
together, typically producing a results figure or table showing all object classes.
In most of these cases the default is to treat the class with the highest confidence
in a detection as its respective label, and ignore any classes with lesser scores.
In the per-class usage, this will not be the case, and the confidence score for each
category will be considered regardless of other categories, and a plot/chart for a
single category will be generated (for each category).

.. _scoring and evaluation: https://github.com/VIAME/VIAME/blob/master/examples/scoring_and_evaluation

-----------------------------------
PRC and Confusion Matrices (KWCOCO)
-----------------------------------

detection_prcs_and_conf_matrix_across_all

detection_prcs_and_conf_matric_per_category

-----------------------------------------------------------
Receiver Operating Curves (ROC) and Fixed Detection Metrics
-----------------------------------------------------------

detection_rocs_across_all

detection_rocs_per_category

-------------------------------------------------------
MOT - MOTA, IDF1, and other High-Level Track Statistics
-------------------------------------------------------

track_mot_stats_across_all

track_mot_stats_per_category


In addition to generating various MOT scores (e.g. IDF1, MOTA), this script allows
for the generation of custom filter file, with an optimal pre-class threshold
selected automatically for each class that optimizes some metric. This output
file (dive.config.json) can be loaded into the DIVE interface.

--------------------------------------------------
KWANT - Basic Track and Detection-Level Properties
--------------------------------------------------

track_kwant_stats_across_all

track_kwant_stats_per_category


The KWANT package provides scoring tools that can be used to
calculate the probability of detecting an item, along with other scoring
metrics such as ROC curves, specificity, sensitivities, etc. The input to
these tools must be in the Kitware kw18 format. Several scripts are provided to
convert other formats (such as habcam annotations and Scallop-tk outputs) to
kw18 format. The format is very simple so additional converters can be easily
created. 

An example of running scoring tools can be found `here`_.
The scoring tool takes two files: the actual detections in the truth
file and the computed detections. The computed detections are scored
against the truth file to give a set of statistics as shown below. Additional
parameters that can be passed to the tool and other options can be found in
the `KWANT documentation`_.

.. _here: https://github.com/VIAME/VIAME/blob/master/examples/scoring_and_evaluation/
.. _KWANT documentation: https://github.com/Kitware/kwant/blob/master/doc/manuals/introduction.rst

::

  HADWAV Scoring Results:
     Detection-Pd: 0.748387
     Detection-FA: 8
     Detection-PFA: 0.0338983
     Frame-NFAR: not computed
     Track-Pd: 0.748387
     Track-FA: 8
     Computed-track-PFA: 0.0338983
     Track-NFAR: not computed
     Avg track (continuity, purity ): 13.693, 1
     Avg target (continuity, purity ): 20.1419, 0.748387
     Track-frame-precision: 0.947826

The tool was originally written to analyze object tracks in full
motion video imagery so some of the terminology and calculated metrics
may not apply.

One main metric is the probability of detection Pd. This is calculated
as follows:

    Pd = (num detections match truth) / (num truth)

Detection files can be written in the kw18 format by using the
appropriate writer in the pipeline or by running one of these
converters. One downside to using the kw18 writer in the pipeline is
that the image file name is not captured.  All the converters take the
same set of command line options. For example:

::

  Usage: habcam_to_kw18.pl [opts] file
    Options:
      --help                     print usage
      --write-file file-name     Write image file/index correspondence to file
      --read-file  file-name     Read image file/index correspondence to file

In order to get the best statistics the number of images processed
must be the same as the number of images in the truth set. Computed
detections and truth are compared on an image basis so the number of
truth entries must be limited to the same number of images as the
computed detections. The options to these converters aide in this regard.

Calculated detections are converted first and use the --out-file
option to write out the list of files processed. The truth set is
processed next with the --in-file option referring to the file created
in the previous step. The --cache-only flag should be added to this
second conversion to cause images not in the first step to be skipped.

The score_tracks tool is run as follows:

::

  score_tracks --computed-tracks computed_det.kw18 --truth-tracks ground_truth2.kw18

A full list of the options can be coaxed from the tool by using the `-?` option.
