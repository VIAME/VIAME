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
