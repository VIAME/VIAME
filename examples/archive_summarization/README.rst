
===========================
Video Archive Summarization
===========================

This document corresponds to `this example online`_, in addition to the
archive_summarization example folder in a VIAME installation.

.. _this example online: https://github.com/Kitware/VIAME/tree/master/examples/archive_summarization


This example covers scripts which simultaneously create a searchable index of video archive
and plots detailing different organism counts over time. The 'summarize_and_index_videos'
script performs both of these tasks, while the 'summarize_videos' script only performs the
later. Plots are generated, for each video, in the 'database' output folder, and can
alternatively be viewed by the 'launch_timeline_viewer' script. Queries can be performed
via the 'launch_search_interface' script, in a fashion similar to both the 'viame_ingest' example
in 'search_and_rapid_model_generation' and in the binary install guide.
