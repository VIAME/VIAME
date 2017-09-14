Sprokit Architecture
====================

Sprokit is a "**S**\ tream **Pro**\ cessing Tool\ **kit**" that provides
infrastructure for chaining together algorithms into pipelines for
processing streaming data sources.  The most common use case of Sprokit
is for video processing, but Sprokit is data type agnostic and could be
used for any type of streaming data.  Sprokit allows the user to dynamically
connect and configure a pipeline by chaining together processing nodes
called "processes" into a directed graph with data sources and sinks.
Sprokit schedules the jobs to run each process and keep data flowing through
pipeline.  Sprokit also allows processes written in Python to be
interconnected with those written in C++.

.. toctree::
   :maxdepth: 3

   getting-started
   process
   how_to_process
   plugins
   plugin_explorer
   pipeline_design
   pipeline_declaration
   pipeline_examples
   how_to_pipeline
