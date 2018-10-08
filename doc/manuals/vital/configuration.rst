Configuration
=============

In many computer software systems, system configuration is essentially an
afterthought.  Frequently, a few “.ini” or “.yaml” files are created that carry
a some key configuration settings and that’s the end of it. In contrast to this,
the configuration of a computer vision system is frequently a first order
component of the system’s working technology.  Computer vision algorithms tend
to be highly configurable, exposing many different execution parameters and,
perhaps more importantly, are very tunable for different operating conditions,
performance profiles, and operational characteristics.

To facilitate this, KWIVER provides a hierarchical configuration system
including a flexible configuration language that is used for virtually all
aspects of KWIVER’s operation.  The following sections will detail how to use
KWIVER's *config_block* architecture and the configuration language used to
create and manipualte *config_blocks*.

.. toctree::
   :maxdepth: 2

   config_usage
   config_file_format
