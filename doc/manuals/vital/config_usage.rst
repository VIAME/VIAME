vital Configuration Usage
============================

Meta: The goal if this document is to provide a single reference on
how to *use* the config library not necessarily the detailed config
specification syntax or the API. (but maybe it should) There is a
config file format document which may be useful as a separate
document/section at the end.

- what is a config
- how to create them
- best practices


Introduction
------------
The vital `config_block` supports general configuration tasks where a
general purpose key/value pair is needed.

Configuration Features
----------------------
config features. what they do and why you would want to use them
- relativepath
- macro providers and how they can be used to make portable and reusable config files



From File to config_block
-------------------------
Using config_block_io to directly convert config file into block. This
can be used by a main program that manages configs and algorithms
directly. The read_congifg_file() uses a complex set of rules to
locate config files based on host system and application name.


Establishing Expected Config
----------------------------
Typically the expected config is formulated by creating a config block
with all the expected keys, default values, and entry description.

Don't be shy with the entry description. This description serves as
the design specification for the entry. The expected format is a short
description followed by a longer detailed description separated by two
new-lines.

std::string description = "Short description\n\n\"
                          "Longer description which contains all information needed "
                          "to correctly specify this parameter including any range "
                          "limitations etc.";

The long description deos not need any new-line characters for
formatting unless specific formatting is desired. The text is wrapped
into a text block by all tools that display it.

Verifying a Configuration
'''''''''''''''''''''''''
Config verification using config_diff() tools.  Explain how to
determine if any expected or required config items are not supplied or
any unexpected config items are supplied. These techniques are useful
for detecting misspelled config keys.

Usage by Algorithms
'''''''''''''''''''
Config comes from a file that is read by an application
(usually). General structure of a config block for an algorithm. This
may have changed with the addition of the block/endblock features.

- correct sequence of calls when dealing with algorithms. There seems
  to be some disagreement about this.

  check_config();
  set_config();
  get_config() - optional. explain why and where it is needed

  How to configure nested algorithms. That is algorithms that create other algorithms.
  Explain what the code looks like and what the config file looks like.

Usage by Processes
''''''''''''''''''
Config comes from a section of the pipe file. The config may be
included and it may be shared with an application as above. That is, a
config section can be shared between an executable and a pipeline
file. Also add code snippet on how to use config verification calls to
check the config against an expected one.

What are the recommendations for algorithm wrapper processes about
adding a config key for the main algorithm config block.


Config Management Techniques
''''''''''''''''''''''''''''
or best practices for config files

How to use include and block to make reusable config
sections. Starting with the example config section that follows.


    algorithm_instance_name:type = type_name
    algorithm_instance_name:type_name:algo_param = value
    algorithm_instance_name:type_name:threshold = 234

alteratively

    algorithm_instance_name:type = type_name
    block  algorithm_instance_name:type_name
      algo_param = value
      threshold = 234
      ...
    endblock

or

    algorithm_instance_name:type = type_name
    block  algorithm_instance_name
      include type_name.conf
    endblock

where type_name.conf contains

    block   type_name
      algo_param = value
      threshold = 234
      ...
    endblock


### Macros Available in Configuration ###


Config entry provider (Macro) - how to specify the key name in a
stand-alone file.  in a pipeline file. (unbound config blocks,
processes, clusters) in a config fragment. This implies a relative key
specification. This is uncharted territory since the key must be fully
specified. Maybe we need to add a current config block reference to
help resolve relative references.
