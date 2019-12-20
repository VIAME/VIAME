Process Coding Patterns
=======================

Generally the best parctices for writing processes is captured by the
template process, but there are some features that are not universally
needed in all processes. The following code snippets are the best
practice for some frequent operations in a process.

Verifying Config Parameters
---------------------------

The following code snippet can be used in the
``process::_configure()`` method to report any unexpected
configuration parameters. Unexpected parameters are usually a
misspelling of a valid parameter. Tracking these down can take quite a
while. If your process has a well defined set of configuration
parameters, this code can be used to spot misspellings. Note that
processes that wrap algorithms (arrows) can not usually do this since
the configuration of the arrow is not known to the process.::

    void
    your_process
    ::_configure()
    {
      // check for extra config keys
      auto cd = this->config_diff();
      cd.warn_extra_keys( logger() );

      // regular config processing
    }
