Arrows Coding Patterns
======================

There are a few common operations that are used in many arrows that
can be termed coding patterns. Here are a few of them.

Resolving expected and supplied configurations
----------------------------------------------

There are cases where an arrow has an expected configuration with
default values for the parameters. The supplied config may not have
all the required config items so the working config should take the
items that are supplied to the ``set_configuration()`` method and
supply any missing values using the defaults that the arrow is
expecting. This can be done as follows::

    void
    your_arrow
    ::set_configuration(vital::config_block_sptr config_in)
    {
      // Starting with our generated config_block to ensure that assumed values are present
      vital::config_block_sptr config = this->get_configuration();

      // Then merge the supplied config, overwriting the default values.
      config->merge_config( config_in );

      d->m_dp         = config->get_value<double>( "dp" );
      d->m_min_dist   = config->get_value<double>( "min_dist" );
      // Other config items as required
    }


Detecting Extra or Misspelled Parameters
----------------------------------------

A very common problem when configuring arrows is misspelling the
configuration parameter name. In the general case, the system can not
determine if a parameter is required by an arrow or not. The following
code in the ``set_configuration()`` method will log unexpected
configuration items as long as the arrow has a fixed set of parameters.::

    void
    your_arrow
    ::set_configuration(vital::config_block_sptr config_in)
    {
      // Starting with our generated config_block to ensure that assumed values are present
      vital::config_block_sptr config = this->get_configuration();

      // Compare expected configuration with the one supplied.
      kwiver::vital::config_difference cd( config, config_in );
      cd.warn_extra_keys( logger() );

      // merge and/or handle parameters as needed
    }

Additional unexpected parameters can be detected in the
``check_configuration()`` method as follows::

    bool
    your_arrow
    ::check_configuration(vital::config_block_sptr config_in) const
    {
      // Get expected comfiguration
      vital::config_block_sptr config = this->get_configuration();

      // Compare against the supplied configuration
      kwiver::vital::config_difference cd( config, config_in );

      // Return TRUE if there are any extra config keys/items
      return cd.warn_extra_keys( logger() );
    }
