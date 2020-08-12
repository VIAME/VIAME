Logging Guidelines
==================

The following are general guidelines for using logging in KWIVER. In
addition, the available log levels are listed below and guidance on
which level applies in a given situation.

Logger Names
------------

Logger instances are identified by a name string which can be composed
as a hierarchy with each level separated by a period ('.'). The main
purpose of the logger name is to provide structure for grouping
loggers so their output can be controlled in a manner to get the
desired subset of log messages needed to verify correct operation or
to diagnose a problem. The output can be controlled at any level in
the naming hierarchy. Logger names should represent a logical
structure of the application or framework (based on fuctional
groupings) rather than on a physical file name or class name approach.
The location of the log message is provided directly by the file name
and line number attributes.

FATAL
-----

This should generally only be used for recording a failure that
prevents the system starting, i.e. the system is completely
unusable. It is also possible that errors during operation will also
render the system unusable.


ERROR
-----

Records that something went wrong, i.e. some sort of failure occurred, and either:

- The system was not able to recover from the error, or

- The system was able to recover, but at the expense of losing some
information or failing to honour a request.

- This should be immediately brought to the attention of an operator. Or
to rephrase it, if your error does not need immediate investigation by
an operator, then it isn’t an error.

- To permit monitoring tools to watch the log files for ERRORs and
WARNings is crucial that:

These get logged:

- Sufficient information is provided to identify the cause of the
problem

- The logging is done in a standard way, which lends itself to automatic monitoring.

- For example, if the error is caused by a configuration failure, the
configuration filename should be provided (especially if you have more
than one file, yuck), as well as the property causing the problem.

WARN
----

A WARN message records that something in the system was not as
expected. It is not an error, i.e. it is not preventing correct
operation of the system or any part of it, but it is still an
indicator that something is wrong with the system that the operator
should be aware of, and may wish to investigate. This level may be
used for errors in user-supplied information.

INFO
----

INFO priority messages are intended to show what’s going on in the
system, at a broad-brush level. INFO messages do not indicate that
something’s amiss (use WARN or ERROR for that), and the system should
be able to run at full speed in production with INFO level logging.

The following types of message are probably appropriate at INFO level:

<System component> successfully initialised

<Transaction type> transaction started, member: <member number>, amount: <amount>

<Transaction type> transaction completed, txNo: <transaction number>,
member: <member number>, amount: <amount>, result: <result code>

DEBUG
-----

DEBUG messages are intended to help isolate a problem in a running
system, by showing the code that is executed, and the context
information used during that execution. In many cases, it is that
context information that is most important, so you should take pains
to make the context as useful as possible. For example, the message
‘load_plugin() started’ says nothing about which plugin is being
loaded or from which file, or anything else that might help us to
relate this to an operation that failed.

In normal operation, a production system would not be expected to run
at DEBUG level. However, if there is an occasional problem being
experienced, DEBUG logging may be enabled for an extended period, so
it’s important that the overhead of this is not too high (up to 25% is
perhaps OK).

The following types of message are probably appropriate at DEBUG level:

Entering <class name>.<method name>, <argument name>: <argument value>, [<argument name>: <argument value>…]

Method <class name>.<method name> completed [, returning: <return value>]

<class name>.<method name>: <description of some action being taken, complete with context information>

<class name>.<method name>: <description of some calculated value, or decision made, complete with context information>

Please note that DEBUG messages are intended to used for debugging in
production systems, so must be written for public consumption. In
particular, please avoid any messages in a non-standard format, e.g.

DEBUG ++++++++++++ This is here cause company “Blah” sucks +++++++++++

If a DEBUG message is very expensive to generate, you can guard it
with a logger.IS_DEBUG_ENABLED() if check. Just make sure that nothing
that happens inside that if block is required for normal system
operation. Only sendmail should require debug logging to work.

TRACE
-----

TRACE messages are intended for establishing the flow of control of
the system. Typically TRACE messages are generated upon entering and
exiting functions or methods.

When to log an Exception?
-------------------------

Ideally, an exception should only be logged by the code that handles
the exception. Code that merely translates the exception should do no
logging.
