This folder contains an example for video search.

It currently requires Linux or Mac systems, and a VIAME build with SMQTK turned on.

An arbitrary tracking pipeline is used to first generate spatio-temporal object tracks
representing object candidate locations in video. Descriptors are generated around these
object tracks, which get indexed into a database and can be queried upon. By indicating
which query results are correct, a model can be trained for a new object category and
saved out to be reused again.
