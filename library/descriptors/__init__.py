# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Descriptor indexing and querying, in python.

`iqr_adaboost` is the model the C++ IQR session trains and scores through;
`index_descriptors` and `query_service` are the indexing and query tools.
From `viame.core` in P2-T06. None of them registers an algorithm or a
process, so this package declares nothing and is absent from
`BUILTIN_PLUGIN_PACKAGES`; `viame.descriptors.torchvision` is not.
"""
