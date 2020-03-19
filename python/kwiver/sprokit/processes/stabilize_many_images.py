# ckwg +29
# Copyright 2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# XXX This class should be defined somewhere else
from .simple_homog_tracker import Transformer

@Transformer.decorate
def stabilize_many_images():
    """Create a Transformer that performs stabilization on multiple
    images captured by a multi-camera system.  The .step call expects
    one argument:
    - a list of images (XXX what image type?)
    and returns:
    - a list of homographies, one for each input image, that maps them
      and other processed images to a common coordinate system (XXX
      what homography type?)

    """
    ris = register_image_set()
    eh = estimate_homography()
    output = None
    while True:
        images, = yield output
        fds = list(map(compute_features_and_descriptors, images))
        spatial_homogs = ris.step(fds)
        all_features, all_descs = merge_feature_and_descriptor_sets(
            warp_features_and_descriptors(h, *fd)
            for h, fd in zip(spatial_homogs, fds)
            # XXX Maybe we want to support something like this to
            # handle questionable matches:
            #   if h is not None
        )
        temporal_homog = eh.step(all_features, all_descs)
        output = [compose_homographies(temporal_homog, sh)
                  for sh in spatial_homogs]

@Transformer.decorate
def register_image_set():
    """Create a Transformer that registers sets of images (described
    by feature and descriptor sets) from a camera system together,
    maintaining information on previously estimated relative
    positions.  The .step call expects one argument:
    - a list of feature-set--descriptor-set pairs (adjacent images
      should overlap) (XXX what feature and descriptor types?)
    and returns:
    - a list of homographies, one for each input image, that map them
      to some common coordinate space (XXX what homography type?)

    """
    raise NotImplementedError

def estimate_homography():
    """Create a Transformer that estimates homographies using features and
    descriptors.  The .step call expects two arguments:
    - a feature set
    - the corresponding descriptor set
    and returns:
    - a homography mapping the current coordinates to those in a
      reference frame

    """
    raise NotImplementedError

def compute_features_and_descriptors(image):
    """Return a pair containing the feature set and corresponding
    descriptor set for the given image.

    """
    raise NotImplementedError

def warp_features_and_descriptors(homog, features, descriptors):
    """Return a pair of the given feature set and descriptor set, warped
    according to the provided homography.

    """
    raise NotImplementedError

def merge_feature_and_descriptor_sets(fd_pairs):
    """Merge an iterable of pairs of feature sets and corresponding
    descriptor sets into a single pair of one feature set and the
    corresponding descriptor set.

    Near-duplicates (e.g. those nearly sharing a position) are removed
    except for a single copy.

    """
    raise NotImplementedError

def compose_homographies(second, first):
    """Return a homography corresponding to applying the first, then the
    second homography.

    """
    raise NotImplementedError
