# ckwg +29
# Copyright 2023 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#    * Neither name of Kitware, Inc. nor the names of any contributors may be used
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

import numpy as np
import os
import shutil

def pad_img_to_fit_bbox( img, x1, y1, x2, y2 ):
    import cv2
    img = cv2.copyMakeBorder( img, - min( 0, y1 ), max( y2 - img.shape[0], 0),
            -min( 0, x1 ), max( x2 - img.shape[1], 0), cv2.BORDER_CONSTANT )

    y2 += -min( 0, y1 )
    y1 += -min( 0, y1 )
    x2 += -min( 0, x1 )
    x1 += -min( 0, x1 )

    return img, x1, x2, y1, y2

def safe_crop( img, x1, y1, x2, y2 ):
   if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox( img, x1, y1, x2, y2 )
   return img[ y1:y2, x1:x2, : ]

def recurse_copy( src, dst, max_depth = 10, ignore = ".json" ):
    if max_depth < 0:
        return src
    if os.path.isdir( src ):
        for entry in os.listdir( src ):
            recurse_copy(
              os.path.join( src, entry ),
              dst,
              max_depth - 1,
              ignore )
    elif not src.endswith( ignore ):
        shutil.copy2( src, dst )
