% This file is part of VIAME, and is distributed under an OSI-approved
% BSD 3-Clause License. See either the root top-level LICENSE file or
% https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

function detector_initialize()
% This function is called to initialize the detector after all
% configuration parameters have beed applied as assignments.
  disp ('In initialize');

  global GMM_detector;
  GMM_detector = vision.ForegroundDetector('NumTrainingFrames', num_frames, 'InitialVariance', init_var);
end
