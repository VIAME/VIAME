% This file is part of VIAME, and is distributed under an OSI-approved
% BSD 3-Clause License. See either the root top-level LICENSE file or
% https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

function detect( in_image )
  global detected_object_set;
  global detected_object_classification;
  global detected_object_chips;

% This function is called to perform the detection operation on the
% supplied image.

% Need a format for returning the detections.
%
% Data needed to create a detection object:
% - bounding box for detection in pixel coords
% - Confidence value 0.0 - 1.0
% - Optional list of classification names and scores
%
% detected_object = [ ul_x, ul_y, lr_x, lr_y, confidence ]
% detected_object_set = [ detected_object; detected_object; ... ]
%
% for i = mum_detections
  % for j = num_classes-for detections(i)
% detected_object_classification(i,j).name = 'class';
% detected_object_classification(i,j).score = 0.23;
%

  % Print out text
  Disp('Hello World')

  % for example
  % 3 detections on this object
  % Box coordinates are tl-x, tl-y, lr-x, lr-y
  detected_object_set = [ 100 120 220 220 .56; % box and confidence for detection
                          550 550 860 860 .77;
                          900 500 1040 1040 .54];

  % Classification of the detections are optional, but if there are any
  % they must be represented in the following structure.
  % There *must* be the same number of rows in the classification array as there are detections.
  % 2 possible classifications for object 1
  detected_object_classification(1,1).name='scallop';
  detected_object_classification(1,1).score=.56;
  detected_object_classification(1,2).name='rock';
  detected_object_classification(1,2).score=.3;

  detected_object_classification(2,1).name='scallop';
  detected_object_classification(2,1).score=.56;

  detected_object_classification(3,2).name='rock-lobster';
  detected_object_classification(3,2).score=.3;

  % Classification of the detections are optional, but if there are any
  % they must be represented in the following structure.
  % There *must* be the same number of rows in the classification array as there are detections.
  % 2 possible classifications for object 1
  detected_object_chips(1).chip = [ 100 120 123 0 12 0;
                                    10 2013 20 2 1;
                                    10 20 20 ];
  detected_object_chips(2).chip = [ 100 120 123 0 12 0;
                                    10 2013 20 2 1;
                                    10 20 20 ];
  detected_object_chips(3).chip = [ 100 120 123 0 12 0;
                                    10 2013 20 2 1;
                                    10 20 20 02 1 ];

  image( in_image );  % TEMP
end
