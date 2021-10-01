
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

  global GMM_detector;
  global factor
  global min_aspect;
  global max_aspect;
  global factor;

[mask, imd, GMM_detector]=gmm_background_remove(GMM_detector,in_image,factor);
targets=extract_targets2(mask,imd,min_size,ROI,min_aspect,max_aspect,factor);
% detected_object_set  = extract_chip_coords(targets);
[detected_object_set, detected_object_chips] = extract_chip_coords2(targets,imd);

image( in_image );  % TEMP

end
