clear all; clc; close all;

global detected_object_set;
global detected_object_classification;
global annotation_data;

endpoint = 'http://localhost:8080';


detected_object_set = [ 100 120 220 220 .56; ... % box and confidence for detection
                          550 550 860 860 .77; ...
                          900 500 1040 1040 .54];

detected_object_classification(1,1).name='scallop';
detected_object_classification(1,1).score=.56;
detected_object_classification(1,2).name='rock';
detected_object_classification(1,2).score=.3;

detected_object_classification(2,1).name='scallop';
detected_object_classification(2,1).score=.56;

detected_object_classification(3,2).name='rock-lobster';
detected_object_classification(3,2).score=.3;

uuid = char(java.util.UUID.randomUUID.toString.toLowerCase);

a = build_annotation_data(detected_object_set, detected_object_classification, uuid);
for i = 1:length(a)
   annotation_data{i} = put_annotation(endpoint, a{i}, 'file:/path/to/image.png');
end