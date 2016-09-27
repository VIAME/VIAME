function a = build_annotation_data(detected_object_set, detected_object_classification, uuid, observer)
% build_annotation_data - build a datastructure for an annotation from detector output
%
% Usage:
%   a = build_annotation_data(detected_object_set, detected_object_classification, uuid)
%
% Inputs:
%   detected_object_set = A detected object structure. Provided by VIAME via global variable
%   detected_object_classification = classification output provided by VIAME.
%   uuid = A UUID for the image set (or source video). Will eventually be passed through by VIAME but
%          you may have to provide it yourself for now.
%   observer = The name of the observer who executed the run.
%  
% Output:
%   a = The resulting annotation from the datastore as a structure. An example of the fields in 
%       structure:
%                      observation_uuid: '76a5fe09-ac47-4a55-9dae-5fa90326824b'
%                               concept: 'Nanomia bijuga'
%                              observer: 'brian'
%                 observation_timestamp: '2016-09-23T18:32:42.784Z'
%                  video_reference_uuid: '318bd938-91b0-4d7d-adca-7725cd526339'
%                    imaged_moment_uuid: 'd7488926-ba6d-480d-91c2-085ba181fe2a'
%                    recorded_timestamp: '2016-09-23T11:32:16.016Z'
%                          associations: []
%                      image_references: [1x1 struct]
%
%       The image_references sub-structure is:
%                                   url: 'file:/foo.bar/woot.png'
%                         height_pixels: 0
%                          width_pixels: 0
%                     last_updated_time: '2016-09-23T18:42:54Z'
%                                  uuid: 'b8d988c3-3e57-4b3d-bbfb-d532519addc8'
%
% Example:
%   detected_object_set = [ 100 120 220 220 .56; ... % box and confidence for detection
%                           550 550 860 860 .77; ...
%                           900 500 1040 1040 .54];
%
%   detected_object_classification(1,1).name='scallop';
%   detected_object_classification(1,1).score=.56;
%   detected_object_classification(1,2).name='rock';
%   detected_object_classification(1,2).score=.3;
%
%   detected_object_classification(2,1).name='scallop';
%   detected_object_classification(2,1).score=.56;
%
%   detected_object_classification(3,2).name='rock-lobster';
%   detected_object_classification(3,2).score=.3;
%
%   uuid = char(java.util.UUID.randomUUID.toString);
%
%   a = build_annotation_data(detected_object_set, detected_object_classification, uuid);

% Brian Schlining (MBARI)
% 2016-09-23

    if nargin == 3
       observer = 'undefined'; 
    end

    [r c] = size(detected_object_set);
    n = 0;
    for i = 1:r
        k = length(detected_object_classification(r, :));
        % Build Annotation
        for j = 1:k
            n = n + 1;
            s.('uuid') = uuid;
            s.('observer') = observer;
            s.('concept') = detected_object_classification(i, j).name;
            
            v = mat2str(detected_object_set(i, :));
            if ~isempty(v)
                s.('associations'){1}.('link_name') = 'image detector bounding box and confidence [x y w h c]';
                s.('associations'){1}.('link_value') = v(2: end - 1);
            end
            
            v = num2str(detected_object_classification(i, j).score);
            if ~isempty(v)
                s.('associations'){2}.('link_name') = 'image detector score';
                s.('associations'){2}.('link_value') = v;
            end
            
            s.('associations'){3}.('link_name') = 'image detector name';
            % HACK: VIAME does not yet support passing the name of the image detector through a pipeline
            s.('associations'){3}.('link_value') = 'your detector name';
            a{n} = s;
        end
    end
end