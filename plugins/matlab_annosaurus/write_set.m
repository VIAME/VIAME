function write_set(detected_object_set, detected_object_classification, image_name)
    
    % Make the annotation data accessible outside the function if needed
    global annotation_data;
    
    endpoint = 'http://localhost:8080';
    
    % Need a UUID to group images together. VIAME does not yet pass a grouping key.
    % But this feature has been requested
    uuid = char(java.util.UUID.randomUUID.toString.toLowerCase);
    
    a = build_annotation_data(detected_object_set, detected_object_classification, uuid)
    for i = 1:length(a)
        % NOTE That we just get the image_name, we really need a fully resolved path to convert to a file url.
       annotation_data{i} = put_annotation(endpoint, a{i}, image_name);
    end
    
end