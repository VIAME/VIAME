function b = put_annotation(endpoint, s, image_url)
% put_annotation - Store an annotation structure in MBARI'a annotation database
%
% Usage:
%   b = put_annotation(endpoint, s, image_url)
%
% Inputs:
%   endpoint = The base URL of the annosaurus endpoint. For testing this is 'http://localhost:8080'
%   s = An annotation structure from buld_annotation_data
%   image_url = The URL of the image that the annotations are for.

% Brian Schlining
% 2016-09-23
    
    date_format = java.text.SimpleDateFormat('yyyy-MM-dd''T''HH:mm:ss.sss''Z''');
    date_format.setTimeZone(java.util.TimeZone.getTimeZone('UTC'));
    rt = char(date_format.format(java.util.Date)); % HACK: This is a fake index. Waiting for pipelines to pass image timestamps
    
    % Post annotation
    a = webwrite([endpoint '/v1/annotations'], ...
       'video_reference_uuid', s.uuid, ...
       'recorded_timestamp', rt, ...
       'concept', s.concept, ...
       'observer', s.observer);
       
    % Post image. If it has same recorded timestamp and videoreferenceuuid as annotation they will be associated in db
    try 
        i = webwrite([endpoint '/v1/images'], ...
           'video_reference_uuid', s.uuid, ...
           'recorded_timestamp', rt, ...
           'url', image_url); % HACK: We don't have full path to image from VIAME
           % Also, VIAME does not handle image URLs. So we are storing a local file URL. Not ideal
           % We can also pass image width, height and format (mimetype) if we want ...
    catch me
        e = char(java.net.URLEncoder.encode(image_url, 'UTF-8'));
        i = webread([endpoint '/v1/images/url/' e]);
    end
    
    % Post associations 
    for j = 1:length(s.associations)
        ass = s.associations{j};
        link_value = ['{"image_reference_uuid": "' i.image_reference_uuid  '", "value": "' ass.link_value '" }'];
        webwrite([endpoint '/v1/associations'], ...
           'observation_uuid', a.observation_uuid, ...
           'link_name', ass.link_name, ...
           'link_value', link_value); 
    end
    
    % Read back our complete annotation from the data store and convert to matlab structure
    b = webread([endpoint '/v1/annotations/' a.observation_uuid]);
    
end