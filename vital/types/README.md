**bounding_box** - 

**camera** - Defines the interface base class camera and the subclass simple_camera, which capture all of the intrinsic and extrinsic parameters (e.g., position and orientation) of a camera. The intrinsic parameters are represented with an instance of the intrinsic_camera class.

**camera_intrinsics** - Defines the interface base class camera_intrinsics and the subclass simple_camera_intrinsics, which capture all of the internal details of a camera (e.g., focal length, principal point, distortion). The methods of the class allow ray vectors in the camera to camera coordinate system to be moved to image coordinates, and image coordinates to be unprojected into a ray in the camera coordinate system.

**camera_map** - Defines the interface base class camera_map and the subclass simple_camera_map, which allow representation of a group of camera instances.

**color** - Defines the rgb_color struct, which captures the red, green, and blue values of a single pixel.

**covariance** - Representation of 2-d and 3-d covariance matrices.

**descriptor** - 

**descriptor_set** - 

**detected_object** - 

**detected_object_set** - 

**detected_object_type** - 

**essential_matrix** - 

**feature** - 

**feature_set** - 

**fundamental_matrix** - 

**geo_point** - 

**geo_polygon** - 

**geo_MGRS** - 

**homography** - Defines a 3x3 homography matrix object allowing 2-D to 2-D point mappings.

**homography_f2f** - Frame to frame homography definition.

**homography_f2w** - Frame to world homography definition.

**image** - 

**image_container** - 

**landmark** - 

**landmark_map** - 

**match_set** - 

**matrix** - 

**mesh** - 

**polygon** - 

**rotation** - Defines a rotation object, which can operate on a vector or other rotation object. Supports conversions between types: quaternion (native internal format), axis/angle, yaw/pitch/roll, direction cosine matrix.

**similarity** - 

**timestamp** - 

**timestamp_config** - 

**track** - 

**track_set** - 

**uuid** - 

**vector** - wrapper on the Eigen libraries vector object.
