import json
import sys

if len( sys.argv ) < 3 or len( sys.argv ) > 4:
  print "json_to_kw18.py [input_file] [output_file]"
  sys.exit(0)

data = json.load( open( sys.argv[1] ) )

f = open( sys.argv[2], 'w' )

id_counter = 0;

for frame in data["frames"]:
  print "Processing frame ID " + str( frame["frame_id"] )

  for frame_roi in frame["frame_rois"]:
    
    id_counter = id_counter + 1
    
    x = frame_roi["roi_x"]
    y = frame_roi["roi_y"]
    h = frame_roi["roi_h"]
    w = frame_roi["roi_w"]
    
    lbl = frame_roi["roi_label"]["label_name"]

    if lbl != "Negative":

      output_str = str( id_counter ) + " 1 " + str( frame["frame_id"] ) + " 0 0 0 0 "
      output_str = output_str + str(int(x+w/2)) + " " + str(int(y+h/2)) + " "
      output_str = output_str + str(x) + " " + str(y)+ " " + str(x+w) + " " + str(y+h)
      output_str = output_str + " 0 0 0 0 0\n"

      f.write( output_str )
