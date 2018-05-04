import os
import shutil
import sys
import argparse
import glob
import cv2
import json
import ImageIO

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'objects_list',
        help='List of objects (.json)'
    )
    args = parser.parse_args()

    reader = open(os.path.join(os.environ['FS_ROOT'], args.objects_list), 'rt')
    data = json.load(reader)
    reader.close()
    input_file = data['input_file']
    output_dir = os.path.join(os.environ['FS_ROOT'], 'dataset', os.path.splitext(os.path.basename(input_file))[0])
    offset = min([data['frames'][f]['frame_id'] for f in range(len(data['frames']))])
    clip = (f + offset) / 900
    if args.objects_list.find('selected') == -1:
        sub_dir = 'clip{0:03d}'.format(clip)
    else:
        sub_dir = 'selected'
    if os.path.isfile('{0}/{1}/All/objects.json'.format(output_dir, sub_dir)) and os.path.isfile('{0}/{1}/Export/objects.json'.format(output_dir, sub_dir)):
        return
    elif os.path.isfile('{0}/{1}/All/objects.json'.format(output_dir, sub_dir)):
        reader = open('{0}/{1}/All/objects.json'.format(output_dir, sub_dir), 'rt')
        data = json.load(reader)
        reader.close()
    else:
        if not os.path.exists('{0}/{1}/All'.format(output_dir, sub_dir)):
            os.makedirs('{0}/{1}/All'.format(output_dir, sub_dir))
        if not os.path.exists('{0}/{1}/Export'.format(output_dir, sub_dir)):
            os.makedirs('{0}/{1}/Export'.format(output_dir, sub_dir))

    if os.path.isfile(os.path.join(os.environ['FS_ROOT'], input_file)):
        cap = cv2.VideoCapture(os.path.join(os.environ['FS_ROOT'], input_file))
        if cap.isOpened() == False:
            print('Cannot open input file: ' + os.path.join(os.environ['FS_ROOT'], input_file))
            exit(1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,offset)
    else:
        filelist = sorted(glob.glob(os.path.join(os.environ['FS_ROOT'], input_file, '*.jpg')))
    
    for f in range(len(data['frames'])):
        if os.path.isfile(os.path.join(os.environ['FS_ROOT'], input_file)):
            ret,img = cap.read()
        else:
            img = cv2.imread(filelist[f + offset])
            ret = img is not None

        if ret:
            for r in range(len(data['frames'][f]['frame_rois'])):
                #print json.dumps(data['frames'][f]['frame_rois'][r], indent = 4)
                roi = (data['frames'][f]['frame_rois'][r]['roi_x'], data['frames'][f]['frame_rois'][r]['roi_y'], data['frames'][f]['frame_rois'][r]['roi_w'], data['frames'][f]['frame_rois'][r]['roi_h'])
                print roi
                if not os.path.isfile('{0}/{1}/All/objects.json'.format(output_dir, sub_dir)):
                    output = '{0}/{1}/Import/{2}/images/frame{3:06d}/{3:06d}_{4:06d}.jpg'.format(output_dir, sub_dir, data['frames'][f]['frame_rois'][r]['roi_label']['label_name'], data['frames'][f]['frame_id'], data['frames'][f]['frame_rois'][r]['roi_id'])
                    print output
                    ImageIO.saveImg(output, img, roi)
                    output = '{0}/{1}/All/images/frame{2:06d}/{2:06d}_{3:06d}.jpg'.format(output_dir, sub_dir, data['frames'][f]['frame_id'], data['frames'][f]['frame_rois'][r]['roi_id'])
                    ImageIO.saveImg(output, img, roi)
                if not os.path.isfile('{0}/{1}/Export/objects.json'.format(output_dir, sub_dir)) and data['frames'][f]['frame_rois'][r]['roi_label']['label_name'] != 'Unsorted':
                    output = '{0}/{1}/Export/{2}/{3:06d}_{4:06d}.jpg'.format(output_dir, sub_dir, data['frames'][f]['frame_rois'][r]['roi_label']['label_name'], data['frames'][f]['frame_id'], data['frames'][f]['frame_rois'][r]['roi_id'])
                    print output
                    ImageIO.saveImg(output, img, roi)
    if not os.path.isfile('{0}/{1}/All/objects.json'.format(output_dir, sub_dir)):
        shutil.copy(os.path.join(os.environ['FS_ROOT'], args.objects_list), '{0}/{1}/All/objects.json'.format(output_dir, sub_dir))
    if not os.path.isfile('{0}/{1}/Export/objects.json'.format(output_dir, sub_dir)):
        shutil.copy('{0}/{1}/All/objects.json'.format(output_dir, sub_dir), '{0}/{1}/Export/objects.json'.format(output_dir, sub_dir))

if __name__ == '__main__':
    main(sys.argv)

