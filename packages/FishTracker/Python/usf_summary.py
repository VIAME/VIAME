import cv2
import glob
import json
import os
import sys

def main(argv):
    if len(argv) == 1:
        src_dir = '/home/junhu/Downloads/August2017_NewTraining_LargeDataset/HD_Video_priority'
        data = dict()
        data['videos'] = []
        for file in glob.glob(os.path.join(src_dir, '*')):
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fnums  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            video = dict()
            video['path'] = file
            video['ranges'] = []
            loop = dict()
            loop['start'] = 0
            loop['stop'] = int(fnums)
            loop['step'] = int(round(fps / 6.0))
            video['ranges'].append(loop)
            data['videos'].append(video)

        writer = open('USF_HD.json', 'w')
        writer.write(json.dumps(data, indent = 4))
        writer.close()

        src_dir = '/home/junhu/Downloads/August2017_NewTraining_LargeDataset/SD_Video_backup'
        data = dict()
        data['videos'] = []
        for file in glob.glob(os.path.join(src_dir, '*')):
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fnums  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            video = dict()
            video['path'] = file
            video['ranges'] = []
            loop = dict()
            loop['start'] = 0
            loop['stop'] = int(fnums)
            loop['step'] = int(round(fps / 6.0))
            video['ranges'].append(loop)
            data['videos'].append(video)

        writer = open('USF_SD.json', 'w')
        writer.write(json.dumps(data, indent = 4))
        writer.close()
    else:
        _, config_file, tgt_dir = argv
        reader = open(config_file, 'r')
        data = json.load(reader)
        reader.close()
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        frame_cnt = 0
        for video in data['videos']:
            frame_list = []
            for loop in video['ranges']:
                frame_list += range(loop['start'], loop['stop'], loop['step'])

            print video['path']
            cap = cv2.VideoCapture(video['path'])
            fnums  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            for f in range(int(fnums)):
                ret, img = cap.read()
                if ret:
                    if f in frame_list:
                        output = '{0}/{1:06d}.jpg'.format(tgt_dir, frame_cnt)
                        frame_cnt += 1
                        print 'Frame[{0}] -> {1}'.format(f, output)
                        cv2.imwrite(output, img)

if __name__ == '__main__':
    main(sys.argv)

