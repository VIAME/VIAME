import glob
import json
import os
import subprocess
import sys

def main(argv):
    _, shell_cmd, label_file, datapath = argv
    folders = []
    for f in glob.glob(os.path.join(datapath, '*')):
        if os.path.isdir(f):
            folders.append(os.path.basename(f))
    label = dict()
    label['data_path'] = datapath
    label['neg_dics'] = dict()
    label['pos_dics'] = dict()
    if 'negative' in folders:
        label['neg_dics']['negative'] = 0
        next = 1
    elif 'Negative' in folders:
        label['neg_dics']['Negative'] = 0
        next = 1
    else:
        next = 0
    for folder in sorted(folders):
        if folder not in ['negative', 'Negative']:
           label['pos_dics'][folder] = next
           next += 1

    model_name = os.path.splitext(os.path.basename(label_file))[0]
    cmdline = [shell_cmd, model_name, str(next)]
    print cmdline
    p = subprocess.Popen(cmdline)
    print str(p.pid)
    p.wait()

    print json.dumps(label, indent = 4)
    writer = open(label_file,'wt')
    writer.write(json.dumps(label, indent = 4))
    writer.close()

if __name__ == '__main__':
    main(sys.argv)

