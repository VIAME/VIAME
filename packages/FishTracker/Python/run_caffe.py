import sys
import subprocess
import json

def main(argv):
    if len(argv) == 1:
        config = dict()
        config['shell_script'] = 'run_caffe_train.sh'
        config['arguments'] = dict()
        config['arguments']['caffe_command'] = 'train'
        config['options'] = dict()
        config['options']['weights'] = 'bvlc_reference_caffenet.caffemodel'

        config['options']['solver'] = 'fish_type/solver.prototxt'
        writer = open('fish_type_train.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['solver'] = 'mbari_type/solver.prototxt'
        writer = open('mbari_type_train.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['solver'] = 'usf_type/solver.prototxt'
        writer = open('usf_type_train.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()
    else:
        _, config_file = argv
        reader = open(config_file, 'rt')
        config = json.load(reader)
        reader.close()
        cmdline = ['./' + str(config['shell_script'])]
        for key, value in config['arguments'].iteritems():
            cmdline.append(str(value))
        for key, value in config['options'].iteritems():
            cmdline.append('--' + str(key) + '=' + str(value))
        print cmdline
        pid = subprocess.Popen(cmdline).pid
        print str(pid)

if __name__ == '__main__':
    main(sys.argv)

