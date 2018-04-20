import sys
import subprocess
import json

def main(argv):
    if len(argv) == 1:
        label = dict()
        label['data_path'] = '/docker/data/MBARI/annotated'
        label['pos_dics'] = dict()
        label['neg_dics'] = dict()
        label['pos_dics']['black_eelpout'] = 1
        label['pos_dics']['crab'] = 2
        label['pos_dics']['longnose_skate'] = 3
        label['pos_dics']['starfish'] = 4
        label['pos_dics']['northpacific_hakefish'] = 5
        label['pos_dics']['rockfish'] = 6
        label['pos_dics']['sea_anemone'] = 7
        label['pos_dics']['seasnail'] = 8
        label['pos_dics']['seaurchin'] = 9
        label['pos_dics']['sunflowerstar'] = 10
        label['neg_dics']['negative'] = 0
        writer = open('fish_type.json','wt')
        writer.write(json.dumps(label, indent = 4))
        writer.close()

        label['data_path'] = '/docker/data/MBARI/labeled'
        label['pos_dics'].clear()
        label['neg_dics'].clear()
        label['pos_dics']['Actinostolidae'] = 1
        label['pos_dics']['Calliostoma_platinum'] = 2
        label['pos_dics']['Chionoecetes_tanneri'] = 3
        label['pos_dics']['Glyptocephalus_zachirus'] = 4
        label['pos_dics']['Lycodes_diapterus'] = 5
        label['pos_dics']['Mediaster_aequalis'] = 6
        label['pos_dics']['Merluccius_productus'] = 7
        label['pos_dics']['Porifera'] = 8
        label['pos_dics']['Psolus_squamatus'] = 9
        label['pos_dics']['Raja_rhina'] = 10
        label['pos_dics']['Rathbunaster_californicus'] = 11
        label['pos_dics']['Sebastes_2species'] = 12
        label['pos_dics']['Sebastolobus_altivelis'] = 13
        label['pos_dics']['Strongylocentrotus_fragilis'] = 14
        label['pos_dics']['Stylasterias_forreri'] = 15
        label['neg_dics']['Negative'] = 0
        writer = open('mbari_type.json','wt')
        writer.write(json.dumps(label, indent = 4))
        writer.close()

        label['data_path'] = '/docker/data/USF/labeled'
        label['pos_dics'].clear()
        label['neg_dics'].clear()
        label['pos_dics']['FISH_GrayAngelfish'] = 1
        label['pos_dics']['FISH_GraySnapper'] = 2
        label['pos_dics']['FISH_Holocentridae'] = 3
        label['pos_dics']['FISH_Lionfish'] = 4
        label['pos_dics']['FISH_Ostracidae'] = 5
        label['pos_dics']['FISH_Priacanthidae'] = 6
        label['pos_dics']['HABITAT_BIO_FaunalBed_Alcyonacea'] = 7
        label['pos_dics']['HABITAT_BIO_FaunalBed_DiverseColonizers'] = 8
        label['pos_dics']['HABITAT_BIO_FaunalBed_Sponges'] = 9
        label['pos_dics']['HABITAT_BIO_UnknownEncrustingOrganism'] = 10
        label['pos_dics']['HABITAT_GEO_LRHB'] = 11
        label['pos_dics']['HABITAT_GEO_LRHB_Sand'] = 12
        label['pos_dics']['HABITAT_GEO_Sand'] = 13
        label['pos_dics']['HABITAT_GEO_Sand_ScatteredDebris'] = 14
        label['neg_dics']['Negative'] = 0
        writer = open('usf_type.json','wt')
        writer.write(json.dumps(label, indent = 4))
        writer.close()

        config = dict()
        config['shell_script'] = 'run_fish_type.sh'
        config['python_file'] = 'Inference.py'
        config['arguments'] = dict()
        config['arguments']['input_file'] = ''
        config['options'] = dict()
        config['options']['gpu'] = 'True'
        config['options']['mean_file'] = 'model/ilsvrc_2012_mean.npy'
        config['options']['largeRoIRatio'] = '0.5'
        config['options']['goodRoIRatio'] = '0.05'
        config['options']['smallRoIRatio'] = '0.01'

        config['options']['pretrained_model'] = 'model/snapshot/fish_type_iter_80000.caffemodel'
        config['options']['model_def'] = 'model/fish_type/deploy.prototxt'
        config['options']['model_type'] = 'model/fish_type/fish_type.json'
        config['options']['mode'] = 'flow'
        config['options']['flowEnable'] = 'False'
        config['options']['removeDominateMotion'] = 'False'
        config['options']['linearStretch'] = 'False'
        config['options']['ks'] = '400'
        config['options']['flowthr'] = '40.0'
        writer = open('raw_video.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['pretrained_model'] = 'model/snapshot/fish_type_iter_80000.caffemodel'
        config['options']['model_def'] = 'model/fish_type/deploy.prototxt'
        config['options']['model_type'] = 'model/fish_type/fish_type.json'
        config['options']['mode'] = 'sel'
        config['options']['flowEnable'] = 'True'
        config['options']['removeDominateMotion'] = 'False'
        config['options']['linearStretch'] = 'False'
        config['options']['ks'] = '400'
        config['options']['flowthr'] = '40.0'
        writer = open('prop_mbari_type.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['pretrained_model'] = 'model/snapshot/fish_type_iter_80000.caffemodel'
        config['options']['model_def'] = 'model/fish_type/deploy.prototxt'
        config['options']['model_type'] = 'model/fish_type/fish_type.json'
        config['options']['mode'] = 'sel'
        config['options']['flowEnable'] = 'True'
        config['options']['removeDominateMotion'] = 'True'
        config['options']['linearStretch'] = 'True'
        config['options']['ks'] = '200'
        config['options']['flowthr'] = '40.0'
        writer = open('prop_usf_type.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['pretrained_model'] = 'model/snapshot/mbari_type_iter_80000.caffemodel'
        config['options']['model_def'] = 'model/mbari_type/deploy.prototxt'
        config['options']['model_type'] = 'model/mbari_type/mbari_type.json'
        config['options']['mode'] = 'net'
        config['options']['flowEnable'] = 'True'
        config['options']['removeDominateMotion'] = 'False'
        config['options']['linearStretch'] = 'False'
        config['options']['ks'] = '400'
        config['options']['flowthr'] = '40.0'
        writer = open('class_mbari_type.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()

        config['options']['pretrained_model'] = 'model/snapshot/usf_type_iter_80000.caffemodel'
        config['options']['model_def'] = 'model/usf_type/deploy.prototxt'
        config['options']['model_type'] = 'model/usf_type/usf_type.json'
        config['options']['mode'] = 'net'
        config['options']['flowEnable'] = 'True'
        config['options']['removeDominateMotion'] = 'True'
        config['options']['linearStretch'] = 'True'
        config['options']['ks'] = '200'
        config['options']['flowthr'] = '40.0'
        writer = open('class_usf_type.json','wt')
        writer.write(json.dumps(config, indent = 4))
        writer.close()
    else:
        config_file = argv[1]
        reader = open(config_file, 'rt')
        config = json.load(reader)
        reader.close()
        config['arguments']['input_file'] = argv[2]
        if len(argv) > 3:
            config['options']['start'] = argv[3]
        if len(argv) > 4:
            config['options']['stop'] = argv[4]
        cmdline = ['./' + str(config['shell_script']), './' + str(config['python_file'])]
        for key, value in config['arguments'].iteritems():
            cmdline.append(str(value))
        for key, value in config['options'].iteritems():
            cmdline.append('--' + str(key) + '=' + str(value))
        print cmdline
        p = subprocess.Popen(cmdline)
        print str(p.pid)

if __name__ == '__main__':
    main(sys.argv)

