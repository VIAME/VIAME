import os
import sys

from SpectralClustering import *

def main(argv):
    if len(argv) == 1:
        param = dict()
        param['clusteringType'] = 'kmeans'
        param['threshMethod'] = 'byValue'
        param['page_size'] = 3000
        param['top_k'] = 100
        param['diff_thresh'] = 0.6
        param['eigen_num'] = 300
        param['label_offset'] = 0
        param['std_thresh'] = 0.75
        param['deviate_ratio_thresh'] = 2.0
        writer = open('blind_seg.json','wt')
        writer.write(json.dumps(param, indent = 4))
        writer.close()
    else:
        _, config_file, datadir = argv
        reader = open(os.path.join(os.environ['FS_ROOT'], config_file), 'rt')
        param = json.load(reader)
        reader.close()
        clusteringType = param['clusteringType']
        threshMethod = param['threshMethod']
        page_size = param['page_size']
        top_k = param['top_k']
        diff_thresh = param['diff_thresh']
        eigen_num = param['eigen_num']
        label_offset = param['label_offset']
        std_thresh = param['std_thresh']
        deviate_ratio_thresh = param['deviate_ratio_thresh']

        sample_dir_ = os.path.join(os.environ['FS_ROOT'], datadir, 'All/images')
        json_file_ = os.path.join(os.environ['FS_ROOT'], datadir, 'All/features.json')
        output_folder_ = os.path.join(os.environ['FS_ROOT'], datadir, 'Cluster')
        if not os.path.isdir(output_folder_):
            os.mkdir(output_folder_)
    
        namelist_, featurelist_ = data_read(sample_dir_, json_file_)
        label_list_, keyimage_list_ = data_segment(featurelist_, page_size, top_k, diff_thresh, threshMethod, clusteringType)
        #label_list_, clusters_, keyimage_list_, cluster_feature_list_ = data_hierarchy_segment(featurelist_, page_size, top_k, diff_thresh, threshMethod, clusteringType, std_thresh, deviate_ratio_thresh, eigen_num, label_offset)
        for i in keyimage_list_.keys():
            print i, namelist_[keyimage_list_[i]]
        outputClusters_3(namelist_, label_list_, keyimage_list_, sample_dir_, output_folder_, output_folder_)

        reader = open(os.path.join(os.environ['FS_ROOT'], datadir, 'All/objects.json'), 'rt')
        data = json.load(reader)
        reader.close()
        label_names = dict()
        for f in range(len(data['frames'])):
            for r in range(len(data['frames'][f]['frame_rois'])):
                name = '{0:06d}_{1:06d}.jpg'.format(f, r)
                if name not in namelist_:
                    print '{0} is not labeled'.format(name)
                else:
                    data['frames'][f]['frame_rois'][r]['roi_label']['label_id'] = label_list_.tolist()[namelist_.index(name)]
                    if namelist_.index(name) in keyimage_list_.values():
                        data['frames'][f]['frame_rois'][r]['roi_score'] = 0.0
                        label_names[data['frames'][f]['frame_rois'][r]['roi_label']['label_id']] = data['frames'][f]['frame_rois'][r]['roi_label']['label_name']
        print label_names
        for f in range(len(data['frames'])):
            for r in range(len(data['frames'][f]['frame_rois'])):
                if not data['frames'][f]['frame_rois'][r]['roi_label']['label_name'] == label_names[data['frames'][f]['frame_rois'][r]['roi_label']['label_id']]:
                   print f,r
                   print data['frames'][f]['frame_rois'][r]['roi_label']['label_name'], label_names[data['frames'][f]['frame_rois'][r]['roi_label']['label_id']]
                   data['frames'][f]['frame_rois'][r]['roi_label']['label_name'] = label_names[data['frames'][f]['frame_rois'][r]['roi_label']['label_id']]
                   data['frames'][f]['frame_rois'][r]['roi_score'] = 0.0
        writer = open(os.path.join(os.environ['FS_ROOT'], datadir, 'All/objects.json'), 'wt')
        writer.write(json.dumps(data, indent = 4))
        writer.close()

if __name__ == '__main__':
    main(sys.argv)

