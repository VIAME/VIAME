import glob
import json
import os
import shutil

labels = ['FISH_GrayAngelfish',
          'FISH_GraySnapper',
          'FISH_Holocentridae',
          'FISH_Lionfish',
          'FISH_Ostracidae',
          'FISH_Priacanthidae',
          'HABITAT_BIO_FaunalBed_Alcyonacea',
          'HABITAT_BIO_FaunalBed_DiverseColonizers',
          'HABITAT_BIO_FaunalBed_Sponges',
          'HABITAT_BIO_UnknownEncrustingOrganism',
          'HABITAT_GEO_LRHB',
          'HABITAT_GEO_LRHB_Sand',
          'HABITAT_GEO_Sand',
          'HABITAT_GEO_Sand_ScatteredDebris',
          'Negative']
dest_dir = '/docker/data/USF/labeled'
for label in labels:
    if not os.path.exists(os.path.join(dest_dir, label)):
        os.mkdir(os.path.join(dest_dir, label))

annotations = dict()
annotations['folders'] = []
for root, dirs, files in os.walk('/docker/data/USF/annotated'):
    for dir in dirs:
        folder = dict()
        folder['path'] = os.path.join(root, dir)
        words = dir.split('_')
        if 'FOLDERS' not in words:
            if all(x not in words for x in ['SMALL', 'Small', 'too small', 'NoID', 'Unknown', 'Unkown', 'Unsure']):
                folder['tags'] = []
                for word in words:
                    if not word.isdigit() and word not in ['and', 'AttachedFauna', 'AttachedFanuna', 'Hard', '0Hard', '1Hard', '2Hard', '0Soft', '1Soft', '2Soft']:
                        if word == 'BIOLOGICAL':
                            folder['tags'].append('BIO')
                        elif word in ['1LRHB', '2LRHB']:
                            folder['tags'].append('LRHB')
                        elif word in ['1Sand', '2Sand']:
                            folder['tags'].append('Sand')
                        elif word in ['wScatteredDebris', '2wScatteredDebris', '3wScatteredRock']:
                            folder['tags'].append('ScatteredDebris')
                        elif word == 'Sponge':
                            folder['tags'].append('Sponges')
                        else:
                            folder['tags'].append(word)
                annotations['folders'].append(folder)
            else:
                for image in glob.glob(os.path.join(folder['path'], '*')):
                    shutil.copy(image, os.path.join(dest_dir, 'Negative'))
writer=open('/docker/data/USF/usf_annotation.json', 'w')
writer.write(json.dumps(annotations, indent = 4))
writer.close()

for folder in annotations['folders']:
    label_cnt = 0;
    if 'GrayAngelfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_GrayAngelfish'))
    elif 'Holocentridae' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Holocentridae'))
    if 'GraySnapper' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_GraySnapper'))
    if 'Ostracidae' in folder['tags'] or 'Boxfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Ostracidae'))
    if 'Priacanthidae' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Priacanthidae'))
    if 'Lionfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Lionfish'))
    if 'Alcyonacea' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed_Alcyonacea'))
    if 'DiverseColonizers' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed_DiverseColonizers'))
    if 'Sponges' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed_Sponges'))
    if 'UnknownEncrustingOrganism' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_UnknownEncrustingOrganism'))
    if 'LRHB' in folder['tags'] and 'Sand' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_LRHB_Sand'))
    elif 'Sand' in folder['tags'] and 'ScatteredDebris' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Sand_ScatteredDebris'))
    elif 'Sand' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Sand'))
    elif 'LRHB' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_LRHB'))
    if label_cnt == 0:
        print folder['path'], label_cnt

