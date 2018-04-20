import glob
import json
import os
import shutil

labels = ['FISH_Angelfish',
          'FISH_Creolefish',
          'FISH_Grouper',
          'FISH_Hogfish',
          'FISH_Jack',
          'FISH_Lionfish',
          'FISH_Porgy',
          'FISH_Snapper',
          'FISH_Stingray',
          'FISH_Triggerfish',
          'HABITAT_BIO_FaunalBed',
          'HABITAT_BIO_FaunalBed_Alcyonacea',
          'HABITAT_BIO_FaunalBed_Sponge',
          'HABITAT_GEO_Hard',
          'HABITAT_GEO_Hard_AnthropogenicStructure',
          'HABITAT_GEO_Hard_RockOutcrop',
          'HABITAT_GEO_Soft_Flat',
          'HABITAT_GEO_Soft_Hummocky',
          'HABITAT_GEO_Soft_Ripple',
          'Negative']
dest_dir = '/docker/data/USF2/labeled'
for label in labels:
    if not os.path.exists(os.path.join(dest_dir, label)):
        os.mkdir(os.path.join(dest_dir, label))

annotations = dict()
annotations['folders'] = []
for root, dirs, files in os.walk('/docker/data/USF2/annotated'):
    for dir in dirs:
        folder = dict()
        folder['path'] = os.path.join(root, dir)
        words = dir.split('_')
        if 'REVISED' not in words:
            if all(x not in words for x in ['TOO', 'NO', 'BLURRED', 'Blurred', 'LargeNoID', 'UNKNOWN']):
                folder['tags'] = []
                for word in words:
                    if not word.isdigit():
                        if word == 'Amberjack':
                            folder['tags'].append('Jack')
                        elif word == 'ARTIFICIAL':
                            folder['tags'].append('GEO')
                            folder['tags'].append('AnthropogenicStructure')
                        elif word == 'Faunalbed':
                            folder['tags'].append('FaunalBed')
                        elif word == 'Ripples':
                            folder['tags'].append('Ripple')
                        elif word == 'Sponges':
                            folder['tags'].append('Sponge')
                        elif word == 'wGravel':
                            folder['tags'].append('wGravel')
                            if 'Flat' not in folder['tags']:
                                folder['tags'].append('Flat')
                        else:
                            folder['tags'].append(word)
                annotations['folders'].append(folder)
            else:
                for image in glob.glob(os.path.join(folder['path'], '*')):
                    shutil.copy(image, os.path.join(dest_dir, 'Negative'))
writer=open('/docker/data/USF2/usf2_annotation.json', 'w')
writer.write(json.dumps(annotations, indent = 4))
writer.close()

for folder in annotations['folders']:
    label_cnt = 0;
    if 'Angelfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Angelfish'))
    if 'Creolefish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Creolefish'))
    if 'Grouper' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Grouper'))
    if 'Hogfish' in folder['tags'] or 'Boxfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Hogfish'))
    if 'Jack' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Jack'))
    if 'Lionfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Lionfish'))
    if 'Porgy' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Porgy'))
    if 'Snapper' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Snapper'))
    if 'Stingray' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Stingray'))
    if 'Triggerfish' in folder['tags']:
        label_cnt += 1;
        for image in glob.glob(os.path.join(folder['path'], '*')):
            shutil.copy(image, os.path.join(dest_dir, 'FISH_Triggerfish'))
    if 'FaunalBed' in folder['tags']:
        if 'Alcyonacea' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed_Alcyonacea'))
        elif 'Sponge' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed_Sponge'))
        else:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_BIO_FaunalBed'))
    if 'Hard' in folder['tags']:
        if 'RockOutcrop' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Hard_RockOutcrop'))
        elif 'AnthropogenicStructure' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Hard_AnthropogenicStructure'))
        else:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Hard'))
    if 'Soft' in folder['tags']:
        if 'Flat' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Soft_Flat'))
        if 'Hummocky' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Soft_Hummocky'))
        if 'Ripple' in folder['tags']:
            label_cnt += 1;
            for image in glob.glob(os.path.join(folder['path'], '*')):
                shutil.copy(image, os.path.join(dest_dir, 'HABITAT_GEO_Soft_Ripple'))

    if label_cnt != 1:
        print folder['path'], label_cnt

