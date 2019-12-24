import numpy as np
import pickle
import yaml
from pathlib import Path

mode = 'events'

if mode == 'hpatches':
    with open('configs/config_mp_hpatches_repeatability.yaml', 'r') as f:
        config = yaml.load(f)
elif mode == 'events':
    with open('configs/config_events_repeatability.yaml', 'r') as f:
        config = yaml.load(f)

dataset_folder = 'COCO/patches' if config['dataset'] == 'events' else 'HPatches'
base_path = Path('/home/ubuntu/data', dataset_folder)
folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
image_paths = []
warped_image_paths = []
homographies = []

for path in folder_paths:
    if config['alteration'] == 'i' and path.stem[0] != 'i':
        continue
    if config['alteration'] == 'v' and path.stem[0] != 'v':
        continue
    num_images = 1 if config['dataset'] == 'events' else 5
    file_ext = '.ppm' if config['dataset'] == 'hpatches' else '.jpg'
    for i in range(2, 2 + num_images):
        image_paths.append(str(Path(path, '1' + file_ext)))
        warped_image_paths.append(str(Path(path, str(i) + file_ext)))
        homographies.append(np.loadtxt(str(Path(path, 'H_1_' + str(i)))))

files = {'image_paths': image_paths,
         'warped_image_paths': warped_image_paths,
         'homography': homographies}

picklefile = Path('/home/ubuntu/data', config['picklefile'])
with open(picklefile, 'wb') as handle:
    pickle.dump(files, handle, protocol=pickle.HIGHEST_PROTOCOL)