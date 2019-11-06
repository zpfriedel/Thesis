import pickle
import yaml
from pathlib import Path


# mode: mp_train, sp_train or mp_export
mode = 'mp_export'

if mode == 'mp_train':
    with open('configs/config_mp_coco_train.yaml', 'r') as f:
        config = yaml.load(f)
elif mode == 'sp_train':
    with open('configs/config_sp_coco_train.yaml', 'r') as f:
        config = yaml.load(f)
elif mode == 'mp_export':
    with open('configs/config_mp_coco_export.yaml', 'r') as f:
        config = yaml.load(f)

basepath = Path('/home/ubuntu/data', 'COCO/train2014/')
image_paths = list(basepath.iterdir())

names = [p.stem for p in image_paths]
image_paths = [str(p) for p in image_paths]
files = {'image_paths': image_paths, 'names': names}

if config['labels']:
    label_paths = []
    for n in names:
        p = Path('/home/ubuntu/data', config['labels'], '{}.npz'.format(n))
        assert p.exists(), 'Image {} has no corresponding label {}'.format(n, p)
        label_paths.append(str(p))
    files['label_paths'] = label_paths

picklefile = Path('/home/ubuntu/data', config['picklefile'])
with open(picklefile, 'wb') as handle:
    pickle.dump(files, handle, protocol=pickle.HIGHEST_PROTOCOL)