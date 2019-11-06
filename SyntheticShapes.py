import numpy as np
import yaml
import cv2 as cv
import os
import tarfile
import shutil
import pickle
from pathlib import Path
from tqdm import tqdm
import synthetic_dataset


def dump_primitive_data(primitive, tar_path, config):
    """
    Generate synthetic shapes dataset

    Arguments:
        primitive: shape to draw
        tar_path: path to dump primitive shape and points
    """
    temp_dir = Path(os.environ['TMPDIR'], primitive)
    synthetic_dataset.set_random_state(np.random.RandomState(config['generation']['random_seed']))

    for split, size in config['generation']['split_sizes'].items():
        im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
        im_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=primitive+'/'+split, leave=True):
            image = synthetic_dataset.generate_background(
                config['generation']['image_size'],
                **config['generation']['params']['generate_background'])
            points = np.array(getattr(synthetic_dataset, primitive)(
                image, **config['generation']['params'].get(primitive, {})))
            points = np.flip(points, 1)  # reverse convention with opencv

            b = config['preprocessing']['blur_size']
            image = cv.GaussianBlur(image, (b, b), 0)
            points = (points * np.array(config['preprocessing']['resize'], np.float)
                      / np.array(config['generation']['image_size'], np.float))
            image = cv.resize(image, tuple(config['preprocessing']['resize'][::-1]), interpolation=cv.INTER_LINEAR)

            cv.imwrite(str(Path(im_dir, '{}.png'.format(i))), image)
            np.save(Path(pts_dir, '{}.npy'.format(i)), points)

    # Pack into a tar file
    tar = tarfile.open(str(tar_path), mode='w:gz')
    tar.add(str(temp_dir), arcname=primitive)
    tar.close()
    shutil.rmtree(str(temp_dir))


with open('configs/config_ss.yaml', 'r') as f:
    config = yaml.load(f)

basepath = Path('/home/ubuntu/data', 'SyntheticShapes')
basepath.mkdir(parents=True, exist_ok=True)
splits = {s: {'images': [], 'points': []} for s in ['training', 'validation', 'test']}

primitives = config['drawing_primitives']

for primitive in primitives:
    tar_path = Path(basepath, '{}.tar.gz'.format(primitive))
    if not tar_path.exists():
        dump_primitive_data(primitive, tar_path, config)

    tar = tarfile.open(str(tar_path))
    temp_dir = Path(basepath, 'Shapes')
    if not Path(temp_dir, '{}'.format(primitive)).exists():
        tar.extractall(path=str(temp_dir))
    tar.close()

    truncate = config['truncate'].get(primitive, 1)
    path = Path(str(temp_dir), primitive)
    for s in splits:
        e = [str(p) for p in Path(str(path), 'images', s).iterdir()]
        f = [p.replace('images', 'points') for p in e]
        f = [p.replace('.png', '.npy') for p in f]
        splits[s]['images'].extend(e[:int(truncate * len(e))])
        splits[s]['points'].extend(f[:int(truncate * len(f))])

for s in splits:
    perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
    for obj in ['images', 'points']:
        splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

picklefile = Path(basepath, config['picklefile'])
with open(picklefile, 'wb') as handle:
    pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)