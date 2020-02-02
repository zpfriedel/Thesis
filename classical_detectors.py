import cv2 as cv
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm
from utils import data_gen_hpatches
from model_utils import box_nms
from demo_superpoint import SuperPointNet, SuperPointFrontend


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

basepath = '/home/ubuntu/data'

def classical_detector(im, **config):
    if config['model'] == 'harris':
        im = np.uint8(im * 255)
        detections = cv.cornerHarris(im, 4, 3, 0.04)

    elif config['model'] == 'sift':
        im = np.uint8(im * 255)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints, _ = sift.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses

    elif config['model'] == 'fast':
        im = np.uint8(im * 255)
        detector = cv.FastFeatureDetector_create(15)
        corners = detector.detect(im.astype(np.uint8))
        detections = np.zeros(im.shape[:2], np.float)
        for c in corners:
            detections[tuple(np.flip(np.int0(c.pt), 0))] = c.response

    elif config['model'] == 'pretrained_magic_point':
        weights_path = '/home/ubuntu/data/superpoint_v1.pth'
        fe = SuperPointFrontend(weights_path=weights_path,
                                nms_dist=config['nms'],
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=True)
        points, desc, detections = fe.run(im[:, :, 0])

    return detections.astype(np.float32)

with open('configs/config_classical_hpatches_repeatability.yaml', 'r') as f:
    config = yaml.load(f)

picklefile = Path(basepath, config['picklefile'])
with open(picklefile, 'rb') as handle:
    files = pickle.load(handle)

output_dir = Path(basepath, config['export_name'])
if not output_dir.exists():
    os.makedirs(output_dir, exist_ok=True)

with open(basepath + '/' + config['export_name'] + '/' + 'config.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)

export_gen = data_gen_hpatches(files, norm=config['normalize'], **config)
dataset = export_gen.make_one_shot_iterator().get_next()

with tf.device('/cpu:0'):
    probability1 = tf.map_fn(lambda i: tf.py_func(lambda x: classical_detector(x, **config), [i], tf.float32),
                             dataset['image'])
    probability2 = tf.map_fn(lambda i: tf.py_func(lambda x: classical_detector(x, **config), [i], tf.float32),
                             dataset['warped_image'])
if config['events']:
    probability2 = probability2 * dataset['homography']['mask']

if config['nms']:
    probability1 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=0., keep_top_k=config['top_k']), probability1)
    probability2 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=0., keep_top_k=config['top_k']), probability2)

tf.keras.backend.get_session().graph.finalize()

i = 0
while True:
    try:
        if not config['events']:
            prob1, prob2, image, warped_image, homography = tf.keras.backend.get_session().run(
                [probability1, probability2, dataset['image'], dataset['warped_image'], dataset['homography']])
        else:
            prob1, prob2, image, warped_image, homography = tf.keras.backend.get_session().run(
                [probability1, probability2, dataset['image'], dataset['warped_image'], dataset['homography']['H']])
        data = {'image': np.squeeze(image, axis=0), 'warped_image': np.squeeze(warped_image, axis=0),
                'homography': np.squeeze(homography, axis=0)}
        pred = {'prob': np.squeeze(prob1, axis=0), 'warped_prob': np.squeeze(prob2, axis=0),
                'homography': np.squeeze(homography, axis=0)}

        if not ('name' in data):
            pred.update(data)
        filename = data['name'].decode('utf-8') if 'name' in data else str(i)
        filepath = Path(output_dir, '{}.npz'.format(filename))
        np.savez_compressed(filepath, **pred)
        i += 1
        pbar.update(1)
        if i == config['eval_iter']:
            break

    except tf.errors.OutOfRangeError:
        break