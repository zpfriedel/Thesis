import cv2 as cv
import h5py
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm
from model_utils import box_nms
from utils import data_gen_hpatches
from demo_superpoint import SuperPointNet, SuperPointFrontend


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

basepath = '/home/ubuntu/data'

def classical_detector_descriptor(im, **config):
    if config['model'] == 'sift':
        im = np.uint8(im * 255)
        sift = cv.xfeatures2d.SIFT_create(nfeatures=1500)
        keypoints, desc = sift.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)

        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 128), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc

    elif config['model'] == 'orb':
        im = np.uint8(im * 255)
        orb = cv.ORB_create(nfeatures=1500)
        keypoints, desc = orb.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)

        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 32), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc

    elif config['model'] == 'fastfreak':
        im = np.uint8(im * 255)
        fast = cv.FastFeatureDetector_create(15)
        freak = cv.xfeatures2d.FREAK_create()
        keypoints = fast.detect(im)
        keypoints, desc = freak.compute(im, keypoints)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)

        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 64), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc

    elif config['model'] == 'pretrained_super_point':
        weights_path = '/home/ubuntu/data/superpoint_v1.pth'
        fe = SuperPointFrontend(weights_path=weights_path,
                                nms_dist=config['nms'],
                                conf_thresh=0.015,
                                nn_thresh=0.7,
                                cuda=True)
        points, desc, detections = fe.run(im[:, :, 0])
        points = points.astype(int)
        descriptors = np.zeros((im.shape[0], im.shape[1], 256), np.float)
        descriptors[points[1, :], points[0, :]] = np.transpose(desc)

    detections = detections.astype(np.float32)
    descriptors = descriptors.astype(np.float32)
    return (detections, descriptors)

with open('configs/config_classical_hpatches_descriptors.yaml', 'r') as f:
    config = yaml.load(f)

picklefile = Path(basepath, config['picklefile'])
with open(picklefile, 'rb') as handle:
    files = pickle.load(handle)

output_dir = Path(basepath, config['export_name'])
if not output_dir.exists():
    os.makedirs(output_dir, exist_ok=True)
h5file = h5py.File(Path(basepath, config['h5_export_name']), 'w')

with open(basepath + '/' + config['export_name'] + '/' + 'config.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)

export_gen = data_gen_hpatches(files, norm=config['normalize'], **config)
dataset = export_gen.make_one_shot_iterator().get_next()

with tf.device('/cpu:0'):
    probability1, descriptor1 = tf.map_fn(lambda i: tf.py_func(lambda x: classical_detector_descriptor(x, **config),
        [i], (tf.float32, tf.float32)), dataset['image'], [tf.float32, tf.float32])
    probability2, descriptor2 = tf.map_fn(lambda i: tf.py_func(lambda x: classical_detector_descriptor(x, **config),
        [i], (tf.float32, tf.float32)), dataset['warped_image'], [tf.float32, tf.float32])

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
            prob1, prob2, desc1, desc2, image, warped_image, homography =\
                tf.keras.backend.get_session().run([probability1, probability2, descriptor1, descriptor2,
                                                    dataset['image'], dataset['warped_image'], dataset['homography']])
        else:
            prob1, prob2, desc1, desc2, image, warped_image, homography = \
                tf.keras.backend.get_session().run([probability1, probability2, descriptor1, descriptor2,
                                                    dataset['image'], dataset['warped_image'],
                                                    dataset['homography']['H']])
        data = {'image': np.squeeze(image, axis=0), 'warped_image': np.squeeze(warped_image, axis=0),
                'homography': np.squeeze(homography, axis=0)}
        pred = {'prob': np.squeeze(prob1, axis=0), 'warped_prob': np.squeeze(prob2, axis=0),
                'desc': np.squeeze(desc1, axis=0), 'warped_desc': np.squeeze(desc2, axis=0),
                'homography': np.squeeze(homography, axis=0)}

        if not ('name' in data):
            pred.update(data)

        filename = data['name'].decode('utf-8') if 'name' in data else str(i)
        for key, value in zip(pred.keys(), pred.values()):
            file = filename + '/' + key
            h5file.create_dataset(file, data=value)
        i += 1
        pbar.update(1)
        if i == config['eval_iter']:
            break

    except tf.errors.OutOfRangeError:
        h5file.close()
        break