import h5py
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm
from model_utils import total_loss, precision_metric, recall_metric, warped_precision_metric, warped_recall_metric,\
    threshold_precision_metric, threshold_recall_metric, warped_threshold_precision_metric,\
    warped_threshold_recall_metric, repeatability_metric, box_nms
from utils import data_gen_hpatches


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

basepath = '/home/ubuntu/data'

with open('configs/config_sp_hpatches_descriptors.yaml', 'r') as f:
    config = yaml.load(f)

model = tf.keras.models.load_model(basepath + '/' + config['model'],
                                   custom_objects={'total_loss': total_loss,
                                                   'precision': precision_metric(0),
                                                   'recall': recall_metric(0),
                                                   'warped_precision': warped_precision_metric(0),
                                                   'warped_recall': warped_recall_metric(0),
                                                   'threshold_precision': threshold_precision_metric(0),
                                                   'threshold_recall': threshold_recall_metric(0),
                                                   'warped_threshold_precision': warped_threshold_precision_metric(0),
                                                   'warped_threshold_recall': warped_threshold_recall_metric(0),
                                                   'repeatability': repeatability_metric(np.zeros((1, 1), np.int32),
                                                                                         np.zeros((1, 1), np.int32))})
model.summary()

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

if config['preprocessing']['resize']:
    H1, W1, margin = tf.constant(config['preprocessing']['resize'][0]), tf.constant(config['preprocessing']['resize'][1]),\
                   tf.constant(config['valid_border_margin'])
    H2, W2 = H1, W1
else:
    H1, W1, H2, W2, margin = tf.constant(240), tf.constant(320), tf.constant(216), tf.constant(288),\
                             tf.constant(config['valid_border_margin'])

if not config['events']:
    outputs = model([dataset['image'], dataset['warped_image']])
else:
    warped_image = tf.pad(dataset['warped_image'][0, ..., 0], paddings=tf.constant([[0, 24], [0, 32]]),
                          mode='CONSTANT', constant_values=0)[tf.newaxis, ..., tf.newaxis]
    outputs = model([dataset['image'], warped_image])

probability1 = tf.nn.softmax(outputs[..., :65], axis=-1)
probability1 = tf.squeeze(tf.depth_to_space(probability1[:, :, :, :-1], block_size=8), axis=-1)
probability2 = tf.nn.softmax(outputs[..., 65:130], axis=-1)
probability2 = tf.squeeze(tf.depth_to_space(probability2[:, :, :, :-1], block_size=8), axis=-1)

if config['events']:
    probability2 = probability2[:, :216, :288]
    probability2 = probability2 * dataset['homography']['mask']

probability1 = tf.image.crop_to_bounding_box(probability1[..., tf.newaxis], margin, margin,
                                             H1 - 2*margin, W1 - 2*margin)
probability1 = tf.squeeze(tf.image.pad_to_bounding_box(probability1, margin, margin, H1, W1), axis=-1)
probability2 = tf.image.crop_to_bounding_box(probability2[..., tf.newaxis], margin, margin,
                                             H2 - 2*margin, W2 - 2*margin)
probability2 = tf.squeeze(tf.image.pad_to_bounding_box(probability2, margin, margin, H2, W2), axis=-1)

probability1 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                           keep_top_k=config['top_k']), probability1)
probability2 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                           keep_top_k=config['top_k']), probability2)

descriptor1 = tf.image.resize_bicubic(outputs[..., 130:386],
                                      config['grid_size'] * tf.shape(outputs[..., 130:386])[1:3])
descriptor1 = tf.nn.l2_normalize(descriptor1, axis=-1)
descriptor2 = tf.image.resize_bicubic(outputs[..., 386:],
                                      config['grid_size'] * tf.shape(outputs[..., 386:])[1:3])
descriptor2 = tf.nn.l2_normalize(descriptor2, axis=-1)

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