import numpy as np
import os
import pickle
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm
from utils import data_gen_hpatches
from model_utils import detector_loss, total_loss, precision_metric, recall_metric, warped_precision_metric,\
    warped_recall_metric, box_nms


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

basepath = '/home/ubuntu/data'

# mode: mp or sp
mode = 'mp'

if mode == 'mp':
    with open('configs/config_mp_hpatches_repeatability.yaml', 'r') as f:
        config = yaml.load(f)
    model = tf.keras.models.load_model(basepath + '/' + config['model'],
                                       custom_objects={'detector_loss': detector_loss,
                                                       'precision': precision_metric(0),
                                                       'recall': recall_metric(0)})
else:
    with open('configs/config_sp_hpatches_export.yaml', 'r') as f:
        config = yaml.load(f)
    model = tf.keras.models.load_model(basepath + '/' + config['model'],
                                       custom_objects={'total_loss': total_loss,
                                                       'precision': precision_metric(0),
                                                       'recall': recall_metric(0),
                                                       'warped_precision': warped_precision_metric(0),
                                                       'warped_recall': warped_recall_metric(0)})
model.summary()

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

H, W, margin = tf.constant(config['preprocessing']['resize'][0]), tf.constant(config['preprocessing']['resize'][1]),\
               tf.constant(config['valid_border_margin'])

if mode == 'mp_repeatability':
    probability1 = model(dataset['image'])
    probability2 = model(dataset['warped_image'])
else:
    outputs = model([dataset['image'], dataset['warped_image']])
    probability1 = outputs[..., :65]
    probability2 = outputs[..., 65:130]

probability1 = tf.nn.softmax(probability1, axis=-1)
probability1 = tf.squeeze(tf.depth_to_space(probability1[:, :, :, :-1], block_size=8), axis=-1)
probability2 = tf.nn.softmax(probability2, axis=-1)
probability2 = tf.squeeze(tf.depth_to_space(probability2[:, :, :, :-1], block_size=8), axis=-1)

probability1 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                   keep_top_k=config['top_k']), probability1)
probability2 = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                   keep_top_k=config['top_k']), probability2)

probability1 = tf.image.crop_to_bounding_box(probability1[..., tf.newaxis], margin, margin,
                                             H - 2*margin, W - 2*margin)
probability1 = tf.squeeze(tf.image.pad_to_bounding_box(probability1, margin, margin, H, W), axis=-1)
probability2 = tf.image.crop_to_bounding_box(probability2[..., tf.newaxis], margin, margin,
                                             H - 2*margin, W - 2*margin)
probability2 = tf.squeeze(tf.image.pad_to_bounding_box(probability2, margin, margin, H, W), axis=-1)

tf.keras.backend.get_session().graph.finalize()

i = 0
while True:
    try:
        prob1, prob2, image, warped_image, homography = tf.keras.backend.get_session().run([probability1, probability2,
                                                                                           dataset['image'],
                                                                                           dataset['warped_image'],
                                                                                           dataset['homography']])
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