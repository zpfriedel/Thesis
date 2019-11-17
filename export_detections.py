import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import yaml
from pathlib import Path
from tqdm import tqdm
from model_utils import detector_loss, precision_metric, recall_metric, threshold_precision_metric,\
    threshold_recall_metric, box_nms
from utils import data_gen_coco
from homographies import homography_adaptation


def draw_interest_points(img, points):
    # Convert img in RGB and draw in green the interest points
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][1], points[i][0]), 1, (0, 255, 0), -1)
    return img_rgb


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]
basepath = '/home/ubuntu/data'

with open('configs/config_mp_coco_export.yaml', 'r') as f:
    config = yaml.load(f)

model = tf.keras.models.load_model(basepath + '/' + config['model'],
                                   custom_objects={'detector_loss': detector_loss,
                                                   'precision': precision_metric(0), 'recall': recall_metric(0),
                                                   'threshold_precision': threshold_precision_metric(0),
                                                   'threshold_recall': threshold_recall_metric(0)})
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

export_gen = data_gen_coco(files, 'export', batch_size=config['export_batch_size'], norm=config['normalize'], **config)
data = export_gen.make_one_shot_iterator().get_next()

H, W, margin = tf.constant(config['preprocessing']['resize'][0]), tf.constant(config['preprocessing']['resize'][1]),\
               tf.constant(config['homography_adaptation']['valid_border_margin'])

outputs = homography_adaptation(data['image'], model, config['homography_adaptation'])
prob = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                   keep_top_k=config['top_k']), outputs['prob'])
prob = tf.image.crop_to_bounding_box(prob[..., tf.newaxis], margin, margin, H - 2*margin, W - 2*margin)
prob = tf.squeeze(tf.image.pad_to_bounding_box(prob, margin, margin, H, W), axis=-1)

prediction = tf.cast(tf.greater_equal(prob, config['detection_threshold']), dtype=tf.int32)

tf.keras.backend.get_session().graph.finalize()

while True:
    try:
        image, name, pred = tf.keras.backend.get_session().run([data['image'], data['name'], prediction])
        points = {'points': [np.array(np.where(e)).T for e in pred]}
        """
        for i in range(5):
            plt.figure()
            plt.imshow(draw_interest_points(image[i,...,0], points['points'][i]))
        plt.show()
        """
        for p, i, n in zip(d2l(points), image, name):
            filename = n.decode('utf-8')
            filepath = Path(output_dir, '{}.npz'.format(filename))
            np.savez_compressed(filepath, **p)
            pbar.update(1)

    except tf.errors.OutOfRangeError:
        break