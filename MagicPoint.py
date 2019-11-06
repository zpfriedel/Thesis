import pickle
import yaml
import tensorflow as tf
from pathlib import Path
from tensorflow.python.client import timeline
from utils import data_gen_ss, data_gen_coco, ModelCheckpointOptimizer
from model_utils import detector_loss, precision_metric, recall_metric,\
    encoder, encoder_build, detector_head, detector_build


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

basepath = '/home/ubuntu/data'

# mode: shapes or coco
mode = 'shapes'

if mode == 'shapes':
    with open('configs/config_mp_ss.yaml', 'r') as f:
        config = yaml.load(f)
    picklefile = Path(basepath, 'SyntheticShapes', config['picklefile'])
    with open(picklefile, 'rb') as handle:
        splits = pickle.load(handle)
    train_gen = data_gen_ss(splits, split_name='training', batch_size=config['train_batch_size'],
                            norm=config['normalize'], **config)
    validation_gen = data_gen_ss(splits, split_name='validation', batch_size=config['eval_batch_size'],
                                 norm=config['normalize'], **config)
else:
    with open('configs/config_mp_coco_train.yaml', 'r') as f:
        config = yaml.load(f)
    picklefile = Path(basepath, config['picklefile'])
    with open(picklefile, 'rb') as handle:
        files = pickle.load(handle)
    train_gen = data_gen_coco(files, split_name='training', batch_size=config['train_batch_size'],
                              norm=config['normalize'], **config)
    validation_gen = data_gen_coco(files, split_name='validation', batch_size=config['eval_batch_size'],
                                   norm=config['normalize'], **config)

with open(basepath + '/' + config['log_dir'] + '/' + 'config.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Encoder
img_in = tf.keras.Input(shape=(None, None, 1), name='image')
e = encoder()
encoder_layers = encoder_build(e, img_in)

# Detector Head
d = detector_head()
det = detector_build(d, encoder_layers, **config)

model = tf.keras.Model(img_in, det['logits'])
model.summary()
"""
tf.keras.utils.plot_model(model, config['model_visual'], show_shapes=True)
"""

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']), loss=detector_loss,
              metrics=[precision_metric(det['pred']), recall_metric(det['pred'])],
              options=run_options, run_metadata=run_metadata)

if not config['pretrained_model']:
    pass
else:
    model.load_weights(basepath + '/' + config['pretrained_weights'])
    """
    model._make_train_function()
    with open(basepath + '/' + config['pretrained_optimizer'], 'rb') as opt:
        weight_values = pickle.load(opt)
    model.optimizer.set_weights(weight_values)
    """

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=basepath + '/' + config['log_dir'])
checkpoint_model = tf.keras.callbacks.ModelCheckpoint(filepath=basepath + '/' + config['checkpoint_model'],
                                                               monitor='val_loss', save_best_only=False)
checkpoint_weights = tf.keras.callbacks.ModelCheckpoint(filepath=basepath + '/' + config['checkpoint_weights'],
                                                                 monitor='val_loss', save_best_only=False,
                                                                 save_weights_only=True)
checkpoint_optimizer = ModelCheckpointOptimizer(filepath=basepath + '/' + config['checkpoint_optimizer'],
                                                monitor='val_loss', save_best_only=False)

history = model.fit(x=train_gen, steps_per_epoch=config['steps_per_epoch'], epochs=config['epochs'],
                    validation_data=validation_gen, validation_steps=config['validation_steps'],
                    callbacks=[tensorboard_callback, checkpoint_model, checkpoint_weights, checkpoint_optimizer])

"""
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('/home/ubuntu/data/mp_timeline.json', 'w') as f:
    f.write(ctf)
"""