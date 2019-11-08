import cv2 as cv
import logging
import numpy as np
import os
import pickle
import tensorflow as tf
import photometric_augmentation as photaug
from homographies import sample_homography, compute_valid_mask, warp_points, filter_points


def photometric_augmentation(data, **config):
    """
    Perform homographic augmentation on images and keypoints

    Arguments:
        data: tensorflow dataset
    Return:
        data: tensorflow dataset with warped image
    """
    primitives = photaug.augmentations
    prim_configs = [config['params'].get(p, {}) for p in primitives]

    indices = tf.range(len(primitives))
    if config['random_order']:
        indices = tf.random.shuffle(indices)

    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c))
                    for j, (p, c) in enumerate(zip(primitives, prim_configs))]
        image = tf.case(fn_pairs)
        return i + 1, image

    _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                             step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    """
    Perform homographic augmentation on images and keypoints

    Arguments:
        data: tensorflow dataset
    Return:
        ret: tensorflow dataset with warped image, keypoints and mask (optionally, homography matrix)
    """
    image_shape = tf.shape(data['image'])[:2]
    homography = sample_homography(image_shape, **config['params'])[0]
    warped_image = tf.contrib.image.transform(data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, config['valid_border_margin'])

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points, 'valid_mask': valid_mask}
    if add_homography:
        ret['homography'] = homography
    return ret


def add_dummy_valid_mask(data, mode='training', adjust=False, **config):
    """
    Adds a mask to ignore sections of a warped image
    """
    valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    if mode == 'validation' and not adjust:
        H, W, margin = tf.constant(config['preprocessing']['resize'][0]),\
                       tf.constant(config['preprocessing']['resize'][1]),\
                       tf.constant(config['augmentation']['homographic']['valid_border_margin'])
        valid_mask = tf.image.crop_to_bounding_box(valid_mask[tf.newaxis, ..., tf.newaxis], margin, margin,
                                                   H - 2 * margin, W - 2 * margin)
        valid_mask = tf.squeeze(tf.squeeze(tf.image.pad_to_bounding_box(valid_mask, margin, margin, H, W), axis=-1),
                                axis=0)
    if mode == 'training' and adjust:
        H, W, margin = tf.constant(config['preprocessing']['resize'][0]), \
                       tf.constant(config['preprocessing']['resize'][1]), \
                       tf.constant(config['augmentation']['homographic']['valid_border_margin'])
        valid_mask = tf.image.crop_to_bounding_box(data['valid_mask'][tf.newaxis, ..., tf.newaxis], margin, margin,
                                                   H - 2 * margin, W - 2 * margin)
        valid_mask = tf.squeeze(tf.squeeze(tf.image.pad_to_bounding_box(valid_mask, margin, margin, H, W), axis=-1),
                                axis=0)

    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data, add_homography=False, sp=False, norm=False, size=None):
    """
    Creates a keypoint map for the "truth" keypoints and prepares data for model
    """
    if size is None:
        size = [240, 320]
    n = tf.constant(255.)
    image_shape = tf.shape(data['image'])[:2]
    kp = tf.minimum(tf.cast(tf.round(data['keypoints']), dtype=tf.int32), image_shape-1)
    kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)

    if sp and not add_homography:
        return {**data, 'keypoint_map': kmap}
    elif sp and add_homography:
        kmap = tf.cast(kmap[..., tf.newaxis], dtype=tf.float32)
        valid_mask = tf.cast(data['valid_mask'][..., tf.newaxis], dtype=tf.float32)
        warped_kmap = tf.cast(data['warped']['keypoint_map'][..., tf.newaxis], dtype=tf.float32)
        warped_valid_mask = tf.cast(data['warped']['valid_mask'][..., tf.newaxis], dtype=tf.float32)
        homography = tf.pad(data['warped']['homography'][..., tf.newaxis],
                            paddings=tf.constant([[0, size[0] - 8], [0, size[1] - 1]]),
                            mode='CONSTANT', constant_values=0)
        homography = homography[..., tf.newaxis]
        label = tf.concat([kmap, valid_mask, warped_kmap, warped_valid_mask, homography], axis=-1)
        if norm:
            return {'image': tf.cast(data['image'] / n, dtype=tf.float32),
                    'warped_image': tf.cast(data['warped']['image'] / n, dtype=tf.float32)}, label
        else:
            return {'image': tf.cast(data['image'], dtype=tf.float32),
                    'warped_image': tf.cast(data['warped']['image'], dtype=tf.float32)}, label
    else:
        kmap = kmap[..., tf.newaxis]
        valid_mask = data['valid_mask']
        valid_mask = valid_mask[..., tf.newaxis]
        label = tf.concat([kmap, valid_mask], axis=-1)
        if norm:
            return {'image': tf.cast(data['image'] / n, dtype=tf.float32)}, label
        else:
            return {'image': tf.cast(data['image'], dtype=tf.float32)}, label


def ratio_preserving_resize(image, **config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.cast(new_size, dtype=tf.int32), method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1])


def get_ss_data(files, split_name, norm, **config):
    """
    Read data from file

    Arguments:
        files: image and point filenames for training, validation and test sets
        split_name: training, validation or test
        norm: normalize images (True or False)
    Return:
        data: tensorflow dataset for given split_name
    """
    def _read_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=1)
        return tf.cast(image, tf.float32)

    def _read_points(filename):
        return np.load(filename.decode('utf-8')).astype(np.float32)

    # Initialize dataset with file names
    data = tf.data.Dataset.from_tensor_slices((files[split_name]['images'], files[split_name]['points']))
    # Read image and point coordinates
    data = data.map(lambda image, points: (_read_image(image), tf.py_func(_read_points, [points], tf.float32)))
    data = data.map(lambda image, points: (image, tf.reshape(points, [-1, 2])))

    if split_name == 'validation':
        data = data.take(config['validation_size'])

    data = data.map(lambda image, kp: {'image': image, 'keypoints': kp})
    data = data.map(lambda d: add_dummy_valid_mask(d, mode=split_name, **config), num_parallel_calls=None)

    # Apply augmentation
    if split_name == 'training':
        if config['augmentation']['photometric']['enable']:
            data = data.map(lambda d: photometric_augmentation(d, **config['augmentation']['photometric']),
                            num_parallel_calls=None)
        if config['augmentation']['homographic']['enable']:
            data = data.map(lambda d: homographic_augmentation(d, **config['augmentation']['homographic']),
                            num_parallel_calls=None)

    # Convert the point coordinates to a dense keypoint map
    data = data.map(lambda d: add_keypoint_map(d, norm=norm), num_parallel_calls=None)

    return data


def data_gen_ss(splits, split_name, batch_size, norm, **config):
    """
    Data Generator for ML model

    Arguments:
        splits: filenames for training, validation and test sets
        split_name: training, validation or test
        batch_size: size of batch for given split_name
        norm: normalize images (True or False)
    """
    data = get_ss_data(splits, split_name, norm, **config)
    data = data.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    data = data.make_one_shot_iterator()

    return data


def get_coco_data(files, split_name, mode, norm, **config):
    """
    Read data from file

    Arguments:
        files: image and point filenames for training, validation and export sets
        split_name: training, validation or export
        mode: Magicpoint or Superpoint
        norm: normalize images (True or False)
    Return:
        data: tensorflow dataset for given split_name
    """
    n = tf.constant(255.)

    def _read_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        return image

    def _preprocess(image):
        image = tf.image.rgb_to_grayscale(image)
        if config['preprocessing']['resize']:
            image = ratio_preserving_resize(image, **config['preprocessing'])
        return image

    def _read_points(filename):
        return np.load(filename.decode('utf-8'))['points'].astype(np.float32)

    def rescale(image):
        return tf.cast(image / n, dtype=tf.float32)

    names = tf.data.Dataset.from_tensor_slices(files['names'])
    images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
    images = images.map(_read_image)
    images = images.map(_preprocess)
    if split_name == 'export':
        if norm:
            images = images.map(rescale)

    data = tf.data.Dataset.zip({'image': images, 'name': names})

    has_keypoints = 'label_paths' in files
    is_training = split_name == 'training'

    # Add keypoints
    if has_keypoints:
        kp = tf.data.Dataset.from_tensor_slices(files['label_paths'])
        kp = kp.map(lambda path: tf.py_func(_read_points, [path], tf.float32))
        kp = kp.map(lambda points: tf.reshape(points, [-1, 2]))
        data = tf.data.Dataset.zip((data, kp)).map(lambda d, k: {**d, 'keypoints': k})
        data = data.map(lambda d: add_dummy_valid_mask(d, mode=split_name, **config), num_parallel_calls=None)

    # Keep only the first elements for validation
    if split_name == 'validation':
        data = data.take(config['validation_size'])

    # Generate the warped pair
    if config['warped_pair']['enable']:
        assert has_keypoints
        data = data.map(lambda d: add_dummy_valid_mask(d, mode=split_name, adjust=True, **config),
                        num_parallel_calls=None)
        warped = data.map(lambda d: homographic_augmentation(d, add_homography=True, **config['warped_pair']),
                          num_parallel_calls=None)
        if is_training and config['augmentation']['photometric']['enable']:
            warped = warped.map(lambda d: photometric_augmentation(d, **config['augmentation']['photometric']),
                                num_parallel_calls=None)
        warped = warped.map(lambda w: add_keypoint_map(w, sp=True), num_parallel_calls=None)
        # Merge with the original data
        data = tf.data.Dataset.zip((data, warped))
        data = data.map(lambda d, w: {**d, 'warped': w})

    # Data augmentation
    if has_keypoints and is_training:
        if config['augmentation']['photometric']['enable']:
            data = data.map(lambda d: photometric_augmentation(d, **config['augmentation']['photometric']),
                            num_parallel_calls=None)
        if config['augmentation']['homographic']['enable']:
            assert not config['warped_pair']['enable']  # doesn't support hom. aug.
            data = data.map(lambda d: homographic_augmentation(d, **config['augmentation']['homographic']),
                            num_parallel_calls=None)

    # Generate the keypoint map
    if has_keypoints:
        if mode == 'sp':
            data = data.map(lambda d: add_keypoint_map(d, add_homography=True, sp=True, norm=norm),
                            num_parallel_calls=None)
        else:
            data = data.map(lambda d: add_keypoint_map(d, norm=norm), num_parallel_calls=None)

    return data


def data_gen_coco(files, split_name, batch_size, norm, mode='mp', **config):
    """
    Data Generator for ML model

    Arguments:
        files: filenames for training, validation and export sets
        split_name: training, validation or export
        batch_size: size of batch for given split_name
        norm: normalize images (True or False)
        mode: Magicpoint or Superpoint
    """
    data = get_coco_data(files, split_name, mode, norm, **config)
    if split_name == 'export':
        data = data.batch(batch_size)
    else:
        data = data.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        data = data.make_one_shot_iterator()

    return data


def get_hpatches_data(files, norm, **config):
    """
    Read data from file

    Arguments:
        files: image and point filenames for export sets
        norm: normalize images (True or False)
    Return:
        data: tensorflow dataset for given split_name
    """
    n = tf.constant(255.)

    def _get_shape(image):
        return tf.shape(image)[:2]

    def _get_scale(shapes):
        return tf.reduce_max(tf.cast(tf.divide(tf.convert_to_tensor(config['preprocessing']['resize'],
                                                                    dtype=tf.float32),
                                               tf.cast(shapes, dtype=tf.float32)), dtype=tf.float32))

    def _read_image(path):
        return cv.imread(path.decode('utf-8'))

    def _preprocess(image, norm):
        tf.Tensor.set_shape(image, [None, None, 3])
        image = tf.image.rgb_to_grayscale(image)
        if config['preprocessing']['resize']:
            image = ratio_preserving_resize(image, **config['preprocessing'])
        if norm:
            image = tf.cast(image / n, dtype=tf.float32)
        else:
            image = tf.cast(image, dtype=tf.float32)
        return image

    def _preprocess_warped(zip_data, norm):
        image = zip_data['image']
        new_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32) * zip_data['scale']
        tf.Tensor.set_shape(image, [None, None, 3])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(tf.image.resize_images(image, tf.cast(new_size, dtype=tf.int32),
                                               method=tf.image.ResizeMethod.BILINEAR), dtype=tf.float32)
        if norm:
            image = tf.cast(image / n, dtype=tf.float32)
        else:
            image = tf.cast(image, dtype=tf.float32)
        return image

    def _warp_image(image):
        H = sample_homography(tf.shape(image)[:2])
        warped_im = tf.contrib.image.transform(image, H, interpolation="BILINEAR")
        return {'warped_im': warped_im, 'H': H}

    def _adapt_homography_to_preprocessing(zip_data):
        image = zip_data['image']
        H = tf.cast(zip_data['homography'], dtype=tf.float32)
        source_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        target_size = tf.cast(tf.convert_to_tensor(config['preprocessing']['resize']), dtype=tf.float32)
        s = tf.reduce_max(tf.cast(tf.divide(target_size, source_size), dtype=tf.float32))

        fit_height = tf.greater(tf.divide(target_size[0], source_size[0]), tf.divide(target_size[1], source_size[1]))
        padding_y = tf.cast(((source_size[0] * s - target_size[0]) / tf.constant(2.)), dtype=tf.int32)
        padding_x = tf.cast(((source_size[1] * s - target_size[1]) / tf.constant(2.)), dtype=tf.int32)
        tx = tf.cond(fit_height, lambda: padding_x, lambda: tf.constant(0))
        ty = tf.cond(fit_height, lambda: tf.constant(0), lambda: padding_y)
        translation = tf.stack([tf.constant(1), tf.constant(0), tx, tf.constant(0), tf.constant(1), ty,
                                tf.constant(0), tf.constant(0), tf.constant(1)])
        translation = tf.cast(tf.reshape(translation, [3, 3]), dtype=tf.float32)

        down_scale = tf.diag(tf.stack([1 / s, 1 / s, tf.constant(1.)]))
        up_scale = tf.diag(tf.stack([s, s, tf.constant(1.)]))
        H = tf.matmul(up_scale, tf.matmul(H, tf.matmul(down_scale, translation)))
        return H

    images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
    images = images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
    warped_images = tf.data.Dataset.from_tensor_slices(files['warped_image_paths'])
    warped_images = warped_images.map(lambda path: tf.py_func(_read_image, [path], tf.uint8))
    homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
    if config['preprocessing']['resize']:
        homographies = tf.data.Dataset.zip({'image': images, 'homography': homographies})
        homographies = homographies.map(_adapt_homography_to_preprocessing)
        img_shapes = images.map(_get_shape)
        scales = img_shapes.map(_get_scale)
        warped_images = tf.data.Dataset.zip({'image': warped_images, 'scale': scales})
        warped_images = warped_images.map(lambda d: _preprocess_warped(d, norm=norm))
    else:
        warped_images = warped_images.map(lambda d: _preprocess(d, norm=norm))

    images = images.map(lambda d: _preprocess(d, norm=norm))

    data = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                'homography': homographies})

    return data


def data_gen_hpatches(files, norm, **config):
    """
    Data Generator for ML model

    Arguments:
        files: filenames for export sets
        norm: normalize images (True or False)
    """
    data = get_hpatches_data(files, norm, **config)
    data = data.batch(1)

    return data

class ModelCheckpointOptimizer(tf.keras.callbacks.Callback):
    """
    Saves optimizer state
    """
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq='epoch', load_weights_on_restart=False):
        super(ModelCheckpointOptimizer, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.load_weights_on_restart = load_weights_on_restart
        self.epochs_since_last_save = 0
        self._samples_seen_since_last_saving = 0
        self.period = 1

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
            raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

        self._chief_worker_only = False

    def set_model(self, model):
        self.model = model
        if (not self.save_weights_only and not model._is_graph_network and model.__class__.__name__ != 'Sequential'):
            self.save_weights_only = True

    def on_train_begin(self, logs=None):
        filepath_to_load = None
        if (self.load_weights_on_restart and filepath_to_load is not None and os.path.exists(filepath_to_load)):
            try:
                self.model.load_weights(filepath_to_load)
            except (IOError, ValueError) as e:
                raise ValueError('Error loading file from {}. Reason: {}'.format(filepath_to_load, e))

    def on_train_end(self, logs=None):
        logs = logs or {}
        return

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if isinstance(self.save_freq, int):
            self._samples_seen_since_last_saving += logs.get('size', 1)
            if self._samples_seen_since_last_saving >= self.save_freq:
                self._save_optimizer(epoch=self._current_epoch, logs=logs)
                self._samples_seen_since_last_saving = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            self._save_optimizer(epoch=epoch, logs=logs)

    def _save_optimizer(self, epoch, logs):
        logs = logs or {}

        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            file_handle, filepath = self._get_file_handle_and_path(epoch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, ''skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            symbolic_weights = getattr(self.model.optimizer, 'weights')
                            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
                            with open(filepath, 'wb') as f:
                                pickle.dump(weight_values, f)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    symbolic_weights = getattr(self.model.optimizer, 'weights')
                    weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
                    with open(filepath, 'wb') as f:
                        pickle.dump(weight_values, f)

    def _get_file_handle_and_path(self, epoch, logs):
        return None, self.filepath.format(epoch=epoch + 1, **logs)
