import tensorflow as tf
from homographies import warp_points, flat2mat


def convBlock(conv_layer, bn_layer, input):

    x = conv_layer(input)
    x = bn_layer(x)

    return x


def encoder():
    params_conv = {'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
    params_pool = {'pool_size': 2, 'strides': 2, 'padding': 'same'}

    conv1a = tf.keras.layers.Conv2D(filters=64, name='conv1a', **params_conv)
    bn1 = tf.keras.layers.BatchNormalization(name='bn1')
    conv1b = tf.keras.layers.Conv2D(filters=64, name='conv1b', **params_conv)
    bn2 = tf.keras.layers.BatchNormalization(name='bn2')
    pool1 = tf.keras.layers.MaxPool2D(name='pool1', **params_pool)
    conv2a = tf.keras.layers.Conv2D(filters=64, name='conv2a', **params_conv)
    bn3 = tf.keras.layers.BatchNormalization(name='bn3')
    conv2b = tf.keras.layers.Conv2D(filters=64, name='conv2b', **params_conv)
    bn4 = tf.keras.layers.BatchNormalization(name='bn4')
    pool2 = tf.keras.layers.MaxPool2D(name='pool2', **params_pool)
    conv3a = tf.keras.layers.Conv2D(filters=128, name='conv3a', **params_conv)
    bn5 = tf.keras.layers.BatchNormalization(name='bn5')
    conv3b = tf.keras.layers.Conv2D(filters=128, name='conv3b', **params_conv)
    bn6 = tf.keras.layers.BatchNormalization(name='bn6')
    pool3 = tf.keras.layers.MaxPool2D(name='pool3', **params_pool)
    conv4a = tf.keras.layers.Conv2D(filters=128, name='conv4a', **params_conv)
    bn7 = tf.keras.layers.BatchNormalization(name='bn7')
    conv4b = tf.keras.layers.Conv2D(filters=128, name='conv4b', **params_conv)
    bn8 = tf.keras.layers.BatchNormalization(name='bn8')

    layers = {'conv1a': conv1a, 'bn1': bn1, 'conv1b': conv1b, 'bn2': bn2, 'pool1': pool1,
              'conv2a': conv2a, 'bn3': bn3, 'conv2b': conv2b, 'bn4': bn4, 'pool2': pool2,
              'conv3a': conv3a, 'bn5': bn5, 'conv3b': conv3b, 'bn6': bn6, 'pool3': pool3,
              'conv4a': conv4a, 'bn7': bn7, 'conv4b': conv4b, 'bn8': bn8}

    return layers


def encoder_build(e, img_in):
    x = convBlock(e['conv1a'], e['bn1'], img_in)
    x = convBlock(e['conv1b'], e['bn2'], x)
    x = e['pool1'](x)
    x = convBlock(e['conv2a'], e['bn3'], x)
    x = convBlock(e['conv2b'], e['bn4'], x)
    x = e['pool2'](x)
    x = convBlock(e['conv3a'], e['bn5'], x)
    x = convBlock(e['conv3b'], e['bn6'], x)
    x = e['pool3'](x)
    x = convBlock(e['conv4a'], e['bn7'], x)
    x = convBlock(e['conv4b'], e['bn8'], x)

    return x


def detector_head():
    params_conv = {'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
    params = {'kernel_size': 1, 'padding': 'same', 'activation': None}

    conv5 = tf.keras.layers.Conv2D(filters=256, name='conv5', **params_conv)
    bn9 = tf.keras.layers.BatchNormalization(name='bn9')
    det = tf.keras.layers.Conv2D(filters=65, name='det', **params)
    bn10 = tf.keras.layers.BatchNormalization(name='bn10')

    layers = {'conv5': conv5, 'bn9': bn9, 'det': det, 'bn10': bn10}

    return layers


def detector_build(d, in_layer, **config):
    x = convBlock(d['conv5'], d['bn9'], in_layer)
    x = convBlock(d['det'], d['bn10'], x)

    prob = tf.nn.softmax(x, axis=-1)
    # Strip the extra “no interest point” dustbin
    prob = prob[:, :, :, :-1]
    prob = tf.depth_to_space(prob, block_size=8)
    prob = tf.squeeze(prob, axis=-1)

    prob = tf.map_fn(lambda p: box_nms(p, config['nms'], min_prob=config['detection_threshold'],
                                       keep_top_k=config['top_k']), prob)
    """
    prob = tf.map_fn(lambda p: spatial_nms(p, config['nms']), prob)
    """
    pred = tf.cast(tf.greater_equal(prob, config['detection_threshold']), dtype=tf.int32)

    return {'logits': x, 'pred': pred}


def descriptor_head(filters):
    params_conv = {'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}
    params = {'kernel_size': 1, 'padding': 'same', 'activation': None}

    conv6 = tf.keras.layers.Conv2D(filters=256, name='conv6', **params_conv)
    bn11 = tf.keras.layers.BatchNormalization(name='bn11')
    desc = tf.keras.layers.Conv2D(filters=filters, name='desc', **params)
    bn12 = tf.keras.layers.BatchNormalization(name='bn12')

    layers = {'conv6': conv6, 'bn11': bn11, 'desc': desc, 'bn12': bn12}

    return layers


def descriptor_build(d, in_layer):
    x = convBlock(d['conv6'], d['bn11'], in_layer)
    x = convBlock(d['desc'], d['bn12'], x)
    """
    with tf.device('/cpu:0'):  # op not supported on GPU yet
        desc = tf.image.resize_bicubic(x, 8 * tf.shape(x)[1:3])
    desc = tf.nn.l2_normalize(desc, axis=-1)
    """
    return x


def detector_loss(y_true, y_pred):
    """
    Calculates sparse softmax cross entropy loss

    Arguments:
        y_true: a.k.a. keypoint_map: truth interest point labels (0 or 1) of shape [N, H, W, 1]
        y_pred: output of detector head of shape [N, H/8, W/8, 65]
        valid_mask: only considers points inside image dimensions in the case of homographies;
                    same shape as keypoint_map
    """
    keypoint_maps = y_true[..., 0]
    valid_masks = y_true[..., 1]

    labels = keypoint_maps[..., tf.newaxis]
    labels = tf.space_to_depth(labels, block_size=8)
    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    labels = tf.argmax(labels * tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)

    valid_masks = valid_masks[..., tf.newaxis]
    valid_masks = tf.space_to_depth(valid_masks, block_size=8)
    valid_masks = tf.reduce_prod(valid_masks, axis=3)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y_pred, weights=valid_masks)

    return loss


def total_loss(y_true, y_pred):
    """

    """
    def det_loss(logits, keypoint_maps, valid_masks):
        """
        Calculates sparse softmax cross entropy loss
        """
        keypoint_maps = keypoint_maps[..., tf.newaxis]
        keypoint_maps = tf.space_to_depth(keypoint_maps, block_size=8)
        shape = tf.concat([tf.shape(keypoint_maps)[:3], [1]], axis=0)
        keypoint_maps = tf.concat([2 * keypoint_maps, tf.ones(shape)], 3)
        keypoint_maps = tf.argmax(keypoint_maps * tf.random.uniform(tf.shape(keypoint_maps), 0, 0.1), axis=3)

        valid_masks = valid_masks[..., tf.newaxis]
        valid_masks = tf.space_to_depth(valid_masks, block_size=8)
        valid_masks = tf.reduce_prod(valid_masks, axis=3)

        loss_det = tf.losses.sparse_softmax_cross_entropy(labels=keypoint_maps, logits=logits, weights=valid_masks)

        return loss_det

    def desc_loss(descriptors, warped_descriptors, homographies, valid_mask=None):
        """

        """
        positive_margin = tf.constant(1.)
        negative_margin = tf.constant(0.2)
        lambda_d = tf.constant(800.)  # 800

        # Compute the position of the center pixel of every cell in the image
        (batch_size, Hc, Wc) = tf.unstack(tf.cast(tf.shape(descriptors)[:3], dtype=tf.int32))
        coord_cells = tf.stack(tf.meshgrid(tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
        coord_cells = coord_cells * 8 + 8 // 2  # (Hc, Wc, 2)
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image

        # Compute the position of the warped center pixels
        warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
        # warped_coord_cells is now a list of the warped coordinates of all the center
        # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

        # Compute the pairwise distances and filter the ones less than a threshold
        # The distance is just the pairwise norm of the difference of the two grids
        # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
        coord_cells = tf.cast(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]), dtype=tf.float32)
        warped_coord_cells = tf.reshape(warped_coord_cells, [batch_size, Hc, Wc, 1, 1, 2])
        cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
        s = tf.cast(tf.less_equal(cell_distances, 8 - 0.5), dtype=tf.float32)
        # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
        # homography is at a distance from (h', w') less than config['grid_size']
        # and 0 otherwise

        # Compute the pairwise dot product between descriptors: d^t * d'
        descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
        warped_descriptors = tf.reshape(warped_descriptors, [batch_size, 1, 1, Hc, Wc, -1])
        dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
        # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        # descriptor at position (h, w) in the original descriptors map and the
        # descriptor at position (h', w') in the warped image

        # Compute the loss
        positive_dist = tf.maximum(0., positive_margin - dot_product_desc)
        negative_dist = tf.maximum(0., dot_product_desc - negative_margin)
        loss = lambda_d * s * positive_dist + (1 - s) * negative_dist

        # Mask the pixels if bordering artifacts appear
        valid_mask = tf.cast(valid_mask[..., tf.newaxis], dtype=tf.float32)  # for GPU
        valid_mask = tf.space_to_depth(valid_mask, block_size=8)
        valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
        valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

        normalization = tf.reduce_sum(valid_mask) * tf.cast(Hc * Wc, dtype=tf.float32)
        loss_desc = tf.reduce_sum(valid_mask * loss) / normalization

        return loss_desc


    lambda_loss = tf.constant(1.)  # 1
    """
    logits = y_pred[..., :65]
    warped_logits = y_pred[..., 65:130]
    descriptors = y_pred[..., 130:386]
    warped_descriptors = y_pred[..., 386:]

    kp_maps = y_true[..., 0]
    valid_masks = y_true[..., 1]
    warped_kp_maps = y_true[..., 2]
    warped_valid_masks = y_true[..., 3]
    homographies = y_true[..., 4]
    homographies = homographies[..., :8, 0]
    """
    # Compute the loss for the detector head
    detector_loss = det_loss(y_pred[..., :65], y_true[..., 0], y_true[..., 1])
    warped_detector_loss = det_loss(y_pred[..., 65:130], y_true[..., 2], y_true[..., 3])

    # Compute the loss for the descriptor head
    descriptor_loss = desc_loss(y_pred[..., 130:386], y_pred[..., 386:], y_true[:, :8, 0, 4], y_true[..., 3])

    loss = detector_loss + warped_detector_loss + lambda_loss * descriptor_loss

    return loss


def precision_metric(nms_pred):

    def precision(y_true, y_pred):
        """
        Arguments:
            y_true: a.k.a. keypoint_map: truth interest point labels (0 or 1) of shape [N, H, W, 1]
            nms_pred: predictions of keypoint_map (0 or 1) after NMS of shape [N, H, W]
            y_pred: logits of shape [N, H/8, W/8, 65]
        Return:
            precision
        """
        labels = tf.cast(y_true[..., 0], dtype=tf.int32)
        mask = tf.cast(y_true[..., 1], dtype=tf.int32)
        prec = tf.reduce_sum(mask * (nms_pred * labels)) / tf.reduce_sum(mask * nms_pred)

        return prec

    return precision


def recall_metric(nms_pred):

    def recall(y_true, y_pred):
        """
        Arguments:
            y_true: a.k.a. keypoint_map: truth interest point labels (0 or 1) of shape [N, H, W, 1]
            nms_pred: predictions of keypoint_map (0 or 1) after NMS
            y_pred: logits of shape [N, H/8, W/8, 65]
        Return:
            recall
        """
        labels = tf.cast(y_true[..., 0], dtype=tf.int32)
        mask = tf.cast(y_true[..., 1], dtype=tf.int32)
        rec = tf.reduce_sum(mask * (nms_pred * labels)) / tf.reduce_sum(mask * labels)

        return rec

    return recall


def warped_precision_metric(nms_pred):

    def warped_precision(y_true, y_pred):
        """
        Arguments:
            y_true: a.k.a. keypoint_map: truth interest point labels (0 or 1) of shape [N, H, W, 1]
            nms_pred: predictions of keypoint_map (0 or 1) after NMS of shape [N, H, W]
            y_pred: logits of shape [N, H/8, W/8, 65]
        Return:
            precision
        """
        labels = tf.cast(y_true[..., 2], dtype=tf.int32)
        mask = tf.cast(y_true[..., 3], dtype=tf.int32)
        prec = tf.reduce_sum(mask * (nms_pred * labels)) / tf.reduce_sum(mask * nms_pred)

        return prec

    return warped_precision


def warped_recall_metric(nms_pred):

    def warped_recall(y_true, y_pred):
        """
        Arguments:
            y_true: a.k.a. keypoint_map: truth interest point labels (0 or 1) of shape [N, H, W, 1]
            nms_pred: predictions of keypoint_map (0 or 1) after NMS
            y_pred: logits of shape [N, H/8, W/8, 65]
        Return:
            recall
        """
        labels = tf.cast(y_true[..., 2], dtype=tf.int32)
        mask = tf.cast(y_true[..., 3], dtype=tf.int32)
        rec = tf.reduce_sum(mask * (nms_pred * labels)) / tf.reduce_sum(mask * labels)

        return rec

    return warped_recall


def threshold_precision_metric(nms_pred):

    def threshold_precision(y_true, y_pred):

        def modify_labels(label):
            kps = tf.cast(tf.where(label > 0), dtype=tf.float32)
            boxes = tf.cond(tf.shape(label)[0] < 240,
                            lambda: tf.concat([(kps-0.01) / [120, 160], (kps+2) / [120, 160]], axis=1),
                            lambda: tf.concat([(kps-0.01) / [240, 320], (kps+2) / [240, 320]], axis=1))
            mask = (boxes >= 0) & (boxes <= tf.cast(tf.concat([tf.shape(label) - 1, tf.shape(label) - 1], axis=-1),
                                                    dtype=tf.float32))
            boxes = tf.boolean_mask(boxes, tf.reduce_all(mask, -1))
            label = tf.image.draw_bounding_boxes(tf.cast(label[tf.newaxis, ..., tf.newaxis], dtype=tf.float32),
                                                 boxes[tf.newaxis, ...])
            label = tf.cast(tf.squeeze(label, axis=[0, -1]), dtype=tf.int32)
            return label

        labels = tf.cast(y_true[..., 0], dtype=tf.int32)
        mask = tf.cast(y_true[..., 1], dtype=tf.int32)
        thresh_labels = tf.map_fn(modify_labels, labels)
        prec = tf.reduce_sum(mask * (nms_pred * thresh_labels)) / tf.reduce_sum(mask * nms_pred)

        return prec

    return threshold_precision


def threshold_recall_metric(nms_pred):

    def threshold_recall(y_true, y_pred):

        def modify_labels(label):
            kps = tf.cast(tf.where(label > 0), dtype=tf.float32)
            boxes = tf.cond(tf.shape(label)[0] < 240,
                            lambda: tf.concat([(kps-0.01) / [120, 160], (kps+2) / [120, 160]], axis=1),
                            lambda: tf.concat([(kps-0.01) / [240, 320], (kps+2) / [240, 320]], axis=1))
            mask = (boxes >= 0) & (boxes <= tf.cast(tf.concat([tf.shape(label) - 1, tf.shape(label) - 1], axis=-1),
                                                    dtype=tf.float32))
            boxes = tf.boolean_mask(boxes, tf.reduce_all(mask, -1))
            label = tf.image.draw_bounding_boxes(tf.cast(label[tf.newaxis, ..., tf.newaxis], dtype=tf.float32),
                                                 boxes[tf.newaxis, ...])
            label = tf.cast(tf.squeeze(label, axis=[0, -1]), dtype=tf.int32)
            return label

        labels = tf.cast(y_true[..., 0], dtype=tf.int32)
        mask = tf.cast(y_true[..., 1], dtype=tf.int32)
        thresh_labels = tf.map_fn(modify_labels, labels)
        rec = tf.reduce_sum(mask * (nms_pred * thresh_labels)) / tf.reduce_sum(mask * labels)

        return rec

    return threshold_recall


def warped_threshold_precision_metric(nms_pred):

    def warped_threshold_precision(y_true, y_pred):

        def modify_labels(label):
            kps = tf.cast(tf.where(label > 0), dtype=tf.float32)
            boxes = tf.cond(tf.shape(label)[0] < 240,
                            lambda: tf.concat([(kps-0.01) / [120, 160], (kps+2) / [120, 160]], axis=1),
                            lambda: tf.concat([(kps-0.01) / [240, 320], (kps+2) / [240, 320]], axis=1))
            mask = (boxes >= 0) & (boxes <= tf.cast(tf.concat([tf.shape(label) - 1, tf.shape(label) - 1], axis=-1),
                                                    dtype=tf.float32))
            boxes = tf.boolean_mask(boxes, tf.reduce_all(mask, -1))
            label = tf.image.draw_bounding_boxes(tf.cast(label[tf.newaxis, ..., tf.newaxis], dtype=tf.float32),
                                                 boxes[tf.newaxis, ...])
            label = tf.cast(tf.squeeze(label, axis=[0, -1]), dtype=tf.int32)
            return label

        labels = tf.cast(y_true[..., 2], dtype=tf.int32)
        mask = tf.cast(y_true[..., 3], dtype=tf.int32)
        thresh_labels = tf.map_fn(modify_labels, labels)
        prec = tf.reduce_sum(mask * (nms_pred * thresh_labels)) / tf.reduce_sum(mask * nms_pred)

        return prec

    return warped_threshold_precision


def warped_threshold_recall_metric(nms_pred):

    def warped_threshold_recall(y_true, y_pred):

        def modify_labels(label):
            kps = tf.cast(tf.where(label > 0), dtype=tf.float32)
            boxes = tf.cond(tf.shape(label)[0] < 240,
                            lambda: tf.concat([(kps-0.01) / [120, 160], (kps+2) / [120, 160]], axis=1),
                            lambda: tf.concat([(kps-0.01) / [240, 320], (kps+2) / [240, 320]], axis=1))
            mask = (boxes >= 0) & (boxes <= tf.cast(tf.concat([tf.shape(label) - 1, tf.shape(label) - 1], axis=-1),
                                                    dtype=tf.float32))
            boxes = tf.boolean_mask(boxes, tf.reduce_all(mask, -1))
            label = tf.image.draw_bounding_boxes(tf.cast(label[tf.newaxis, ..., tf.newaxis], dtype=tf.float32),
                                                 boxes[tf.newaxis, ...])
            label = tf.cast(tf.squeeze(label, axis=[0, -1]), dtype=tf.int32)
            return label

        labels = tf.cast(y_true[..., 2], dtype=tf.int32)
        mask = tf.cast(y_true[..., 3], dtype=tf.int32)
        thresh_labels = tf.map_fn(modify_labels, labels)
        rec = tf.reduce_sum(mask * (nms_pred * thresh_labels)) / tf.reduce_sum(mask * labels)

        return rec

    return warped_threshold_recall


def warp_keypoints(keypoints, H):
    num_points = tf.shape(keypoints)[0]
    homogeneous_points = tf.concat([keypoints, tf.ones((num_points, 1))], axis=1)
    warped_points = tf.matmul(homogeneous_points, tf.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def keep_true_keypoints(points, H, shape):
    warped_points = warp_keypoints(tf.concat([points[:, 1][..., tf.newaxis], points[:, 0][..., tf.newaxis]],
                                             axis=-1), H)
    warped_points = tf.concat([warped_points[:, 1][..., tf.newaxis], warped_points[:, 0][..., tf.newaxis]], axis=-1)
    mask = (warped_points >= 0) & (warped_points <= tf.cast(shape - 1, dtype=tf.float32))
    points = tf.boolean_mask(points, tf.reduce_all(mask, -1))
    return points


def repeatability_metric(nms_pred, warped_nms_pred):

    def repeatability(y_true, y_pred):

        def calc_repeatability(label, pred, warped_pred):
            homography = tf.reshape(label[:8, 0, 4], [1, 8])
            H = flat2mat(homography)[0]
            pred_mask = tf.cast(label[..., 1], dtype=tf.int32)
            warped_pred_mask = tf.cast(label[..., 3], dtype=tf.int32)
            keypoints = tf.cast(tf.where(pred * pred_mask > 0), dtype=tf.float32)
            keypoints = tf.concat([keypoints[:, 1][..., tf.newaxis], keypoints[:, 0][..., tf.newaxis]], axis=-1)
            true_warped_keypoints = warp_keypoints(keypoints, H)
            true_warped_keypoints = tf.concat([true_warped_keypoints[:, 1][..., tf.newaxis],
                                               true_warped_keypoints[:, 0][..., tf.newaxis]], axis=-1)
            mask = (true_warped_keypoints >= 0) & (true_warped_keypoints <= tf.cast(tf.shape(pred) - 1,
                                                                                    dtype=tf.float32))
            true_warped_keypoints = tf.boolean_mask(true_warped_keypoints, tf.reduce_all(mask, -1))
            warped_keypoints = tf.cast(tf.where(warped_pred * warped_pred_mask > 0), dtype=tf.float32)
            warped_keypoints = keep_true_keypoints(warped_keypoints, tf.linalg.inv(H), tf.shape(pred))

            N1 = tf.shape(true_warped_keypoints)[0]
            N2 = tf.shape(warped_keypoints)[0]
            true_warped_keypoints = tf.expand_dims(true_warped_keypoints, axis=1)
            warped_keypoints = tf.expand_dims(warped_keypoints, axis=0)
            norm = tf.linalg.norm(true_warped_keypoints - warped_keypoints, axis=2)
            min1 = tf.reduce_min(norm, axis=1)
            count1 = tf.reduce_sum(tf.cast((min1 <= 3), dtype=tf.int32))
            min2 = tf.reduce_min(norm, axis=0)
            count2 = tf.reduce_sum(tf.cast((min2 <= 3), dtype=tf.int32))
            return tf.cast(((count1 + count2) / (N1 + N2)), dtype=tf.float32)

        rep = tf.map_fn(lambda x: calc_repeatability(x[0], x[1], x[2]), (y_true, nms_pred, warped_nms_pred),
                        dtype=tf.float32)

        return tf.reduce_mean(rep)

    return repeatability


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """
    Performs NMS on the heatmap (prob) by considering hypothetical
    bounding boxes centered at each pixel's location. Optionally only keeps the top k detections.

    Arguments:
        prob: the probability heatmap of shape [H, W]
        size: a scalar, the size of the bounding boxes
        iou: a scalar, the IoU overlap threshold
        min_prob: a threshold under which all probabilities are discarded before NMS
        keep_top_k: an integer, the number of top scores to keep
    Return:
        prob: probability heatmap after NMS

    """
    pts = tf.cast(tf.where(tf.greater_equal(prob, min_prob)), dtype=tf.float32)
    size = tf.constant(size/2.)
    boxes = tf.concat([pts-size, pts+size], axis=1)
    scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))

    with tf.device('/cpu:0'):
        indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)

    pts = tf.gather(pts, indices)
    scores = tf.gather(scores, indices)

    if keep_top_k:
        k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
        scores, indices = tf.nn.top_k(scores, k)
        pts = tf.gather(pts, indices)

    prob = tf.scatter_nd(tf.cast(pts, dtype=tf.int32), scores, tf.shape(prob))

    return prob


def spatial_nms(prob, size):
    """
    Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.
    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the pooling window.
    """
    prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
    pooled = tf.nn.max_pool(prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
    prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
    prob = tf.squeeze(prob, axis=[0, 3])

    return prob