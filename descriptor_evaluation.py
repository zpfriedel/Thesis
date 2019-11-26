import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tqdm import tqdm
from os import path as osp
from glob import glob


def get_paths(exper_name):
    """
    Return a list of paths to the outputs of the experiment.
    """
    return glob(osp.join('/home/ubuntu/data', 'exports/{}/*.npz'.format(exper_name)))


def keep_shared_points(keypoint_map, H, keep_k_points=1000):
    """
    Compute a list of keypoints from the map, filter the list of points by keeping
    only the points that once mapped by H are still inside the shape of the map
    and keep at most 'keep_k_points' keypoints in the image.
    """
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(keypoints, H, keypoint_map.shape)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)


def compute_homography(data, keep_k_points=1000, correctness_thresh=3, orb=False):
    """
    Compute the homography between 2 sets of detections and descriptors inside data.
    """
    shape = data['prob'][...].shape
    real_H = data['homography'][...]

    # Keeps only the points shared between the two views
    keypoints = keep_shared_points(data['prob'][...],
                                   real_H, keep_k_points)
    warped_keypoints = keep_shared_points(data['warped_prob'][...],
                                          np.linalg.inv(real_H), keep_k_points)
    desc = data['desc'][...][keypoints[:, 0], keypoints[:, 1]]
    warped_desc = data['warped_desc'][...][warped_keypoints[:, 0], warped_keypoints[:, 1]]

    # Match the keypoints with the warped_keypoints with nearest neighbor search
    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in matches])
    if len(matches_idx) == 0:
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': [],
                'inliers': [],
                'homography': None}
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv.RANSAC)
    if H is None:
        return {'correctness': 0.,
                'keypoints1': keypoints,
                'keypoints2': warped_keypoints,
                'matches': matches,
                'inliers': inliers,
                'homography': H}

    inliers = inliers.flatten()

    # Compute correctness
    corners = np.array([[0, 0, 1],
                        [shape[1] - 1, 0, 1],
                        [0, shape[0] - 1, 1],
                        [shape[1] - 1, shape[0] - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness = float(mean_dist <= correctness_thresh)

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,
            'inliers': inliers,
            'homography': H}


def homography_estimation(exper_name, keep_k_points=1000,
                          correctness_thresh=3, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the correctness score.
    """
    exper = '/home/ubuntu/data/exports' + '/' + exper_name + '/descriptors.h5'
    with h5py.File(exper, 'r') as f:
        paths = list(f.keys())
        correctness = []
        pbar = tqdm(total=None, leave=False)
        for path in paths:
            data = f[path]
            estimates = compute_homography(data, keep_k_points, correctness_thresh, orb)
            correctness.append(estimates['correctness'])
            pbar.update(1)
        pbar.close()

    return np.mean(correctness)


def get_homography_matches(exper_name, keep_k_points=1000,
                           correctness_thresh=3, num_images=1, orb=False):
    """
    Estimates the homography between two images given the predictions.
    The experiment must contain in its output the prediction on 2 images, an original
    image and a warped version of it, plus the homography linking the 2 images.
    Outputs the keypoints shared between the two views,
    a mask of inliers points in the first image, and a list of matches meaning that
    keypoints1[i] is matched with keypoints2[matches[i]]
    """
    exper = '/home/ubuntu/data/exports' + '/' + exper_name + '/descriptors.h5'
    with h5py.File(exper, 'r') as f:
        paths = list(f.keys())
        outputs = []
        for path in paths[:num_images]:
            data = f[path]
            output = compute_homography(data, keep_k_points, correctness_thresh, orb)
            output['image1'] = data['image'][...]
            output['image2'] = data['warped_image'][...]
            outputs.append(output)

    return outputs


def draw_matches(data):
    keypoints1 = [cv.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    img1 = np.concatenate([data['image1'], data['image1'], data['image1']], axis=2) * 255.
    img2 = np.concatenate([data['image2'], data['image2'], data['image2']], axis=2) * 255.
    return cv.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


experiments = []

num_images = 5
for e in experiments:
    orb = True if e[:3] == 'orb' else False
    outputs = get_homography_matches(e, keep_k_points=1000, correctness_thresh=3, num_images=num_images, orb=orb)
    for output in outputs:
        img = draw_matches(output) / 255.
        plot_imgs([img], titles=[e], dpi=200)

thresholds = [1, 3, 5]
for thresh in thresholds:
    print("Correctness threshold = " + str(thresh) + ":")
    for exp in experiments:
        orb = True if exp[:3] == 'orb' else False
        correctness = homography_estimation(exp, keep_k_points=1000, correctness_thresh=thresh, orb=orb)
        print('{}: {}'.format(exp, correctness))