"""
Reimplement evaluation.mat provided by Adobe in python
Output of `compute_gradient_loss` is sightly different from the MATLAB version provided by Adobe (less than 0.1%)
Output of `compute_connectivity_error` is smaller than the MATLAB version (~5%, maybe MATLAB has a different algorithm)
So do not report results calculated by these functions in your paper.
Evaluate your inference with the MATLAB file `DIM_evaluation_code/evaluate.m`.

by Yaoyi Li
"""

import scipy.ndimage
import numpy as np
from skimage.measure import label
import scipy.ndimage.morphology


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, trimap):

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC

def compute_connectivity_error(pd, gt, trimap, step=0.1):
    pd = pd/255.
    gt = gt/255.
    from scipy.ndimage import morphology
    from skimage.measure import label, regionprops
    h, w = pd.shape
    thresh_steps = np.arange(0, 1 + step, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords
        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]
        dist_maps = scipy.ndimage.distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pd - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    # loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
    loss_unknown = np.sum(np.abs(phi_pd - phi_gt) * (trimap == 128)) / 1000
    return loss_unknown


def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000

def compute_mad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss

import os
import cv2
import sys
import numpy as np
sys.path.insert(0, './utils')
import argparse
from tqdm import tqdm

"""
python evaluate.py \
    --pred-dir /home/yihanhu/workdir/Matting_Baselines/Matteformer/result/AIM500/pred_alpha \
    --label-dir /home/yihanhu/Datasets/Matting/AIM-500/alpha_copy \
    --detailmap-dir /home/yihanhu/Datasets/Matting/AIM-500/trimaps

python evaluate.py \
    --pred-dir /opt/data/private/lyh/ViTMatte/predAlpha/ViTS_Com_wo_grid \
    --label-dir /opt/data/private/lyh/Datasets/Composition-1k-testset/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/Composition-1k-testset/trimaps

    
python evaluate.py \
    --pred-dir /opt/data/private/lyh/matteformer/predDIM/Com1k_best_model/pred_alpha \
    --label-dir /opt/data/private/lyh/Datasets/Composition-1k-testset/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/Composition-1k-testset/trimaps
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='', help="pred alpha dir")
    parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    parser.add_argument('--detailmap-dir', type=str, default='', help="trimap dir")

    args = parser.parse_args()

    mse_loss = []
    sad_loss = []
    mad_loss = []
    grad_loss = []
    conn_loss = []
    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []
    
    for img in tqdm(os.listdir(args.label_dir)):
        # print(img)
        # pred = cv2.imread(os.path.join(args.pred_dir, img.replace('.png', '.jpg')), 0).astype(np.float32)
        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        alpha = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        detailmap = cv2.imread(os.path.join(args.detailmap_dir, img), 0).astype(np.float32)

        # detailmap[detailmap > 0] = 128

        mse_loss_unknown_ = compute_mse_loss(pred, alpha, detailmap)
        sad_loss_unknown_ = compute_sad_loss(pred, alpha, detailmap)[0]
        grad_loss_unknown_ = compute_gradient_loss(pred, alpha, detailmap)
        conn_loss_unknown_ = compute_connectivity_error(pred, alpha, detailmap)

        # detailmap[...] = 128

        # mse_loss_ = compute_mse_loss(pred, label, detailmap)
        # sad_loss_ = compute_sad_loss(pred, label, detailmap)[0]
        # mad_loss_ = compute_mad_loss(pred, label, detailmap)
        # grad_loss_ = compute_gradient_loss(pred, label, detailmap)
        # conn_loss_ = compute_connectivity_error(pred, label, detailmap)

        # print('Whole Image: MSE:', mse_loss_, ' SAD:', sad_loss_, ' MAD:', mad_loss_, 'Grad:', grad_loss_, ' Conn:', conn_loss_)
        # print('Detail Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)
        # print('Detail Region: Grad:', grad_loss_unknown_, ' Conn:', conn_loss_unknown_)

        mse_loss_unknown.append(mse_loss_unknown_)
        sad_loss_unknown.append(sad_loss_unknown_)
        grad_loss_unknown.append(grad_loss_unknown_)
        conn_loss_unknown.append(conn_loss_unknown_)


        # mse_loss.append(mse_loss_)
        # sad_loss.append(sad_loss_)
        # mad_loss.append(mad_loss_)
        # grad_loss.append(grad_loss_)
        # conn_loss.append(conn_loss_)

    print('Average:')
    print('Detail Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    print('Detail Region: GRAD:', np.array(grad_loss_unknown).mean(), ' CONN:', np.array(conn_loss_unknown).mean())
