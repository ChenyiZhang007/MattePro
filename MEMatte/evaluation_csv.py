"""
Reimplement evaluation.mat provided by Adobe in python
Output of `compute_gradient_loss` is sightly different from the MATLAB version provided by Adobe (less than 0.1%)
So do not report results calculated by these functions in your paper.
Evaluate your inference with the MATLAB file `DIM_evaluation_code/evaluate.m`.

by Yaoyi Li
"""

import scipy.ndimage
import numpy as np
from skimage.measure import label
import scipy.ndimage.morphology
import pandas as pd


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
High-resolutin

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_18500 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_17000 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_16000 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_15000 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_14000 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='', help="pred alpha dir")
    parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    parser.add_argument('--detailmap-dir', type=str, default='', help="trimap dir")
    parser.add_argument('--save-path', type=str, default='./evaluation_log.txt')
    parser.add_argument('--global-flag', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--sad', action='store_true')
    parser.add_argument('--grad', action='store_true')
    parser.add_argument('--conn', action='store_true')
    parser.add_argument('--name', type=str, required=True)

    args = parser.parse_args()

    mse_loss_global = []
    sad_loss_global = []
    grad_loss_global = []
    conn_loss_global = []
    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []
    image_names = []
    for img in tqdm(os.listdir(args.label_dir)):
        print(img)
        image_names.append(img)
        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        # pred = cv2.imread(os.path.join(args.pred_dir, img[:-4]+'.jpg'), 0).astype(np.float32)
        alpha = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        detailmap = cv2.imread(os.path.join(args.detailmap_dir, img), 0).astype(np.float32)

        if args.global_flag:
            # Gobal
            detailmap[...] = 128
            mse_loss_global_ = 0 if args.mse == False else compute_mse_loss(pred, alpha, detailmap)
            sad_loss_global_ = 0 if args.sad == False else  compute_sad_loss(pred, alpha, detailmap)[0]
            grad_loss_global_ = 0 if args.grad == False else  compute_gradient_loss(pred, alpha, detailmap)
            conn_loss_global_ = 0 if args.conn == False else  compute_connectivity_error(pred, alpha, detailmap)
            mse_loss_global.append(mse_loss_global_)
            sad_loss_global.append(sad_loss_global_)
            grad_loss_global.append(grad_loss_global_)
            conn_loss_global.append(conn_loss_global_)
            # print('global')
        else:
            # Unknown
            mse_loss_unknown_ =  0 if args.mse == False else  compute_mse_loss(pred, alpha, detailmap)
            sad_loss_unknown_ =  0 if args.sad == False else  compute_sad_loss(pred, alpha, detailmap)[0]
            grad_loss_unknown_ =  0 if args.grad == False else  compute_gradient_loss(pred, alpha, detailmap)
            conn_loss_unknown_ =  0 if args.conn == False else  compute_connectivity_error(pred, alpha, detailmap)

            mse_loss_unknown.append(mse_loss_unknown_)
            sad_loss_unknown.append(sad_loss_unknown_)
            grad_loss_unknown.append(grad_loss_unknown_)
            conn_loss_unknown.append(conn_loss_unknown_)
            # print('local')
            print('Detail Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)
            print('Detail Region: Grad:', grad_loss_unknown_, ' Conn:', conn_loss_unknown_)

    df = pd.DataFrame({
        'Name': image_names,
        'SAD': sad_loss_unknown,
        'MSE': mse_loss_unknown,
        'Grad': grad_loss_unknown,
        'Conn': conn_loss_unknown
    })

    df.to_excel(f'./excel/high_res_vits_7000.xlsx', index=False)

"""
cd /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff

export Max_Number=11000
python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_7000 \
    --label-dir /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy \
    --detailmap-dir /opt/data/private/lyh/Datasets/high_res_pro/trimaps \
    --save-path ./evaluation_log.txt \
    --sad \
    --name test

python evaluation_csv.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/small_route/AIM500_37701 \
    --label-dir /opt/data/private/lyh/Datasets/AIM-500/mask \
    --detailmap-dir /opt/data/private/lyh/Datasets/AIM-500/trimap \
    --save-path ./evaluation_log.txt \
    --sad \
    --name test


export image_name=14996438642_58c976f957_o.jpg
code /opt/data/private/lyh/Datasets/high_res_pro/alpha_copy/$image_name
code /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_19000/$image_name
code /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_11000/$image_name


export image_name=o_3a9a5e7c
code /opt/data/private/lyh/Datasets/AIM-500/mask/$image_name.png
code /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/small_route/AIM500_37701/$image_name.jpg
code /opt/data/private/lyh/matting_baselines/ViTMatte/predAlpha/AIM500_vit_s/$image_name.jpg


code /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/AIM500_patch512_100/$image_name.jpg


code /opt/data/private/lyh/matting_baselines/ViTMatte/predAlpha/AIM500_vit_s/$image_name.jpg





o_cc63fd4a
code /opt/data/private/lyh/Datasets/AIM-500/trimap/$image_name.png
code /opt/data/private/lyh/Datasets/AIM-500/original/$image_name.jpg




export image_name=
cp /opt/data/private/lyh/Datasets/AIM-500/mask/$image_name.png /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/gt
cp /opt/data/private/lyh/Datasets/AIM-500/original/$image_name.jpg /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/images
cp /opt/data/private/lyh/Datasets/AIM-500/trimap/$image_name.png /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/trimaps

cp /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/AIM500_vit_s_025/$image_name.jpg /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/Ours
cp /opt/data/private/lyh/matting_baselines/matteformer/predAlpha/AIM500/pred_alpha/$image_name.jpg /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/matteformer
cp /opt/data/private/lyh/matting_baselines/FBA_Matting/examples/AIM500/${image_name}_alpha.png /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/FBA
cp /opt/data/private/lyh/matting_baselines/ViTMatte/predAlpha/AIM500_vit_s/$image_name.jpg /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/AIM500/ViTMatte

"""