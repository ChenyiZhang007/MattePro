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
python evaluation_super.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/UHRIM/UHRIM_ViTMatte_S_topk0.25_win_global_long_ft/model_0013307.pth \
    --dataset UHRIM \
    --save-path /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/evaluation_log.txt \
    --sad \
    --mse \
    --grad \
    --conn

python evaluation_super.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/max_number_token/high_res_pro_patch512_19000 \
    --dataset High_Res_Pro \
    --save-path /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/evaluation_log.txt \
    --sad \
    --mse \
    --grad \
    --conn


python evaluation_super.py \
    --pred-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha/ppm100_patch_decoder \
    --dataset PPM100 \
    --save-path /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/evaluation_log.txt \
    --sad \
    --mse \
    --grad \
    --conn

python evaluation_super.py \
    --pred-dir ./predAlpha/small_route_wo_eca/AIM500_40394 \
    --dataset AIM500 \
    --save-path ./evaluation_log.txt \
    --sad \
    --mse \
    --grad \
    --conn



python ./evaluation_super.py \
    --pred-dir ./predAlpha/d646_vits_025_wo_trimap \
    --dataset D646 \
    --save-path ./evaluation_log.txt \
    --global-flag\
    --sad \
    --mse \
    --grad \
    --conn

"""

test_dataset_path = {
    "DIM": {
        "label_dir" : "/opt/data/private/lyh/Datasets/Composition-1k-testset/alpha_copy",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/Composition-1k-testset/trimaps",
        "data_dir":"/opt/data/private/lyh/Datasets/Composition-1k-testset"
    },
    "D646": {
        "label_dir" : "/opt/data/private/lyh/Datasets/D646_Test/alpha_copy",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/D646_Test/trimaps",
        "data_dir": "/opt/data/private/lyh/Datasets/D646_Test"
    },
    "AIM500": {
        "label_dir" : "/opt/data/private/lyh/Datasets/AIM-500/mask",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/AIM-500/trimap",
        "data_dir": "/opt/data/private/lyh/Datasets/AIM-500"
    },
    "High_Res_Pro":{
        "label_dir" : "/opt/data/private/lyh/Datasets/high_res_pro/alpha_copy",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/high_res_pro/trimaps",
        "data_dir": "/opt/data/private/lyh/Datasets/high_res_pro"
    },
    "PPM100":{
        "label_dir" : "/opt/data/private/lyh/Datasets/PPM-100/matte",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/PPM-100/trimaps",
        "data_dir": "/opt/data/private/lyh/Datasets/PPM-100"
    },
    "UHRIM": {
        "label_dir" : "/opt/data/private/lyh/Datasets/UHRIM/Test/alpha_copy",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/UHRIM/Test/trimaps",
        "data_dir": "/opt/data/private/lyh/Datasets/UHRIM/Test"
    },

}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='', help="pred alpha dir")
    # parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    # parser.add_argument('--detailmap-dir', type=str, default='', help="trimap dir")
    parser.add_argument('--save-path', type=str, default='./evaluation_log.txt')
    parser.add_argument('--dataset', type=str, default='DIM') # D646, High_res_pro, AIM500
    parser.add_argument('--global-flag', action='store_true')
    parser.add_argument('--mse', action='store_true')
    parser.add_argument('--sad', action='store_true')
    parser.add_argument('--grad', action='store_true')
    parser.add_argument('--conn', action='store_true')
    args = parser.parse_args()

    label_dir = test_dataset_path[args.dataset]['label_dir']
    detailmap_dir = test_dataset_path[args.dataset]['trimap_dir']

    mse_loss_global = []
    sad_loss_global = []
    grad_loss_global = []
    conn_loss_global = []
    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []
    
    for img in tqdm(os.listdir(label_dir)):
        print(img)
        if args.dataset == "AIM500":
            pred = cv2.imread(os.path.join(args.pred_dir, img.replace('.png', '.jpg')), 0).astype(np.float32) # AIM500
        else:
            pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        
        alpha = cv2.imread(os.path.join(label_dir, img), 0).astype(np.float32)
        detailmap = cv2.imread(os.path.join(detailmap_dir, img), 0).astype(np.float32)

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
            print('Global Region: MSE:', mse_loss_global_, ' SAD:', sad_loss_global_)
            print('Global Region: GRAD:', grad_loss_global_, ' CONN:', conn_loss_global_)
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

    print(args.pred_dir)
    print('Average:')
    if args.global_flag:
        print('Global Region: MSE:', np.array(mse_loss_global).mean(), ' SAD:', np.array(sad_loss_global).mean())
        print('Global Region: GRAD:', np.array(grad_loss_global).mean(), ' CONN:', np.array(conn_loss_global).mean())
    else:
        print('Detail Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
        print('Detail Region: GRAD:', np.array(grad_loss_unknown).mean(), ' CONN:', np.array(conn_loss_unknown).mean())

    with open(args.save_path, 'a+') as file:
        print("==========================================", file=file)
        print(f"pred_dir: {args.pred_dir}", file=file)
        print(f"label_dir: {label_dir}", file=file)
        print(f"detailmap_dir: {detailmap_dir}", file=file)
        print(f"global_flag: {args.global_flag}", file=file)
        if args.global_flag:
            print('Global Region: MSE:', np.array(mse_loss_global).mean(), ' SAD:', np.array(sad_loss_global).mean(), file=file)
            print('Global Region: GRAD:', np.array(grad_loss_global).mean(), ' CONN:', np.array(conn_loss_global).mean(), file=file)
        else:
            print('Detail Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean(), file=file)
            print('Detail Region: GRAD:', np.array(grad_loss_unknown).mean(), ' CONN:', np.array(conn_loss_unknown).mean(), file = file)
        print("==========================================", file=file)
