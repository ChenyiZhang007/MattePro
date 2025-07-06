import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology

"""
python evaluation.py \
    --pred-dir /opt/data/private/lyh/ViTMatte_topk_eff/pred_Alpha/debug \
    --dataset-name Com

"""

def compute_mse_loss(pred, target, trimap, global_flag):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    # # if test on whole image (Disitinctions-646), please uncomment this line
    if global_flag is True:
        loss = loss = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])
    return loss


def compute_sad_loss(pred, target, trimap, global_flag):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    # # if test on whole image (Disitinctions-646), please uncomment this line
    if global_flag is True:
        loss = np.sum(error_map)

    return loss / 1000, np.sum(trimap == 128) / 1000



def evaluate(args):
    global_flag = False
    assert args.dataset_name in ["Com", "D646"]
    if args.dataset_name == "D646":
        global_flag = True
        label_dir = "/opt/data/private/lyh/Datasets/D646_Test/alpha_copy"
        trimap_dir = "/opt/data/private/lyh/Datasets/D646_Test/trimaps"
    elif args.dataset_name == "Com":
        label_dir = "/opt/data/private/lyh/Datasets/Composition-1k-testset/alpha_copy"
        trimap_dir = "/opt/data/private/lyh/Datasets/Composition-1k-testset/trimaps"
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []




    bad_case = []

    for i, img in tqdm(enumerate(os.listdir(label_dir))):

        if not((os.path.isfile(os.path.join(args.pred_dir, img)) and
                os.path.isfile(os.path.join(label_dir, img)) and
                os.path.isfile(os.path.join(trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(trimap_dir, img), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap, global_flag)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap, global_flag)[0]



        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area


    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), 'SAD:', np.array(sad_loss_unknown).mean())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True, help="output dir")
    # parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    # parser.add_argument('--trimap-dir', type=str, default='', help="trimap dir")
    parser.add_argument('--dataset-name', type = str, default='Com1k')
    args = parser.parse_args()

    evaluate(args)