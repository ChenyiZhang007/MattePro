import argparse
import torch
import random
import os
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from inference import matting_inference
import cv2
import numpy as np
from detectron2.engine import default_argument_parser

checkpoints_path = "/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/output_of_train"
config_dir = "/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/configs"

log_path = "/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/evaluation"
alpha_save_path = "/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/predAlpha"

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
    "UHRIM": {
        "label_dir" : "/opt/data/private/lyh/Datasets/UHRIM/Test/alpha_copy",
        "trimap_dir" : "/opt/data/private/lyh/Datasets/UHRIM/Test/trimaps",
        "data_dir": "/opt/data/private/lyh/Datasets/UHRIM/Test"
    },
}

def compute_mse_loss(gt, pred, trimap=None, scale=1000, unkonw_region=True):
    error_map = np.abs((gt - pred) / 255.0)
    if unkonw_region:
        error = np.sum(error_map * (trimap == 128))
    else:
        error = np.sum(error_map)
    return error / scale


def compute_sad_loss(gt, pred, trimap=None, scale=1e-3, unkonw_region=True):
    error_map = (gt - pred) / 255.0
    if unkonw_region:
        error = np.sum((error_map ** 2) * (trimap == 128))
        error = error / (np.sum(trimap == 128) + 1e-8)
    else:
        error = np.sum(error_map ** 2) / (error_map.shape[0] * error_map.shape[1])
    return error / scale

def evaluate(pred_dir, label_dir, trimap_dir, unkonw_region=True):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []

    for i, img in enumerate(os.listdir(label_dir)):

        if not((os.path.isfile(os.path.join(pred_dir, img)) and
                os.path.isfile(os.path.join(label_dir, img)) and
                os.path.isfile(os.path.join(trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(trimap_dir, img), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap, unkonw_region=unkonw_region)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap, unkonw_region=unkonw_region)
        print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area

        print('[{}/{}] "{}" processed'.format(i, len(os.listdir(label_dir)), img))

    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    print('* if you want to report scores in your paper, please use the official matlab codes for evaluation.')
    print(pred_dir)
    return np.array(mse_loss_unknown).mean(), np.array(sad_loss_unknown).mean()


def log_result(log_path, version, item, sad, mse):
    print("log file: ", item)
    results = []
    if os.path.exists(os.path.join(log_path, version+".txt")):
        with open(os.path.join(log_path, version+".txt"), "r") as file:
            for line in file.readlines():
                results.append(line.strip())

    with open(os.path.join(log_path, version+".txt"), "w") as file:
        new_line = item + " " + str(sad) + " " + str(mse)
        results.append(new_line)
        results = sorted(results)
        file.writelines("\n".join(results))

class Evaluation:
    def __init__(self,rank,world_size,config_file, checkpoint_name, dataset_name, port, max_number_token, patch_decoder):
        self.config_file = config_file
        self.checkpoint_name = checkpoint_name
        self.dataset_name = dataset_name
        self.max_number_token = max_number_token
        self.patch_decoder = patch_decoder
        self.init_distributed(rank, world_size, port)
        self.init_datasets()
        self.evaluation()
        self.cleanup()


    def init_distributed(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def init_datasets(self):
        self.label_dir = test_dataset_path[self.dataset_name]['label_dir']
        self.trimap_dir = test_dataset_path[self.dataset_name]['trimap_dir']
        self.data_dir = test_dataset_path[self.dataset_name]['data_dir']

        self.log('Initializing datasets')
        self.unfinished = self.get_unfinished()
        if self.rank == 0:
            print("unfinished:",len(self.unfinished))
            print(self.unfinished)
        length = len(self.unfinished) // self.world_size
        self.dataset = [self.unfinished[i:i+length] for i in range(0, len(self.unfinished), length)]
        self.dataloader = self.dataset[self.rank]
        
            
    def evaluation(self):
        unkonw_region = False if self.dataset_name in['D646','SIMD'] else True
        for item in tqdm(self.dataloader):
            version = item.split("/")[-2]
            pt_name = item.split("/")[-1] + ".pth"
            pt_path = os.path.join(checkpoints_path, version, pt_name)
            config_path = os.path.join(config_dir, version + ".py")
            self.log('inference:' + item)
            self.log(pt_path)
            self.log(config_path)
            pred_alpha_path = os.path.join(alpha_save_path, self.dataset_name, version, pt_name)
            matting_inference(config_path, pt_path, pred_alpha_path, self.data_dir, self.rank, self.patch_decoder, self.max_number_token)
            self.log('evaluate:' + item)
            mse, sad = evaluate(pred_alpha_path, self.label_dir, self.trimap_dir, unkonw_region=unkonw_region)
            log_result(log_path, version, item, sad, mse)
    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')

    def get_unfinished(self):
        completed = []
        log_files = os.listdir(log_path)
        for log_file in log_files:
            version = log_file.split(".")[0]
            if version == "Archived":
                continue
            with open(os.path.join(log_path, log_file), 'r') as file:
                for line in file.readlines():
                    item = line.strip().split(' ')[0]
                    if item not in completed:
                        completed.append(item)
        
        versions = os.listdir(checkpoints_path)
        unifinished = []
        for version in versions:
            if self.config_file != "" and version != self.config_file:
                continue
            pt_names = os.listdir(os.path.join(checkpoints_path, version))
            for pt_name in pt_names:
                if "pth" in pt_name:
                    if self.checkpoint_name != "" and pt_name != self.checkpoint_name:
                        continue
                    item = self.dataset_name + "/" + version + "/" + pt_name.split(".p")[0]
                    if item not in completed:
                        unifinished.append(item)
                    
        return sorted(unifinished)

    def cleanup(self):
        dist.destroy_process_group()
    
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type = str, default="ViTMatte_S_100ep")
    parser.add_argument('--checkpoint-name', type = str, default="") # model_0026929.pth
    parser.add_argument('--dataset-name', type = str, default="DIM") # DIM / D646 / Trans640
    parser.add_argument('--port', type = str, default="14376")
    parser.add_argument('--max-number-token', type=int, required=True, default=18500)
    parser.add_argument('--patch-decoder', action='store_true')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        Evaluation,
        nprocs=world_size,
        args=(world_size, args.config_file, args.checkpoint_name, args.dataset_name, args.port, args.max_number_token, args.patch_decoder),
        join=True)
"""
conda activate ViTMatte
cd /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff

CUDA_VISIBLE_DEVICES=0,1 python auto_eval.py \
    --config-file UHRIM_ViTMatte_B_topk0.25_win_global_long_ft_mematte \
    --dataset-name UHRIM \
    --max-number-token 13000 \
    --patch-decoder
    


CUDA_VISIBLE_DEVICES=0,1 python auto_eval.py \
    --config-file Ab_ViTMatte_S_topk0.25_wo_eca \
    --dataset-name DIM \
    --max-number-token 9999999 \
    --patch-decoder
    
CUDA_VISIBLE_DEVICES=2,3 python auto_eval.py \
    --config-file Ab_ViTMatte_S_topk0.25_router_wo_local_global \
    --dataset-name DIM \
    --max-number-token 9999999 \
    --patch-decoder \
    --port 23445
    

    
CUDA_VISIBLE_DEVICES=2 python auto_eval.py \
    --config-file Ab_ViTMatte_S_topk0.25_wo_eff \
    --dataset-name D646 \
    --checkpoint-name model_0051719.pth \
    --port 13478
"""