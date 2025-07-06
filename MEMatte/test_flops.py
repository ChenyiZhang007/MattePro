'''
Inference for Composition-1k Dataset.

Run:
cd /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff

python test_flops.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/high_res_pro

python test_flops.py \
    --config-dir ./configs/ViTMatte_B_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_b_0.25_model_0051719.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/high_res_pro

CUDA_VISIBLE_DEVICES=1 python test_flops.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/test_flops \
    --data-dir /opt/data/private/lyh/Datasets/high_res_matting

CUDA_VISIBLE_DEVICES=1 python test_flops.py \
    --config-dir /opt/data/private/lyh/ViTMatte_topk_eff/configs/ViTMatte_S_100ep_topk0.5_wo_rel.py \
    --checkpoint-dir /opt/data/private/lyh/ViTMatte_topk_eff/output_of_train/ViTMatte_S_100ep_topk0.5_wo_rel/model_final.pth \
    --inference-dir ./predAlpha/debug \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset
'''
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from thop import profile

import warnings
warnings.filterwarnings('ignore')

#Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets

class Composition_1k(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(opj(self.data_dir, 'alpha_copy', self.file_names[idx]))
        tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx]))
        imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx]))
        sample = {}

        sample['trimap'] = F.to_tensor(tris)[0:1, :, :]
        sample['image'] = F.to_tensor(imgs)
        sample['image_name'] = self.file_names[idx]

        return sample

#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
    data_dir='',
):
    #initializing model
    cfg = LazyConfig.load(config_dir)
    cfg.model.teacher_backbone = None
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    print(parameter_count_table(model))
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_dir)

    #initializing dataset
    composition_1k_dataloader = DataLoader(
    dataset = Composition_1k(
        data_dir = data_dir
    ),
    shuffle = False,
    batch_size = 1,
    # collate_fn = collate_fn,
    )
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)
    total_flops = 0
    flops_list = []
    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)
            print(data['image_name'][0])
            # if "oleksandr-sushko-ORt4YUFc_9E-unsplash_6.png" == data['image_name'][0]:
            #     continue
            # output = model(data)['phas'].flatten(0, 2)
            # output = F.to_pil_image(output)
            # output.save(opj(inference_dir, data['image_name'][0]))

            # macs, params = profile(model, inputs=(data,))
            # flops = macs*2 / (1024*1024*1024)
            # flops_list.append(flops / (1024*1024*1024))
            # total_flops += flops
            # print(f"#GFLOPs:{flops} ({total_flops / len(flops_list)})")

            try:
                model.cuda()
                flops = FlopCountAnalysis(model, (data, True))
                flops_list.append(flops.total() / (10 ** 9))
                total_flops += flops.total() / (10 ** 9)
                print(f"#GFLOPs:{flops.total() / (10 ** 9)} ({total_flops / len(flops_list)})")
            except RuntimeError as e:
                print(str(e))
                model.cpu()
                for k in data.keys():
                    if k == 'image_name':
                        continue
                    else:
                        data[k].to(model.device)
                # flops = FlopCountAnalysis(model, (data, True))
                # flops_list.append(flops.total() / (10 ** 9))
                # total_flops += flops.total() / (10 ** 9)
                # print(f"#GFLOPs:{flops.total() / (10 ** 9)} ({total_flops / len(flops_list)})")

            torch.cuda.empty_cache()
    print(f"len: {len(flops_list)}")
    print(f"average flops: {total_flops/len(flops_list)}")


if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
        data_dir = args.data_dir
    )