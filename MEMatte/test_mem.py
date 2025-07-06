'''
Inference for Composition-1k Dataset.

Run:
cd /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff
conda activate ViTMatte

CUDA_VISIBLE_DEVICES=0 python test_mem.py \
    --config-dir ./configs/Ab_ViTMatte_S_topk0.25_eca_wo_dwc.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.25_wo_dwc_model_0037701.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder


CUDA_VISIBLE_DEVICES=1 python test_mem.py \
    --config-dir ./configs/Ab_ViTMatte_S_topk0.25_router_wo_local_global.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.25_router_wo_local_global_model_final.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder

CUDA_VISIBLE_DEVICES=2 python test_mem.py \
    --config-dir ./configs/Ab_ViTMatte_S_topk0.25_wo_eca.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.25_wo_eca_model_0037701.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder


CUDA_VISIBLE_DEVICES=1 python test_mem.py \
    --config-dir ./configs/Ab_ViTMatte_S_topk0.25_wo_distill.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.25_wo_distill_model_final.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder

CUDA_VISIBLE_DEVICES=2 python test_mem.py \
    --config-dir ./configs/Ab_ViTMatte_S_topk0.25_wo_eff.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.25_wo_eff_model_0032315.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder

CUDA_VISIBLE_DEVICES=0 python test_mem.py \
    --config-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/UHR_Image \
    --max-number-token 9999999 \
    --patch-decoder


CUDA_VISIBLE_DEVICES=1 python test_mem.py \
    --config-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/configs/Ab_ViTMatte_S_topk0.1.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/ablation/vit_s_0.1_model0029622.pth \
    --inference-dir ./predAlpha/test \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset \
    --max-number-token 9999999 \
    --patch-decoder
    
/opt/data/private/lyh/Datasets/D646_Test
/opt/data/private/lyh/Datasets/Composition-1k-testset
/opt/data/private/lyh/Datasets/1k_to_8k
/opt/data/private/lyh/Datasets/1_to_8
/opt/data/private/lyh/Datasets/high_res_pro

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
        self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')), reverse=True)
        # self.file_names = sorted(os.listdir(opj(self.data_dir, 'original')))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # phas = Image.open(opj(self.data_dir, 'alpha_copy', self.file_names[idx]))
        # tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx]))
        tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx].replace('jpeg','png')))
        # tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx].split(".")[0] + ".png"))
        imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx]))
        # imgs = Image.open(opj(self.data_dir, 'original', self.file_names[idx]))
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
    rank=None,
    patch_decoder = False,
    max_number_token = None,
):
    #initializing model
    cfg = LazyConfig.load(config_dir)
    cfg.model.teacher_backbone = None
    cfg.model.backbone.max_number_token = max_number_token
    model = instantiate(cfg.model)
    if patch_decoder:
        print("=======================")
        print("using patch decoder")
        print("=======================")
    
    model.to(cfg.train.device if rank is None else rank)
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
    mem_list = []
    for data in tqdm(composition_1k_dataloader):
        torch.cuda.reset_max_memory_allocated()
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)

            output, _, _ = model(data,patch_decoder=patch_decoder)
            output = output['phas'].flatten(0, 2)
            peak_memory_usage = torch.cuda.max_memory_allocated()
            print(f"Peak Mem {data['image_name'][0]}: {peak_memory_usage / (1024 ** 2)} MB")
            mem_list.append(peak_memory_usage / (1024**3))

            torch.cuda.empty_cache()
    print(f'Number:{len(mem_list)}')
    print(f'avg_mem: {sum(mem_list) / len(mem_list)}')

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--inference-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--max-number-token', type=int, required=True, default=18500)
    parser.add_argument('--patch-decoder', action='store_true')
    
    args = parser.parse_args()
    matting_inference(
        config_dir = args.config_dir,
        checkpoint_dir = args.checkpoint_dir,
        inference_dir = args.inference_dir,
        data_dir = args.data_dir,
        patch_decoder = args.patch_decoder,
        max_number_token = args.max_number_token
    )

