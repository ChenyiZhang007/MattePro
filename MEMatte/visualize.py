'''
Inference for Composition-1k Dataset.

Run:
cd /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff


python inference.py \
    --config-dir ./configs/ViTMatte_B_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_b_0.25_model_0051719.pth \
    --inference-dir ./predAlpha/vit_b_com1k \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset

python inference.py \
    --config-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/checkpoints/vit_s_1024_model_0226274.pth \
    --inference-dir ./predAlpha/1024_vit_s_com1k \
    --data-dir /opt/data/private/lyh/Datasets/Composition-1k-testset

    
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/max_number_token/high_res_pro_patch512_22000 \
    --data-dir /opt/data/private/lyh/Datasets/high_res_pro \
    --max-number-token 22000 \
    --patch-decoder

    
python visualize.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir /opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/sparsity_visual/AIM500 \
    --data-dir /opt/data/private/lyh/Datasets/AIM-500 \
    --max-number-token 999999 \
    --patch-decoder



CUDA_VISIBLE_DEVICES=2 python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_1024_model_0226274.pth \
    --inference-dir ./predAlpha/1024_vit_s_high_res_pro \
    --data-dir /opt/data/private/lyh/Datasets/high_res_pro \
    --patch-decoder
    


CUDA_VISIBLE_DEVICES=1 python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/d646_vits_025_wo_trimap \
    --data-dir /opt/data/private/lyh/Datasets/D646_Test

CUDA_VISIBLE_DEVICES=3 python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_1024_model_0226274.pth \
    --inference-dir ./predAlpha/1024_vit_s_d646 \
    --data-dir /opt/data/private/lyh/Datasets/D646_Test

python inference.py \
    --config-dir ./configs/ViTMatte_B_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_b_0.25_model_0051719.pth \
    --inference-dir ./predAlpha/vit_b_d646 \
    --data-dir /opt/data/private/lyh/Datasets/D646_Test

python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/ppm100_patch_decoder \
    --data-dir /opt/data/private/lyh/Datasets/PPM-100 \
    --patch-decoder

python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/1k_8k \
    --data-dir /opt/data/private/lyh/Datasets/1k_to_8k

python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/t460_patch_decoder \
    --data-dir /opt/data/private/lyh/Datasets/Transparent-460/Test \
    --patch-decoder

python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/debug \
    --data-dir /opt/data/private/lyh/Datasets/Transparent-460/OOM \
    --patch-decoder

python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_0.25_model_0040394.pth \
    --inference-dir ./predAlpha/UHR_patch_decoder_w_tri \
    --data-dir /opt/data/private/lyh/Datasets/UHR_test \
    --patch-decoder


python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_1024_model_0226274.pth \
    --inference-dir ./predAlpha/UHR_1024 \
    --data-dir /opt/data/private/lyh/Datasets/UHR_test \
    --patch-decoder


    
python inference.py \
    --config-dir ./configs/ViTMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/vit_s_1024_model_0226274.pth \
    --inference-dir ./predAlpha/1024_AIM500_vit_s_025 \
    --data-dir /opt/data/private/lyh/Datasets/AIM-500

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
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings
warnings.filterwarnings('ignore')

#Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets

class Composition_1k(Dataset):
    def __init__(self, data_dir, finished_list = None):
        self.data_dir = data_dir
        # self.file_names = sorted(os.listdir(opj(self.data_dir, 'merged')), reverse=True)
        self.file_names = sorted(os.listdir(opj(self.data_dir, 'original')))
        # self.file_names = list(set(self.file_names).difference(set(finished_list))) # difference

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # phas = Image.open(opj(self.data_dir, 'alpha_copy', self.file_names[idx]))
        # tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx]))
        # print(self.file_names[idx])
        # print(self.data_dir)
        # tris = Image.open(opj(self.data_dir, 'trimaps', self.file_names[idx].replace('jpeg','png')))
        # imgs = Image.open(opj(self.data_dir, 'merged', self.file_names[idx]))
        
        tris = Image.open(opj(self.data_dir, 'trimap', self.file_names[idx].replace('jpg','png')))
        imgs = Image.open(opj(self.data_dir, 'original', self.file_names[idx]))
        sample = {}

        sample['trimap'] = F.to_tensor(tris)[0:1, :, :]
        sample['image'] = F.to_tensor(imgs)
        sample['image_name'] = self.file_names[idx]
        # print(sample['image_name'])
        return sample


#model and output
def matting_inference(
    config_dir='',
    checkpoint_dir='',
    inference_dir='',
    data_dir='',
    rank=None,
    patch_decoder = False,
    max_number_token = 18500,
):
    # finished_list = os.listdir(inference_dir)

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
        data_dir = data_dir,
        # finished_list = finished_list
    ),
    shuffle = False,
    batch_size = 1,
    # collate_fn = collate_fn,
    )
    
    #inferencing
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)
            # print(data['image_name'][0])
            output, out_pred_prob, out_hard_keep_decision = model(data, patch_decoder)
            output = output['phas'].flatten(0, 2)
            trimap = data['trimap'].squeeze(0).squeeze(0)
            # img = cv2.imread(f"/opt/data/private/lyh/Datasets/AIM-500/original/{data['image_name'][0]}")
            # gray = torch.tensor([211 / 255.0, 211 / 255.0, 211 / 255.0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(img.shape)
            # for i, select_token_map in enumerate(out_pred_prob):
            #     select_token_map = select_token_map.unsqueeze(0)   
            #     select_token_map =1. - torch.nn.functional.interpolate(select_token_map, size=(trimap.shape[0], trimap.shape[1]), mode='nearest')
            #     select_token_map[select_token_map >= 0.5] = 1
            #     select_token_map[select_token_map < 0.5] = 0
            #     B, _, H, W = select_token_map.shape
            #     select_token_map = select_token_map.expand(B, 3, H, W).squeeze(0).permute(1,2,0).cpu().numpy()
            #     result = select_token_map * 211.0 + (1-select_token_map) * img
            #     # result = torch.where(select_token_map.bool(), gray.to(output.device), img.to(output.device))
            #     img_name = data['image_name'][0].replace(".jpg", f"_{i}.jpg")
            #     # print(f"{img_name} : {select_token_map.sum()}")
            #     cv2.imwrite(os.path.join(inference_dir,img_name), result)


            for i, (select_token_map, hard_keep_decision) in enumerate(zip(out_pred_prob, out_hard_keep_decision)):
                img = cv2.imread(f"/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/reduce_token_AIM500/subset_AIM500/original/{data['image_name'][0]}")
                img = img / 255.0
                # hard_keep_decision = torch.zeros_like(select_token_map[..., :1])
                # hard_keep_decision[select_token_map[..., 0] > select_token_map[..., 1]] = 1.
                hard_keep_decision = hard_keep_decision.squeeze(-1).unsqueeze(0)
                select_token_map = select_token_map[..., 0]
                select_token_map = select_token_map.unsqueeze(0)
                print(data['image_name'][0])
                print(hard_keep_decision.sum())
                if hard_keep_decision.sum() == 0:
                    continue
                max_score = select_token_map[hard_keep_decision.bool()].max()
                min_score = select_token_map[hard_keep_decision.bool()].min()
                select_token_map = torch.nn.functional.interpolate(select_token_map, size=(trimap.shape[0], trimap.shape[1]), mode='bilinear').squeeze(0).squeeze(0)
                hard_keep_decision = torch.nn.functional.interpolate(hard_keep_decision, size=(trimap.shape[0], trimap.shape[1]), mode='nearest').squeeze(0).squeeze(0).unsqueeze(-1).cpu().numpy()
                normalized_scores = (select_token_map - min_score) / (max_score - min_score)
                cmap = cm.get_cmap('rainbow')# viridis
                colored_image = cmap(normalized_scores.cpu())
                colored_image = colored_image[:, :, :3] *  hard_keep_decision
                alpha = 0.4
                colored_image = (1-alpha) * img + alpha * colored_image[:, :, :3]
                img_name = data['image_name'][0].replace(".jpg", f"_{i}.jpg")
                plt.imsave(f'/opt/data/private/lyh/Efficient_Matting/ViT_based/ViTMatte_topk_eff/visiualization/reduce_token_AIM500/visual_500token/{img_name}', colored_image)



            output[trimap == 0] = 0
            output[trimap == 1] = 1
            output = F.to_pil_image(output)
            # output.save(opj(inference_dir, data['image_name'][0]))
            torch.cuda.empty_cache()

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
