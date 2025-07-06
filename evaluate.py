import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from tqdm import tqdm
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import copy
import argparse
import numpy as np
import yaml


def get_bbox_from_alpha(alpha_matte):
    """
    Calculate bounding box from alpha matte.
    """
    indices = np.where(alpha_matte > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        raise ValueError("Alpha matte does not contain any non-zero pixels.")
    x1, y1, x2, y2 = min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])
    return [x1, y1, x2, y2]


def matting_inference(model, rawimg, trimap,patchify, device):
    """
    Perform matting inference.
    """

    data = {}
    data['trimap'] = torchvision.transforms.functional.to_tensor(trimap)[0:1, :, :].unsqueeze(0)
    data['image'] = torchvision.transforms.functional.to_tensor(rawimg).unsqueeze(0)
    for k in data.keys():
        data[k].to(model.device)
    patch_decoder = patchify
    output = model(data, patch_decoder)[0]['phas'].flatten(0, 2)
    output *= 255
    output = output.cpu().numpy().astype(np.uint8)[:,:,None]
    
    return output
       



def load_matter(ckpt_path, device):
    """
    Load the matting model.

    """
    cfg = LazyConfig.load('MEMatte/configs/MixData_ViTMatte_S_topk0.25_1024_distill.py')
    cfg.model.teacher_backbone = None
    cfg.model.backbone.max_number_token = 12000
    matmodel = instantiate(cfg.model)
    matmodel = matmodel.to(device)
    matmodel.eval()
    DetectionCheckpointer(matmodel).load(ckpt_path)
    
    return matmodel



def load_model(config_path, ckpt_path, device):

    cfg = LazyConfig.load(config_path)

    model = instantiate(cfg.model)

    model.lora_rank = 4
    model.lora_alpha = 4
    model.init_lora()

    model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)  
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print(ckpt_path)
    return model


def preprocess_inputs(batched_inputs, device):
    """
    Normalize, pad and batch the input images.
    """
    pixel_mean = torch.Tensor([123.675 / 255., 116.280 / 255., 103.530 / 255.]).view(-1, 1, 1).to(device)
    pixel_std = torch.Tensor([58.395 / 255., 57.120 / 255., 57.375 / 255.]).view(-1, 1, 1).to(device)

    output = dict()
    images = batched_inputs["image"].to(device)
    images = (images - pixel_mean) / pixel_std
    assert images.shape[-2] == images.shape[-1] == 1024

    bbox = batched_inputs["bbox"].to(device)
    click = -1 * torch.ones((1,9,3),dtype=torch.float64).to(device)
    output['images'] = images
    output['bbox'] = bbox
    output['click'] = click
    return output


def resize_image_prompt(prompts):
    [[x1, y1, _, x2, y2, _]] = prompts["bbox"]

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    img = prompts["image"]

    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    ori_H, ori_W, _ = prompts["image"].shape
    scale_x = 1024.0 / ori_W
    scale_y = 1024.0 / ori_H
    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    bbox = np.clip(np.array([[x1, y1, x2, y2]]) * 1.0, 0, 1024.0)

    return img, bbox, (ori_H, ori_W), (1024, 1024)




def calculate_metrics(pred_alpha, gt_alpha):
    """
    Calculate MSE and SAD metrics.
    """
    pred_alpha = pred_alpha.astype(np.float32) / 255.0
    gt_alpha = gt_alpha.astype(np.float32) / 255.0

    # MSE
    mse = np.mean((pred_alpha - gt_alpha) ** 2)

    # SAD
    sad = np.sum(np.abs(pred_alpha - gt_alpha)) / 1000.0  # Normalize by 1000

    return mse, sad




def batch_inference(args, image_folder, alpha_folder, output_folder):


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    device = torch.device(args.device)
    model = load_model(args.mattepro_config_path, args.mattepro_ckpt_path, device)
    matter = load_matter(args.mematte_ckpt_path, device)

    image_files = sorted(os.listdir(image_folder))
    alpha_files = sorted(os.listdir(alpha_folder))

    metrics = []

    for image_file, alpha_file in tqdm(zip(image_files, alpha_files), total=len(image_files)):
        image_path = os.path.join(image_folder, image_file)
        alpha_path = os.path.join(alpha_folder, alpha_file)
        image = cv2.imread(image_path)
        gt_alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

        bbox = get_bbox_from_alpha(gt_alpha)
        prompts = {"image": image, "bbox": [[bbox[0], bbox[1], 2, bbox[2], bbox[3], 3]], "click": image}

        box_ori = copy.deepcopy([[bbox[0], bbox[1], bbox[2], bbox[3]]])


        image_resized, bbox_resized, ori_H_W, pad_H_W = resize_image_prompt(prompts)

        input_data = {
            'image': torch.from_numpy(image_resized)[None].to(device),
            'bbox': torch.from_numpy(bbox_resized)[None].to(device),
        }

        with torch.no_grad():
            inputs = preprocess_inputs(input_data, device)
            images, bbox, click = inputs['images'], inputs['bbox'], inputs['click']
            trimap_logits = model.forward((images, (click,bbox)))
           
            pred = F.interpolate(trimap_logits, size=ori_H_W, mode='bilinear', align_corners=False)
            trimap_pred = torch.clip(torch.argmax(pred, dim=1) * 128, min=0, max=255)[0].cpu().numpy().astype(np.uint8)
            pred_alpha = matting_inference(matter, image, trimap_pred, args.pachify, device).squeeze()

        # Save alpha matte
        alpha_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.png")
        Image.fromarray(pred_alpha).save(alpha_output_path)

        # Calculate metrics
        mse, sad = calculate_metrics(pred_alpha, gt_alpha)

        metrics.append((image_file, mse, sad))

    # Calculate and print average metrics
    avg_mse = np.mean([m[1] for m in metrics])
    avg_sad = np.mean([m[2] for m in metrics])

    print(f"\nAverage Metrics: MSE: {avg_mse:.6f}, SAD: {avg_sad:.6f}")

def get_benchmark(args):

    if args.testset == 'AIM500':
        image_folder = os.path.join(args.config['AIM500_PATH'],'original')
        alpha_folder = os.path.join(args.config['AIM500_PATH'],'mask')
        output_folder = './results/aim500'
    if args.testset == 'C1K':
        image_folder = os.path.join(args.config['C1K_PATH'],'merged')
        alpha_folder = os.path.join(args.config['C1K_PATH'],'alpha_copy')
        output_folder = './results/c1k'
    if args.testset == 'P3M500':
        image_folder = os.path.join(args.config['P3M500_PATH'],'original_image')
        alpha_folder = os.path.join(args.config['P3M500_PATH'],'mask')
        output_folder = './results/p3m500'

    return image_folder, alpha_folder, output_folder


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mattepro-config-path', default='configs/MattePro_SAM2.py', type=str)

    parser.add_argument('--mattepro-ckpt-path', default='weights/MattePro.pth', type=str)
    
    parser.add_argument('--mematte-ckpt-path', default='weights/MEMatte.pth', type=str)
    
    parser.add_argument('--testset', default='AIM500', type=str, choices=['AIM500', 'C1K', 'P3M500'])

    parser.add_argument('--device', default='cuda:0', type=str)

    parser.add_argument('--pachify', default=False, type=bool, help="whether to use memory efficient inference")

    return parser.parse_args()


if __name__ == '__main__':
 
    args = parse_args()

    with open('config.yml', 'r') as file:
        args.config = yaml.safe_load(file)

    image_folder, alpha_folder, output_folder = get_benchmark(args)

    batch_inference(args, image_folder, alpha_folder, output_folder)
    print(args.mattepro_ckpt_path)
