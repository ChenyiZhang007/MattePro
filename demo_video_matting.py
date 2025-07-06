import os
import sys
sys.path.append('./BiRefNet')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from tqdm import tqdm
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import argparse
import numpy as np
from BiRefNet.models.birefnet import BiRefNet
from torchvision import transforms


def init_biref(device):

    model = BiRefNet(bb_pretrained=False)


    # Load model weights
    state_dict = torch.load('weights/BiRefNet-HRSOD_D-epoch_130.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model




def matting_inference(model, rawimg, trimap, device):
    """
    Perform matting inference.
    """

    data = {}
    data['trimap'] = torchvision.transforms.functional.to_tensor(trimap)[0:1, :, :].unsqueeze(0)
    data['image'] = torchvision.transforms.functional.to_tensor(rawimg).unsqueeze(0)
    for k in data.keys():
        data[k].to(model.device)
    patch_decoder = True
    output = model(data, patch_decoder)[0]['phas'].flatten(0, 2)
    # output = model(data, patch_decoder)['phas'].flatten(0, 2)
    # trimap = data['trimap'].squeeze(0).squeeze(0)
    # output[trimap == 0] = 0
    # output[trimap == 1] = 1
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
    model.lora_alpha = model.lora_rank
    model.init_lora()

    model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    return model

def preprocess_image(image, image_size):
    """
    Preprocess the image for inference.
    Args:
        image_path (str): Path to the input image.
        image_size (tuple): Target size (height, width).
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load image using PIL
    # image = Image.open(image_path).convert('RGB')
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # Define the transform pipeline
    transform_image = transforms.Compose([
        transforms.Resize(image_size[::-1]),  # Resize to target size
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    # Apply transformations
    image = transform_image(image)
    return image.unsqueeze(0)  # Add batch dimension

def get_bbox_from_birefnet(model,image,device):

    image_size=(1024,1024)
    input_image = preprocess_image(image, image_size).to(device)

    with torch.no_grad():
        scaled_preds = model(input_image)[-1].sigmoid()

    # Resize prediction back to original size
    # original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    res = torch.nn.functional.interpolate(
        scaled_preds,
        size=image.shape[:2],
        mode='bilinear',
        align_corners=True
    )
    binary_mask = (res>0.5)[0,0].cpu().numpy()

    if binary_mask.sum() == 0:
        binary_mask = np.ones_like(binary_mask)

    
    rows = np.any(binary_mask, axis=1) 
    cols = np.any(binary_mask, axis=0)  

    if rows.sum()==1 or cols.sum()==1:
        binary_mask = np.ones_like(binary_mask)
        rows = np.any(binary_mask, axis=1) 
        cols = np.any(binary_mask, axis=0)  


    y_min, y_max = np.where(rows)[0][[0, -1]]  
    x_min, x_max = np.where(cols)[0][[0, -1]]  


    return [x_min, y_min, x_max, y_max]


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


def batch_inference(args, input_path, output_path):


    device = torch.device(args.device)
    model = load_model(args.mattepro_config_path, args.mattepro_ckpt_path, device)
    matter = load_matter(args.mematte_ckpt_path, device)


    birefnet = init_biref(device)

    # Set up input/output paths

    # Open video file

    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), False)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames with progress bar
    for frame_idx in tqdm(range(total_frames), desc=f"Processing {input_path}"):
        ret, frame = cap.read()
        if not ret:
            break
            

        bbox = get_bbox_from_birefnet(birefnet, frame, device)
        prompts = {"image": frame, "bbox": [[bbox[0], bbox[1], 2, bbox[2], bbox[3], 3]], "click": frame}

        image_resized, bbox_resized, ori_H_W, pad_H_W = resize_image_prompt(prompts)
        input_data = {
            'image': torch.from_numpy(image_resized)[None].to(device),
            'bbox': torch.from_numpy(bbox_resized)[None].to(device),
        }
        
        # Generate alpha matte
        with torch.no_grad():
            inputs = preprocess_inputs(input_data, device)
            images, bbox, click = inputs['images'], inputs['bbox'], inputs['click']
            pred = model.forward((images, (click,bbox)))
            pred = F.interpolate(pred, size=ori_H_W, mode='bilinear', align_corners=False)
            trimap_pred = torch.clip(torch.argmax(pred, dim=1) * 128, min=0, max=255)[0].cpu().numpy().astype(np.uint8)
            alpha = matting_inference(matter, frame, trimap_pred, device).squeeze()
            
        out.write(alpha)
        
    cap.release()
    out.release()



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mattepro-config-path', default='configs/MattePro_SAM2.py', type=str)

    parser.add_argument('--mattepro-ckpt-path', default='weights/MattePro.pth', type=str)
    
    parser.add_argument('--mematte-ckpt-path', default='weights/MEMatte.pth', type=str)
    
    parser.add_argument('--input-path', default='videos/example.mp4', type=str, help='a directory containing test videos')

    parser.add_argument('--output-path', default='output.mp4', type=str, help='output path for the results')

    parser.add_argument('--device', default='cuda:0', type=str)


    return parser.parse_args()


if __name__ == '__main__':
 
    args = parse_args()

    batch_inference(args, args.input_path, args.output_path)

