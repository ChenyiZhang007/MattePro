import argparse
import gradio as gr
from gradio_image_prompter import ImagePrompter
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision



def matting_inference(model,rawimg,trimap,device):

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
    print(ckpt_path)
    return model



def preprocess_inputs(batched_inputs,device):
    """
    Normalize, pad and batch the input images.
    """

    pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
    pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],

    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)


    output = dict()

    if "alpha" in batched_inputs:
        alpha = batched_inputs["alpha"].to(device)
    else:
        alpha = None

    bbox = batched_inputs["bbox"].to(device)
    click = batched_inputs["click"].to(device)

    
    images = batched_inputs["image"].to(device)
    images = (images - pixel_mean) / pixel_std
    assert images.shape[-2] == images.shape[-1] == 1024

    if 'trimap' in batched_inputs.keys():
        trimap = batched_inputs["trimap"].to(device)
        assert len(torch.unique(trimap)) <= 3
    else:
        trimap = None

    output['images'] = images
    output['bbox'] = bbox
    output['click'] = click
    output['alpha'] = alpha
    output['trimap'] = trimap

    if 'hr_images' in batched_inputs.keys():
        hr_images = batched_inputs["hr_images"].to(device)
        hr_images = (hr_images - pixel_mean) / pixel_std
        _, _, H, W = hr_images.shape
        if hr_images.shape[-1] % 16 != 0 or hr_images.shape[-2] % 16 != 0:
            new_H = (16 - hr_images.shape[-2] % 16) + H if hr_images.shape[-2] % 16 != 0 else H
            new_W = (16 - hr_images.shape[-1] % 16) + W if hr_images.shape[-1] % 16 != 0 else W
            new_hr_images = torch.zeros((hr_images.shape[0], hr_images.shape[1], new_H, new_W)).to(device)
            new_hr_images[:,:,:H,:W] = hr_images[:,:,:,:]
            del hr_images
            hr_images = new_hr_images
        output['hr_images'] = hr_images
        output['hr_images_ori_h_w'] = (H, W)

    if 'dataset_name' in batched_inputs.keys():
        output['dataset_name'] = batched_inputs["dataset_name"]


    output['condition'] = None

    return output

def transform_image_bbox(prompts):
    if len(prompts["points"]) != 1:
        raise gr.Error("Please input only one BBox.", duration=5)
    [[x1, y1, idx_3, x2, y2, idx_6]] = prompts["points"]
    if idx_3 != 2 or idx_6 != 3:
        raise gr.Error("Please input BBox instead of point.", duration=5)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    img = prompts["image"]
    ori_H, ori_W, _ = img.shape

    scale = 1024 * 1.0 / max(ori_H, ori_W)
    new_H, new_W = ori_H * scale, ori_W * scale
    new_W = int(new_W + 0.5)
    new_H = int(new_H + 0.5)

    img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    padding = np.zeros([1024, 1024, 3], dtype=img.dtype)
    padding[: new_H, : new_W, :] = img
    img = padding
    # img = img[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    [[x1, y1, _, x2, y2, _]] = prompts["points"]
    x1, y1, x2, y2 = int(x1 * scale + 0.5), int(y1 * scale + 0.5), int(x2 * scale + 0.5), int(y2 * scale + 0.5)
    offset = 10
    bbox = np.clip(np.array([[x1-offset, y1-offset, x2+offset, y2+offset]]) * 1.0, 0, 1024.0)

    return img, bbox, (ori_H, ori_W), (new_H, new_W)

def resize_coordinates(points, orig_width, orig_height, target_size=1024):
    """
    将原图坐标点转换为目标分辨率下的坐标点，同时保持flag位不变。

    参数:
    points: np.ndarray, 形状为(n, 3)，表示 n 个点的坐标和flag。
    orig_width: int, 原图的宽度。
    orig_height: int, 原图的高度。
    target_size: int, 目标图像的边长，默认为 1024。

    返回:
    np.ndarray, 形状为(n, 3)，转换后的坐标和原始flag。
    
    """

    points = [[coord[0], coord[1], flag] for coord, flag in points]
    points = np.array(points)  # 转换为 NumPy 数组


    # 计算宽度和高度的缩放比例
    scale_x = target_size / orig_width
    scale_y = target_size / orig_height
    
    # 仅缩放坐标部分
    scaled_points = points.copy()
    scaled_points[:, 0] = points[:, 0] * scale_x  # 缩放x坐标
    scaled_points[:, 1] = points[:, 1] * scale_y  # 缩放y坐标

    # flag部分保持不变
    return scaled_points

def resize_box(box, ori_H, ori_W):
    [[x1, y1, _, x2, y2, _]] = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Adjust bbox coordinates to match the new image size
    
    scale_x = 1024.0 / ori_W
    scale_y = 1024.0 / ori_H
    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    bbox = np.clip(np.array([[x1, y1, x2, y2]]) * 1.0, 0, 1024.0)

    return bbox

def resize_click(clicks, ori_H, ori_W):
    clicks = [[click[0],click[1],click[2]] for click in clicks]
    clicks = np.array(clicks)

    # 计算宽度和高度的缩放比例
    scale_x = 1024 / ori_W
    scale_y = 1024 / ori_H
    
    # 仅缩放坐标部分
    scaled_points = clicks.copy()
    scaled_points[:, 0] = clicks[:, 0] * scale_x  # 缩放x坐标
    scaled_points[:, 1] = clicks[:, 1] * scale_y  # 缩放y坐标

    # flag部分保持不变
    return scaled_points.astype(int)

def enlarge_and_mask(image, bbox, scale=1.1):
    """
    放大框的尺寸，并将框外区域设置为0像素。如果框超出了图像范围，会限制框在图像范围内。
    
    :param image: 输入图像 (numpy 数组)
    :param bbox: 框的坐标，格式为 [x1, y1, x2, y2]
    :param scale: 放大比例 (默认为1.1，表示放大10%)
    :return: 处理后的图像
    """
    # 获取原始框坐标
    x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][3], bbox[0][4]
    
    # 计算框的中心和宽高
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # 放大框的尺寸
    new_width = width * scale
    new_height = height * scale
    
    # 重新计算放大后的框坐标
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    
    # 限制框坐标在图像范围内
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_x2 = min(new_x2, image.shape[1])
    new_y2 = min(new_y2, image.shape[0])
    
    # 创建一个与原图相同大小的全黑图像
    new_image = np.zeros_like(image)
    
    # 提取图像中框内的区域并放置到新的图像上
    new_image[new_y1:new_y2, new_x1:new_x2] = image[new_y1:new_y2, new_x1:new_x2]
    
    return new_image

def resize_image_bbox(prompts,box_aug):

    # get click and bbox
    click = []
    bbox = []
    for point in prompts["points"]:
        if point[3] != 0.0 and point[4] != 0.0:
            bbox.append(point)
        else:
            if point[2]==1:
                click.append([point[0],point[1],1])
            if point[2]==5:
                click.append([point[0],point[1],4])
            if point[2]==0:
                click.append([point[0],point[1],0])

    # get image 
    img = prompts["image"]
    if box_aug:
        img = enlarge_and_mask(img, bbox, scale=1.1)
    ori_H, ori_W, _ = prompts["image"].shape
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0


    # compute click and bbox
    if not bbox:
        bbox = [[-1,-1,-1,-1]]
    else:
        bbox = resize_box(bbox, ori_H, ori_W)
    if not click:
        click = [[-1,-1,-1]]
    else:
        click = resize_click(click, ori_H, ori_W)



    return img, np.array(click), np.array(bbox), (ori_H, ori_W), (1024, 1024)


def parse_args():


    parser = argparse.ArgumentParser()

    parser.add_argument('--mattepro-config-path', default='configs/MattePro_SAM2.py', type=str)

    parser.add_argument('--mattepro-ckpt-path', default='weights/MattePro.pth', type=str)
    
    parser.add_argument('--mematte-ckpt-path', default='weights/MEMatte.pth', type=str)

    parser.add_argument('--device', default='cuda:0', type=str)

    parser.add_argument('--box-aug', default=False, type=bool)

    parser.add_argument('--show-trimap', default=True, type=bool)


    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    device = args.device

    model = load_model(args.mattepro_config_path, args.mattepro_ckpt_path, device)
    matter = load_matter(args.mematte_ckpt_path, device)



    def inference_image(prompts):
        image, click, bbox, ori_H_W, pad_H_W = resize_image_bbox(prompts,args.box_aug)
        


        input_data = {
            'image': torch.from_numpy(image)[None].to(model.device),
            'bbox': torch.from_numpy(bbox)[None].to(model.device),
            'click': torch.from_numpy(click)[None].to(model.device),
        }

        with torch.no_grad():
            inputs = preprocess_inputs(input_data, device) 
            images, bbox, gt_alpha, trimap, condition = inputs['images'], inputs['bbox'], inputs['alpha'], inputs['trimap'], inputs['condition']
            click = inputs['click']
            
            if model.backbone_condition:
                condition_proj = model.condition_embedding(condition) 
            elif model.backbone_bbox_prompt is not None or model.bbox_prompt_all_block is not None:
                condition_proj = bbox
            else:
                condition_proj = None

            prompt = (click, bbox)

            pred = model.forward((images, prompt))
            pred = F.interpolate(pred, size=prompts["image"].shape[0:2], mode='bilinear', align_corners=False)
            trimap = torch.clip(torch.argmax(pred, dim=1) * 128, min=0, max=255)[0].cpu().numpy().astype(np.uint8)

            # Apply matting inference to the result
            alpha = matting_inference(matter, prompts["image"], trimap, device).squeeze()



        # Return the results: alpha and trimap (if requested)
        if args.show_trimap:
            return alpha, trimap
        else:
            return alpha


    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column(scale=45):
                img_in = ImagePrompter(type='numpy', show_label=False, label="Input Image")
                
            with gr.Column(scale=45):
                img_out = gr.Image(type='pil', label="Pred. Alpha")

        with gr.Row():

            with gr.Column(scale=45):
                bt = gr.Button()

            if args.show_trimap:
                with gr.Column(scale=45):
                    trimap_out = gr.Image(type='pil', label="Pred. Trimap")

        if args.show_trimap:
            bt.click(inference_image, inputs=[img_in], outputs=[img_out,trimap_out]) 
        else:
            bt.click(inference_image, inputs=[img_in], outputs=[img_out]) 

    demo.launch()
