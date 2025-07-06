import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.birefnet import BiRefNet, BiRefNetC2F
from config import Config
from utils import path_to_image, save_tensor_img
from image_proc import preproc

Image.MAX_IMAGE_PIXELS = None  # Remove DecompressionBombWarning
config = Config()

# 数据预处理的 transform
def preprocess_image(image_path, image_size):
    """
    Preprocess the image for inference.
    Args:
        image_path (str): Path to the input image.
        image_size (tuple): Target size (height, width).
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load image using PIL
    image = Image.open(image_path).convert('RGB')

    # Define the transform pipeline
    transform_image = transforms.Compose([
        transforms.Resize(image_size[::-1]),  # Resize to target size
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    # Apply transformations
    image = transform_image(image)
    return image.unsqueeze(0)  # Add batch dimension


def preprocess_label(label_path, label_size):
    """
    Preprocess the label image (if available).
    Args:
        label_path (str): Path to the label image.
        label_size (tuple): Target size (height, width).
    Returns:
        torch.Tensor: Preprocessed label tensor.
    """
    label = Image.open(label_path).convert('L')  # Load as grayscale
    transform_label = transforms.Compose([
        transforms.Resize(label_size[::-1]),  # Resize to target size
        transforms.ToTensor()  # Convert to tensor
    ])
    label = transform_label(label)
    return label.unsqueeze(0)  # Add batch dimension


def inference_single_image(model, image_path, pred_path, image_size, device=0):
    """
    Perform inference on a single image.
    Args:
        model: The loaded model.
        image_path: Path to the input image.
        pred_path: Path to save the output prediction.
        image_size: Target image size (height, width).
        device: The device for computation.
    """
    # Preprocess the input image
    input_image = preprocess_image(image_path, image_size).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        scaled_preds = model(input_image)[-1].sigmoid()

    # Resize prediction back to original size
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    res = torch.nn.functional.interpolate(
        scaled_preds,
        size=original_image.shape[:2],
        mode='bilinear',
        align_corners=True
    )

    # Save the prediction result
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    save_tensor_img(res.squeeze(0), pred_path)
    print(f"Inference completed. Output saved to {pred_path}")


def main(args):
    device = config.device
    print(f'Using model: {args.ckpt}')

    # Initialize the model
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=False)

    # Load model weights
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Perform inference
    inference_single_image(
        model=model,
        image_path=args.image_path,
        pred_path=args.pred_path,
        image_size=config.size,
        device=device
    )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Single Image Inference')
    parser.add_argument('--ckpt', type=str, default='BiRefNet-matting-epoch_100.pth', help='Path to the model checkpoint')
    parser.add_argument('--image_path', type=str, default='n00007846_157282.jpg', help='Path to the input image')
    parser.add_argument('--pred_path', type=str, default='./pred.jpg', help='Path to save the output prediction')

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
