import cv2
import numpy as np
import os
import argparse

def get_structuring_kernels(max_size=15):
    kernels = {}
    for k in range(1, max_size + 1):
        kernels[k] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return kernels

def maybe_random_interp(default_interp=cv2.INTER_NEAREST):
    return np.random.choice([
        cv2.INTER_NEAREST, 
        cv2.INTER_LINEAR, 
        cv2.INTER_CUBIC, 
        cv2.INTER_AREA
    ]) if np.random.rand() < 0.5 else default_interp

def generate_trimap(alpha_path, trimap_save_path, max_kernel_size=15, resize_size=(640, 640)):
    # 读取alpha matte
    alpha_ori = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    if alpha_ori is None:
        raise FileNotFoundError(f"Cannot read alpha matte from {alpha_path}")
    h, w = alpha_ori.shape

    # 归一化为0-1之间
    alpha = alpha_ori.astype(np.float32) / 255.0
    alpha_resized = cv2.resize(alpha, resize_size, interpolation=maybe_random_interp())

    # 随机腐蚀核大小
    fg_width = np.random.randint(1, max_kernel_size)
    bg_width = np.random.randint(1, max_kernel_size)

    # 生成腐蚀核
    erosion_kernels = get_structuring_kernels(max_kernel_size)

    # 生成前景/背景掩码
    fg_mask = (alpha_resized + 1e-5).astype(np.uint8)
    bg_mask = (1 - alpha_resized + 1e-5).astype(np.uint8)
    fg_mask = cv2.erode(fg_mask, erosion_kernels[fg_width])
    bg_mask = cv2.erode(bg_mask, erosion_kernels[bg_width])

    # 构建trimap（128为不确定区域）
    trimap = np.ones_like(alpha_resized, dtype=np.uint8) * 128
    trimap[fg_mask == 1] = 255
    trimap[bg_mask == 1] = 0

    # resize回原始大小
    trimap_resized = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)

    # 保存trimap
    cv2.imwrite(trimap_save_path, trimap_resized)
    print(f"Trimap saved to {trimap_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_path', type=str, default='/opt/data/private/zcy/datasets/videomatte_1920x1080/videomatte_motion/0000/com/0000.jpg', help='Path to alpha matte image (grayscale)')
    parser.add_argument('--trimap_save_path', type=str, default='./trimap.png', help='Path to save generated trimap image')
    parser.add_argument('--max_kernel_size', type=int, default=15, help='Maximum erosion kernel size')
    args = parser.parse_args()

    generate_trimap(args.alpha_path, args.trimap_save_path, args.max_kernel_size)
