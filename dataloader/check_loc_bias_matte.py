import cv2
import numpy as np
import os
import random

class ImageCompositor:
    def __init__(self, is_composite=True):
        self.is_composite = is_composite

    def apply_scaling_and_offset(self, fg, bg, alpha, trimap):
        """
        对样本应用缩放和随机偏移。
        
        参数:
            fg (ndarray): 前景图像。
            bg (ndarray): 背景图像。
            alpha (ndarray): Alpha掩码。
            trimap (ndarray): Trimap掩码。
        
        返回:
            tuple: 合成后的图像，更新后的alpha和trimap。
        """
        # 获取尺寸
        bg_h, bg_w = bg.shape[:2]
        fg_h, fg_w = fg.shape[:2]

        # 确保前景图像适合背景图像
        if fg_h > bg_h or fg_w > bg_w:
            # 如果前景图像大于背景图像，调整前景图像为背景图像的大小
            fg = cv2.resize(fg, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
            trimap = cv2.resize(trimap, (bg_w, bg_h), interpolation=cv2.INTER_NEAREST)

        # 随机缩放因子
        scale_factor = np.random.uniform(0.25, 1.0)
        
        # 确保缩放后的前景图像适合背景尺寸
        while fg_h * scale_factor > bg_h or fg_w * scale_factor > bg_w:
            scale_factor = np.random.uniform(0.25, 1.0)
            
        new_size = (int(fg.shape[1] * scale_factor), int(fg.shape[0] * scale_factor))

        # 缩放前景、alpha和trimap
        fg = cv2.resize(fg, new_size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, new_size, interpolation=cv2.INTER_NEAREST)

        # 确保前景图像居中放置在背景图像中
        x_offset = (bg_w - fg.shape[1]) // 2
        y_offset = (bg_h - fg.shape[0]) // 2

        # 创建空的alpha和trimap数组
        alpha_paste = np.zeros((bg_h, bg_w), dtype=alpha.dtype)
        trimap_paste = np.zeros((bg_h, bg_w), dtype=trimap.dtype)

        # 将alpha和trimap放置到新的位置
        alpha_paste[y_offset:y_offset + fg.shape[0], x_offset:x_offset + fg.shape[1]] = alpha
        trimap_paste[y_offset:y_offset + fg.shape[0], x_offset:x_offset + fg.shape[1]] = trimap

        # 合成图像
        if self.is_composite:
            # 合成前景和背景
            bg_patch = bg[y_offset:y_offset + fg.shape[0], x_offset:x_offset + fg.shape[1]]
            composite_patch = fg * alpha[:, :, None] + bg_patch * (1 - alpha[:, :, None])
            bg[y_offset:y_offset + fg.shape[0], x_offset:x_offset + fg.shape[1]] = composite_patch
            return bg, alpha_paste, trimap_paste
        else:
            # 直接粘贴前景图像
            bg[y_offset:y_offset + fg.shape[0], x_offset:x_offset + fg.shape[1]] = fg
            return bg, alpha_paste, trimap_paste


def process_images(fg_folder, bg_folder, alpha_folder, output_image_folder, output_alpha_folder):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_alpha_folder):
        os.makedirs(output_alpha_folder)

    compositor = ImageCompositor()

    # Get all files from the directories
    fg_files = sorted([f for f in os.listdir(fg_folder) if f.endswith('.jpg')])
    bg_files = sorted([f for f in os.listdir(bg_folder) if f.endswith('.jpg')])
    alpha_files = sorted([f for f in os.listdir(alpha_folder) if f.endswith('.jpg')])

    for fg_file, bg_file, alpha_file in zip(fg_files, bg_files, alpha_files):
        fg_path = os.path.join(fg_folder, fg_file)
        bg_path = os.path.join(bg_folder, bg_file)
        alpha_path = os.path.join(alpha_folder, alpha_file)

        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)  # Read as 4 channels if possible (RGBA)
        bg = cv2.imread(bg_path)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)  # Read as single channel

        if fg is not None and bg is not None and alpha is not None:
            # Generate the composite image and updated alpha
            composite_img, updated_alpha, updated_trimap = compositor.apply_scaling_and_offset(fg, bg, alpha, alpha)

            # Save the composite image and alpha
            composite_img_path = os.path.join(output_image_folder, fg_file)
            updated_alpha_path = os.path.join(output_alpha_folder, alpha_file)

            cv2.imwrite(composite_img_path, composite_img)
            cv2.imwrite(updated_alpha_path, updated_alpha)
            print(f"Processed and saved: {fg_file}")

        else:
            print(f"Failed to load: {fg_file}, {bg_file}, {alpha_file}")


if __name__ == "__main__":
    # Set paths for foreground, background, alpha images, and output folders
    fg_folder = "/opt/data/private/zcy/datasets/matte_for_matting/fg"  # Replace with your foreground image folder path
    bg_folder = "/opt/data/private/zcy/datasets/cocolvis/train/images"  # Replace with your background image folder path
    alpha_folder = "/opt/data/private/zcy/datasets/matte_for_matting/alpha"  # Replace with your alpha image folder path
    output_image_folder = "images"  # Replace with your output folder for composite images
    output_alpha_folder = "alpha"  # Replace with your output folder for alpha images

    process_images(fg_folder, bg_folder, alpha_folder, output_image_folder, output_alpha_folder)
