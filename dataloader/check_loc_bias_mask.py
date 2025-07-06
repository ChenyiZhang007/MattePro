import cv2
import numpy as np
import os
import random

class GenMaskReal(object):
    def __init__(self):
        # 定义不同大小的结构元素用于腐蚀操作
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        # 执行连通域分析
        _, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_ori > 0).astype(np.uint8), connectivity=8)

        # 排除背景（标签为0）
        num_components = len(np.unique(labels)) - 1
        valid_components = []

        for i in range(1, num_components + 1):
            # 获取每个连通域的面积（stat[cv2.CC_STAT_AREA]）
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:  # 只选择面积大于500的连通域
                valid_components.append(i)

        # 如果有多个有效的连通域
        if len(valid_components) > 1:
            # 随机选择一个有效的连通域
            chosen_component = random.choice(valid_components)
            
            # 创建只包含选定连通域的掩码
            component_mask = (labels == chosen_component).astype(np.uint8)
        else:
            # 如果没有多个连通域，则不做修改，不保存该图像
            component_mask = None

        # 如果有有效的处理结果，应用掩码并返回
        if component_mask is not None:
            alpha_ori = alpha_ori * component_mask
            return alpha_ori
        else:
            return None  # 表示不保存该图像


def process_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gen_mask_real = GenMaskReal()

    # 获取指定文件夹中的所有图片
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]  # 假设图片是PNG格式

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # 读取图片
        alpha_ori = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if alpha_ori is not None:
            # 生成掩码并应用
            sample = {'alpha': alpha_ori}
            processed_alpha = gen_mask_real(sample)

            # 如果处理后的图片有效，保存
            if processed_alpha is not None:
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, processed_alpha)
                print(f"Processed and saved: {image_file}")
            else:
                print(f"Skipping (no multiple components): {image_file}")
        else:
            print(f"Failed to load: {image_file}")


if __name__ == "__main__":
    # 设置图片文件夹路径和输出文件夹路径
    input_folder = "/opt/data/private/zcy/datasets/SegData_for_matting/Clean_v1/masks"  # 请替换为图片文件夹路径
    output_folder = "./dataloader/mask"  # 请替换为输出文件夹路径

    process_images(input_folder, output_folder)
