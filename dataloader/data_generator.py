import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
from isegm.data.sample import DSample
import matplotlib.pyplot as plt

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms

from   utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

import torch
import numpy as np




def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap, mask = sample['image'][:,:,::-1], sample['alpha'], sample['trimap'], sample['mask']
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
     
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1
        
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        # normalize image
        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            # del sample['image_name']

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        sample['mask'] = torch.from_numpy(mask).float()

        return sample
    
class RandomHorizontalFlip(object):
    """
    Random flip image, fg, alpha, trimap, and mask horizontally.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        fg, alpha, trimap, mask, image = sample['fg'], sample['alpha'], sample['trimap'], sample['mask'], sample['image']
        
        # Flip with a certain probability
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
            trimap = cv2.flip(trimap, 1)
            mask = cv2.flip(mask, 1)
            image = cv2.flip(image, 1)  # Flip the image as well

        sample['fg'], sample['alpha'], sample['trimap'], sample['mask'], sample['image'] = fg, alpha, trimap, mask, image
        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class GenMask_real(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        # Perform connected component analysis
        _, labels, stats, centroids = cv2.connectedComponentsWithStats((alpha_ori > 0).astype(np.uint8), connectivity=8)

        
        # Exclude the background (label 0)
        num_components = len(np.unique(labels)) - 1
        valid_components = []

        for i in range(1, num_components + 1):
            # Get the area of each component (stat[cv2.CC_STAT_AREA])
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:
                valid_components.append(i)
        
        if valid_components:
            # Randomly select one of the valid components (area > 500)
            chosen_component = np.random.choice(valid_components)
            
            # Create a mask that only contains the chosen component
            component_mask = (labels == chosen_component).astype(np.uint8)
        else:
            # If no valid component, no connected component analysis is done
            component_mask = np.ones_like(alpha_ori, dtype=np.uint8)  # No modification, keep original alpha_ori

        # Apply the component mask to alpha_ori
        alpha_ori = alpha_ori * component_mask

        max_kernel_size = 15  # 30
        alpha = cv2.resize(alpha_ori, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        fg_width = np.random.randint(1, max_kernel_size)
        bg_width = np.random.randint(1, max_kernel_size)
        fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        # Generate mask for segmentation
        seg_mask = (alpha >= 0.01).astype(int).astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask
        sample['alpha'] = alpha_ori

        return sample

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample



    
class BGMerge(object):
    def __init__(self,prob=0.5):
        self.prob=prob
        pass
    def __call__(self, sample):

        if np.random.rand < self.prob:
            return sample
        else:
            bg = sample['bg']
            fg2 = sample['fg2']
            alpha_2 = sample['alpha_2']


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( 1024, 1024)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, mask, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['mask'], sample['image_name']
        fg_2, alpha_2 = sample['fg_2'],  sample['alpha_2']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                fg_2 = cv2.resize(fg_2, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha_2 = cv2.resize(alpha_2, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask = cv2.resize(mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        # fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        # alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        # bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        # trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        # mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        fg_crop_2 = cv2.resize(fg_2, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        alpha_crop_2 = cv2.resize(alpha_2, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)


        # if len(np.where(trimap==128)[0]) == 0:
        #     self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
        #                         "left_top: {}".format(name, left_top))
        #     fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        #     alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        #     trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        #     bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        #     mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        sample.update({'fg': fg_crop, 'alpha': alpha_crop, 'fg_2': fg_crop_2, 'alpha_2': alpha_crop_2, 'trimap': trimap_crop, 'mask': mask_crop, 'bg': bg_crop})
        sample['image'] = sample['fg']
        return sample

class ValCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        alpha, trimap, mask, name = sample['alpha'], sample['trimap'], sample['mask'], sample['image_name']
        # bg = sample['bg']
        image = sample['image']
        h, w = trimap.shape
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        # if len(unknown_list) < 10:
        #     left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        # else:
        #     idx = np.random.randint(unknown_num)
        #     left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)
        left_top = None

        # fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        # alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        # bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        # trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        # mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        image = cv2.resize(image, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        # bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)


        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            image = cv2.resize(image, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            # bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        
        sample.update({'image': image, 'alpha': alpha_crop, 'trimap': trimap_crop, 'mask': mask_crop})
        return sample

class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['mask'] = padded_mask

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        max_kernel_size = 15 # 30
        alpha = cv2.resize(alpha_ori, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))



        fg_width = np.random.randint(1, max_kernel_size)
        bg_width = np.random.randint(1, max_kernel_size)
        fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])




        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        ### generate mask
        # low = 0.01
        # high = 1.0
        # thres = random.random() * (high - low) + low
        seg_mask = (alpha >= 0.01).astype(int).astype(np.uint8)
        # random_num = random.randint(0,3)
        # if random_num == 0:
        #     seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        # elif random_num == 1:
        #     seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        # elif random_num == 2:
        #     seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        #     seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        # elif random_num == 3:
        # kernel = np.ones((5,5), np.uint8)
        # seg_mask = cv2.erode(seg_mask, kernel, iterations=np.random.randint(1, 5))
        # seg_mask = cv2.dilate(seg_mask, kernel, iterations=np.random.randint(1, 5))
        
        
        
    
        seg_mask = cv2.resize(seg_mask, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask

        return sample



class Composite(object):
    def __init__(self, is_composite, loc_aug):
        """
        Initialize Composite object.
        
        Args:
            is_composite (bool): If True, performs compositing; otherwise, only scaling and offset.
            loc_aug (bool): If True, applies location augmentation (scaling and random positioning).
        """
        self.is_composite = is_composite
        self.loc_aug = loc_aug

    def __call__(self, sample):
        """
        Processes the input sample to perform compositing and/or augmentation.
        
        Args:
            sample (dict): A dictionary containing 'fg', 'bg', 'alpha', 'trimap', and optionally 'fg2' and 'alpha2'.
        
        Returns:
            dict: Updated sample with composited or transformed images.
        """
        fg, bg, alpha, trimap = sample['fg'], sample['bg'], sample['alpha'], sample['trimap']

        # Validate shapes
        if fg.shape[:2] != bg.shape[:2] or fg.shape[:2] != alpha.shape or fg.shape[:2] != trimap.shape:
            raise ValueError("Shape mismatch between fg, bg, alpha, and trimap.")

        # Clamp values
        alpha = np.clip(alpha, 0, 1)
        fg = np.clip(fg, 0, 255)
        bg = np.clip(bg, 0, 255)

        # If no compositing or augmentation is needed, return the foreground as-is
        if not (self.is_composite and self.loc_aug):
            sample['image'] = fg
            return sample

        # If second foreground and alpha are provided, apply them first
        if np.random.rand() < 0.25:
            sample = self._apply_second_fg(sample)

        # Perform scaling and augmentation
        sample = self._apply_scaling_and_offset(sample)

        return sample

    def _apply_scaling_and_offset(self, sample):
        """
        Apply scaling and random offset to the sample.
        
        Args:
            sample (dict): The input sample.
        
        Returns:
            dict: Updated sample with scaling and random offset applied.
        """
        fg, bg, alpha, trimap = sample['fg'], sample['bg'], sample['alpha'], sample['trimap']

        # Random scale factor
        scale_factor = np.random.uniform(0.25, 1.0)
        new_size = (int(fg.shape[1] * scale_factor), int(fg.shape[0] * scale_factor))

        # Resize foreground, alpha, and trimap
        fg = cv2.resize(fg, new_size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, new_size, interpolation=cv2.INTER_NEAREST)

        # Get dimensions
        bg_h, bg_w = bg.shape[:2]
        fg_h, fg_w = fg.shape[:2]

        # Ensure foreground fits within background
        if fg_h > bg_h or fg_w > bg_w:
            raise ValueError("Foreground is larger than the background after scaling.")

        # Random position for top-left corner
        x_offset = np.random.randint(0, bg_w - fg_w + 1)
        y_offset = np.random.randint(0, bg_h - fg_h + 1)

        # Create empty alpha and trimap arrays
        alpha_paste = np.zeros((bg_h, bg_w), dtype=alpha.dtype)
        trimap_paste = np.zeros((bg_h, bg_w), dtype=trimap.dtype)

        # Place alpha and trimap into their new positions
        alpha_paste[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = alpha
        trimap_paste[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = trimap

        # Composite image
        if self.is_composite:
            # Composite foreground and background
            bg_patch = bg[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w]
            composite_patch = fg * alpha[:, :, None] + bg_patch * (1 - alpha[:, :, None])
            bg[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = composite_patch
            sample['image'] = bg
        else:
            # Paste foreground without blending
            bg[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = fg
            sample['image'] = bg

        # Update sample with new alpha and trimap
        sample['alpha'] = alpha_paste
        sample['trimap'] = trimap_paste

        return sample

    def _apply_second_fg(self, sample):
        """
        Apply the second foreground and its alpha to the background.
        
        Args:
            sample (dict): The input sample containing 'fg2' and 'alpha2'.
        
        Returns:
            dict: Updated sample with the second foreground composited.
        """
        fg2, alpha2, bg = sample['fg_2'], sample['alpha_2'], sample['bg']

        # Clamp values
        alpha2 = np.clip(alpha2, 0, 1)
        fg2 = np.clip(fg2, 0, 255)

        # Random scale factor for the second foreground
        scale_factor = np.random.uniform(0.5, 1.0)
        new_size = (int(fg2.shape[1] * scale_factor), int(fg2.shape[0] * scale_factor))

        # Resize the second foreground and its alpha
        fg2 = cv2.resize(fg2, new_size, interpolation=cv2.INTER_LINEAR)
        alpha2 = cv2.resize(alpha2, new_size, interpolation=cv2.INTER_LINEAR)

        # Get dimensions
        bg_h, bg_w = bg.shape[:2]
        fg2_h, fg2_w = fg2.shape[:2]

        # Ensure the second foreground fits within the background
        if fg2_h > bg_h or fg2_w > bg_w:
            raise ValueError("Second foreground is larger than the background after scaling.")

        # Random position for top-left corner
        x_offset = np.random.randint(0, bg_w - fg2_w + 1)
        y_offset = np.random.randint(0, bg_h - fg2_h + 1)

        # Composite the second foreground onto the background
        bg_patch = bg[y_offset:y_offset + fg2_h, x_offset:x_offset + fg2_w]
        composite_patch = fg2 * alpha2[:, :, None] + bg_patch * (1 - alpha2[:, :, None])
        bg[y_offset:y_offset + fg2_h, x_offset:x_offset + fg2_w] = composite_patch

        # Update the sample with the new background image
        sample['bg'] = bg

        return sample


class CutMask(object):
    def __init__(self, perturb_prob = 0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample['mask'] # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)
        
        mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
        
        sample['mask'] = mask
        return sample


class DataGenerator(Dataset):
    def __init__(self, data, phase="train"):
        self.phase = phase
        self.crop_size = 1024
        # self.alpha = data.alpha

        self.bbox_offset_factor = 0.1
        
        if self.phase == "train":

            self.real_cca_alpha_list = data.real_cca_alpha_list
            self.real_cca_fg_list = data.real_cca_fg_list

            self.composite_alpha_list = data.composite_alpha_list
            self.composite_fg_list = data.composite_fg_list

            self.real_portrait_alpha_list = data.real_portrait_alpha_list
            self.real_portrait_fg_list = data.real_portrait_fg_list

            self.real_glass_alpha_list = data.real_glass_alpha_list
            self.real_glass_fg_list = data.real_glass_fg_list

            self.bg = data.bg
            self.merged = []
            self.trimap = []

        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        
        train_trans_real_cca = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask_real(),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(is_composite=False,loc_aug=False),
            RandomHorizontalFlip(prob=0.5),  # Add RandomHorizontalFlip here
            ToTensor(phase="train")]

        train_trans_real = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(is_composite=False,loc_aug=False),
            RandomHorizontalFlip(prob=0.5),  # Add RandomHorizontalFlip here
            ToTensor(phase="train")]
        
        train_trans_composite = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(is_composite=True,loc_aug=True),
            RandomHorizontalFlip(prob=0.5),  # Add RandomHorizontalFlip here
            ToTensor(phase="train")]
        
       

        test_trans = [ OriginScale(), ToTensor() ]

        self.transform = {
            'train_real':
                transforms.Compose(train_trans_real),
            'train_real_cca':
                transforms.Compose(train_trans_real_cca),
            'train_composite':
                transforms.Compose(train_trans_composite),

            
            'val':
                transforms.Compose([
                    ValCrop((self.crop_size, self.crop_size)),
                    ToTensor(phase="val")
                ]),
            'test':
                transforms.Compose(test_trans)
        }


        self.real_cca_alpha_num = len(self.real_cca_alpha_list)
        self.composite_alpha_num = len(self.composite_alpha_list)
        self.real_portrait_alpha_num = len(self.real_portrait_alpha_list)
        self.real_glass_alpha_num = len(self.real_glass_alpha_list)

        self.bg_num = len(self.bg)

    def __getitem__(self, idx):

        data_candidate = ['real_cca', 'composite', 'real_glass', 'real_portrait']
        weights = [0.3, 0.3, 0.2, 0.2]


        data_type = random.choices(data_candidate, weights=weights, k=1)[0]

        if data_type == 'real_cca':
            phase = "train_real_cca"
            transforms = self.transform[phase]
            fg_index = random.randint(0, self.real_cca_alpha_num-1)
            fg = cv2.imread(self.real_cca_fg_list[fg_index])
            alpha = cv2.imread(self.real_cca_alpha_list[fg_index], 0).astype(np.float32)/255
            image_name = os.path.split(self.real_cca_fg_list[fg_index])[-1]

        if (data_type == 'real_portrait') or (data_type == 'real_glass'):
            if np.random.rand() < 0.5:
                phase = "train_real"
                transforms = self.transform[phase]
                fg_index = random.randint(0, self.real_portrait_alpha_num-1)
                fg = cv2.imread(self.real_portrait_fg_list[fg_index])
                alpha = cv2.imread(self.real_portrait_alpha_list[fg_index], 0).astype(np.float32)/255
                image_name = os.path.split(self.real_portrait_fg_list[fg_index])[-1]
            else:
                phase = "train_real"
                transforms = self.transform[phase]
                fg_index = random.randint(0, self.real_glass_alpha_num-1)
                fg = cv2.imread(self.real_glass_fg_list[fg_index])
                alpha = cv2.imread(self.real_glass_alpha_list[fg_index], 0).astype(np.float32)/255
                image_name = os.path.split(self.real_glass_fg_list[fg_index])[-1]

        if data_type == 'composite':
            phase = "train_composite"
            transforms = self.transform[phase]
            fg_index = random.randint(0, self.composite_alpha_num-1)
            fg = cv2.imread(self.composite_fg_list[fg_index])
            alpha = cv2.imread(self.composite_alpha_list[fg_index], 0).astype(np.float32)/255
            image_name = os.path.split(self.composite_fg_list[fg_index])[-1]


        fg_index_2 = random.randint(0, self.composite_alpha_num-1)
        fg_2 = cv2.imread(self.composite_fg_list[fg_index_2])
        alpha_2 = cv2.imread(self.composite_alpha_list[fg_index_2], 0).astype(np.float32)/255
        
        
        bg = cv2.imread(self.bg[random.randint(0, self.bg_num-1)], 1)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        sample = {'fg': fg, 'alpha': alpha,'fg_2': fg_2, 'alpha_2': alpha_2, 'bg': bg, 'image_name': image_name}
        sample = transforms(sample)
 

        points_array = -1 * np.ones((24*3,3))

        sample = self.generate_prompt(sample)

        sample['points'] = points_array
        sample['instances'] = sample['mask']

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        # if np.random.rand() < 0.25:
        #     fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        #     alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha
    
    def mask_to_bbox(self, mask):
        # 获取非零区域的行和列的索引
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # 找到非零区域的边界
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        if self.bbox_offset_factor:

            bbox_w = max(1, x_max - x_min)
            bbox_h = max(1, y_max - y_min)
            offset_w = math.ceil(self.bbox_offset_factor * bbox_w)
            offset_h = math.ceil(self.bbox_offset_factor * bbox_h)

            x_min = max(0, x_min + np.random.randint(-offset_w, offset_w))
            x_max = min(mask.shape[0] - 1, x_max + np.random.randint(-offset_w, offset_w))
            y_min = max(0, y_min + np.random.randint(-offset_h, offset_h))
            y_max = min(mask.shape[1] - 1, y_max + np.random.randint(-offset_h, offset_h))

            # x_min = max(0, x_min - np.random.randint(0, offset_w))
            # x_max = min(mask.shape[0] - 1, x_max + np.random.randint(0, offset_w))
            # y_min = max(0, y_min - np.random.randint(0, offset_h))
            # y_max = min(mask.shape[1] - 1, y_max + np.random.randint(0, offset_h))
  
        
        # 返回四角坐标
        return (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)
    
    
    def sample_points_from_mask(self, tensor_mask, num_points, flag):
        
         # 将 (1, w, h) 变成 (w, h) 的二维掩码
        mask = tensor_mask.cpu().numpy()
        
        # 找到mask中的所有有效点（非零区域）
        valid_points = np.argwhere(mask > 0)
        
        # 如果有效点数量小于想要的点数，直接返回所有点
        if len(valid_points) <= num_points:
            points = [tuple(pt) for pt in valid_points]
            return points
        
        # 随机采样指定数量的点
        sampled_indices = np.random.choice(len(valid_points), num_points, replace=False)
        sampled_points = valid_points[sampled_indices]
        
        # 将采样的点坐标转换为 (x, y) 的形式返回
        points = [list(pt[::-1]) + [flag] for pt in sampled_points]
        return points
    




    def generate_prompt(self, sample):
        # 初始化默认值
        max_num_points = 3
        default_bbox = torch.tensor([[-1, -1, -1, -1]], dtype=torch.float64)
        default_click = torch.tensor(np.full(((max_num_points+8)*3, 3), -1, dtype=np.float64), dtype=torch.float64)
        default_rectangle = torch.zeros(1, 1024, 1024, dtype=torch.float32)  # 默认全0的rectangle

        # 初步生成 click
        
        final_tensor = np.full(((max_num_points+8) * 3, 3), -1, dtype=np.float64)  # 初始化9x3的张量，值为-1
        point_count = 0  # 用于跟踪插入点的数量

        # 定义类别点采样的最少和最多数量（可通过 self.point_limits 传入）
        point_limits = getattr(self, 'point_limits', {
            'background': (0, max_num_points),  # 背景点的最少和最多数量
            'unknown': (0, max_num_points),     # unknown 点的最少和最多数量
            'foreground': (0, max_num_points)   # 前景点的最少和最多数量
        })

        for index, mask in enumerate(sample['trimap']):
            if index == 0:  # 背景点采样
                min_points, max_points = point_limits['background']
                num_points = np.random.choice(range(min_points, max_points + 1))
                bg_points = self.sample_points_from_mask(mask, num_points, 0)
                for point in bg_points:
                    final_tensor[point_count] = [point[0], point[1], 0]
                    point_count += 1

            if index == 1:  # unknown点采样
                min_points, max_points = point_limits['unknown']
                num_points_uk = np.random.choice(range(min_points, max_points + 1))
                uk_points = self.sample_points_from_mask(mask, num_points_uk, 4)
                for point in uk_points:
                    final_tensor[point_count] = [point[0], point[1], 4]
                    point_count += 1

            if index == 2:  # 前景点采样
                min_points, max_points = point_limits['foreground']
                num_points_fg = np.random.choice(range(min_points, max_points + 1))
                fg_points = self.sample_points_from_mask(mask, num_points_fg, 1)

                # 确保前景点和unknown点总和至少为1
                total_points = len(uk_points) + len(fg_points)
                if total_points < 1:
                    if mask.sum() > 0:  # 检查前景点掩码是否有点
                        num_points_fg = 1
                        fg_points = self.sample_points_from_mask(mask, num_points_fg, 1)
                    elif sample['trimap'][1].sum() > 0:  # 检查unknown点掩码是否有点
                        num_points_uk = 1
                        uk_points = self.sample_points_from_mask(sample['trimap'][1], num_points_uk, 4)

                for point in fg_points: 
                    final_tensor[point_count] = [point[0], point[1], 1]
                    point_count += 1

        sample['click'] = torch.tensor(final_tensor, dtype=torch.float64)

        # 初步生成 bbox
        coarse_mask = (sample['alpha'] > 0)[0].numpy()
        rectangle_mask = self.mask2rectangle(coarse_mask)
        sample['rectangle'] = torch.from_numpy(rectangle_mask).unsqueeze(0)

        # 生成bbox
        corners = self.mask_to_bbox(coarse_mask)
        if corners is not None:
            x_min, y_min = corners[0]
            x_max, y_max = corners[2]
            sample['bbox'] = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float64)
        else:
            sample['bbox'] = default_bbox

        # 确保至少有一个有效值
        # prompt_type = np.random.choice(['box'])
        # prompt_type = np.random.choice(['click'])
        prompt_type = np.random.choice(['box', 'click', 'box_click'])

        if prompt_type == 'box':
            sample['click'] = default_click
        elif prompt_type == 'click':
            sample['bbox'] = default_bbox
            sample['rectangle'] = default_rectangle
        elif prompt_type == 'box_click':
            pass

        return sample



    
    def mask2rectangle(self, mask, scale_range=(0.05, 0.2)):
        """
        Process a mask image (float32 with values 0 and 1) to find its bounding rectangle,
        optionally expand the rectangle, fill the rectangle with 1, and leave the rest as 0.

        Parameters:
        - mask: np.ndarray, input binary mask (float32, values 0 and 1).
        - scale_range: tuple, range of scaling for height and width (default: (0.1, 0.2)).

        Returns:
        - filled_mask: np.ndarray, the processed mask with the bounding rectangle filled.
        """
        # Find all non-zero positions (assumes mask values are 0 and 1)
        non_zero_indices = np.argwhere(mask > 0)

        if non_zero_indices.size == 0:
            # If there are no non-zero pixels, return an empty mask
            return np.zeros_like(mask, dtype=np.float32)

        # Get bounding rectangle coordinates
        y_min, x_min = non_zero_indices.min(axis=0)
        y_max, x_max = non_zero_indices.max(axis=0)

        # Calculate height and width of the bounding rectangle
        height = y_max - y_min + 1
        width = x_max - x_min + 1

        # Calculate expansion factors
        scale = np.random.uniform(*scale_range)  # Randomly select a scale within the range
        expand_h = int(height * scale)  # Expansion for height
        expand_w = int(width * scale)   # Expansion for width

        # Adjust bounding rectangle coordinates
        y_min = max(0, y_min - expand_h)  # Ensure coordinates are within bounds
        x_min = max(0, x_min - expand_w)
        y_max = min(mask.shape[0] - 1, y_max + expand_h)
        x_max = min(mask.shape[1] - 1, x_max + expand_w)

        # Create a new mask and fill the expanded bounding rectangle
        filled_mask = np.zeros_like(mask, dtype=np.float32)
        filled_mask[y_min:y_max+1, x_min:x_max+1] = 1.0

        return filled_mask

    def __len__(self):
        if self.phase == "train":
            return 6000
            # return len(self.bg)
        else:
            return len(self.alpha)