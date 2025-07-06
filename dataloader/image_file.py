import os
import glob
import logging
import functools
import numpy as np

class ImageFile(object):
    def __init__(self, phase='train'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            self.logger.error('No image valid')
        else:
            self.logger.info('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        path_list = sorted(path_list)
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)

        # name_set = set(sorted(name_set,reverse=True))
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                dim_alpha_dir='dir',
                dim_fg_dir='dir',

                d646_alpha_dir='dir',
                d646_fg_dir='dir',

                duts_img_dir='dir',
                duts_mask_dir='dir',

                dis_img_dir='dir',
                dis_mask_dir='dir',

                am2k_fg_dir='dir',
                am2k_alpha_dir='dir',  

                t460_fg_dir='dir',
                t460_alpha_dir='dir',

                p3m_fg_dir='dir',
                p3m_alpha_dir='dir',

                bg_dir='dir',
                alpha_ext=".jpg",
                fg_ext=".jpg",
                bg_ext=".jpg"):
        
        super(ImageFileTrain, self).__init__(phase="train")

        self.bg_dir     = bg_dir
        self.bg_ext     = bg_ext

        self.logger.debug('Load Training Images From Folders')

        # DIM
        valid_fg_list = self._get_valid_names(dim_alpha_dir, dim_fg_dir)
        dim_alpha_list = self._list_abspath(dim_alpha_dir, '.jpg', valid_fg_list)
        dim_fg_list = self._list_abspath(dim_fg_dir, '.jpg', valid_fg_list)

        # D646
        valid_fg_list = self._get_valid_names(d646_alpha_dir, d646_fg_dir)
        d646_alpha_list = self._list_abspath(d646_alpha_dir, '.png', valid_fg_list)
        d646_fg_list = self._list_abspath(d646_fg_dir, '.png', valid_fg_list)

        # DUTS
        with open('dataloader/duts.txt', 'r') as file:
            # 将每行内容去除换行符并添加到列表中
            valid_fg_list = [line.strip() for line in file]
        # valid_fg_list = self._get_valid_names(duts_img_dir, duts_mask_dir)
        duts_alpha_list = self._list_abspath(duts_mask_dir, '.png', valid_fg_list)
        duts_fg_list = self._list_abspath(duts_img_dir, '.jpg', valid_fg_list)

        # DIS
        with open('dataloader/dis.txt', 'r') as file:
            # 将每行内容去除换行符并添加到列表中
            valid_fg_list = [line.strip() for line in file]
        # valid_fg_list = self._get_valid_names(dis_img_dir, dis_mask_dir)
        dis_alpha_list = self._list_abspath(dis_mask_dir, '.png', valid_fg_list)
        dis_fg_list = self._list_abspath(dis_img_dir, '.jpg', valid_fg_list)

        # AM2K
        valid_fg_list = self._get_valid_names(am2k_fg_dir, am2k_alpha_dir)
        am2k_alpha_list = self._list_abspath(am2k_alpha_dir, '.png', valid_fg_list)
        am2k_fg_list = self._list_abspath(am2k_fg_dir, '.jpg', valid_fg_list)   

        # T460
        valid_fg_list = self._get_valid_names(t460_fg_dir, t460_alpha_dir)     
        delete_list = ["pascal-meier-1uVCTVSn-2o-unsplash", "o_0c8f03f0", "paul-wong-uKWdq9TcaiI-unsplash"]
        valid_fg_list =  list(filter(lambda x: x not in delete_list, valid_fg_list)) 
        t460_alpha_list = self._list_abspath(t460_alpha_dir, '.jpg', valid_fg_list)
        t460_fg_list = self._list_abspath(t460_fg_dir, '.jpg', valid_fg_list)

        # P3M
        valid_fg_list = self._get_valid_names(p3m_fg_dir, p3m_alpha_dir)
        delete_list = ["p_ec79cf7f"]
        valid_fg_list =  list(filter(lambda x: x not in delete_list, valid_fg_list))
        p3m_alpha_list = self._list_abspath(p3m_alpha_dir, '.png', valid_fg_list)
        p3m_fg_list = self._list_abspath(p3m_fg_dir, '.jpg', valid_fg_list)


        self.real_cca_alpha_list = duts_alpha_list + dis_alpha_list
        self.real_cca_fg_list = duts_fg_list + dis_fg_list

        self.composite_alpha_list = dim_alpha_list + d646_alpha_list + t460_alpha_list 
        self.composite_fg_list = dim_fg_list + d646_fg_list + t460_fg_list

        self.real_portrait_alpha_list = am2k_alpha_list + p3m_alpha_list
        self.real_portrait_fg_list = am2k_fg_list + p3m_fg_list

        self.real_glass_alpha_list = t460_alpha_list
        self.real_glass_fg_list = t460_fg_list


        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)


    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        if '500' in alpha_dir:
            merged_ext = ".jpg"

        if '2K' in alpha_dir:
            merged_ext = ".jpg"

        if '100' in alpha_dir:
            merged_ext = ".jpg"
            alpha_ext = ".jpg"
            trimap_ext = ".jpg"
        

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.logger.debug('Load Testing Images From Folders')

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = sorted(self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list))
        self.merged = sorted(self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list))
        self.trimap = sorted(self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list))

    def __len__(self):
        return len(self.alpha)

