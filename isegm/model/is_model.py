import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from isegm.model.ops import DistMaps, BatchImageNormalize, ScaleLayer
import cv2
import os
import matplotlib
matplotlib.use('Agg')

class ISModel(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, cpu_dist_maps=False,
                 use_rgb_conv=False, use_leaky_relu=False, # the two arguments only used for RITM
                 with_prev_mask=False, norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 3
        if self.with_prev_mask:
            self.coord_feature_ch += 3

        if use_rgb_conv:
            # Only RITM models need to transform the coordinate features, though they don't use 
            # exact 'rgb_conv'. We keep 'use_rgb_conv' only for compatible issues.
            # The simpleclick models use a patch embedding layer instead 
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        else:
            self.maps_transform=nn.Identity()

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image,points,rectangle):
        # image = self.prepare_input(image)

        coord_features = self.get_coord_features(image, points,rectangle)
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image,coord_features)


        cv2.imwrite('rectangle_demo.png', (coord_features[0,3].detach().cpu().numpy()*255).astype(np.uint8))

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        
        
        # if self.inference:
        #     pass
        findsucai = 0
        if findsucai:

            plt.subplot(2,3,1)
            plt.title('image')
            plt.imshow(image[0].permute(1,2,0).detach().cpu().numpy())

            plt.subplot(2,3,2)
            plt.title('gt_trimap')
            plt.imshow((self.gt_mask[0]*255).permute(1,2,0).detach().cpu().numpy())

            trimap = outputs['instances']
            trimap_mask = (trimap == torch.max(trimap, dim=1)[0].unsqueeze(1)).type(torch.uint8) * 255
            plt.subplot(2,3,3)
            plt.title('pred_trimap')
            plt.imshow(trimap_mask[0].permute(1,2,0).detach().cpu().numpy())
        
            plt.subplot(2,3,4)
            plt.title('foreclick')
            plt.imshow(coord_features[0,2].detach().cpu().numpy())

            plt.subplot(2,3,5)
            plt.title('ukclick')
            plt.imshow(coord_features[0,1].detach().cpu().numpy())

            plt.subplot(2,3,6)
            plt.title('backclick')
            plt.imshow(coord_features[0,0].detach().cpu().numpy())

            plt.suptitle('vis')
            plt.savefig('vis/error')

            # rgb = (image[0].permute(1,2,0).detach().cpu().numpy()*255)
            # cv2.imwrite('vis/rgb.jpg',rgb)
            # cv2.imwrite('vis/trimap_gt.jpg',(self.gt_mask[0]*255).permute(1,2,0).detach().cpu().numpy())
            # click_map = np.zeros([448,448,3])
            # click_map[:,:,2] += coord_features[0,3].detach().cpu().numpy()
            # click_map[:,:,1] += coord_features[0,2].detach().cpu().numpy()
            # click_map[:,:,0] += coord_features[0,1].detach().cpu().numpy()
            # click_map[np.sum(click_map,axis = 2) == 0] += 1
            # cv2.imwrite('vis/click.jpg',click_map*255)

            # trimap_pred_mask = (trimap == torch.max(trimap, dim=1)[0].unsqueeze(1)).type(torch.uint8)
            # trimap_pred_255 = trimap_pred_mask[0,0] * 0 + trimap_pred_mask[0,1] * 128 + trimap_pred_mask[0,2] * 255
            # cv2.imwrite('vis/trimap_pred.jpg',trimap_pred_255.detach().cpu().numpy())

            # trimap_gt_mask = self.gt_mask[0].detach().cpu().numpy()
            # trimap_gt_mask_255 = trimap_gt_mask[0] * 0 + trimap_gt_mask[1] * 128 + trimap_gt_mask[2] * 255
            # cv2.imwrite('vis/trimap_gt.jpg',trimap_gt_mask_255)




        matting = 0
        save_trimap = 0



        if matting or save_trimap:
            
            from inference import single_inference, generator_tensor_dict
            
            image_ori = cv2.imread(self.image_path[0])

            trimap = nn.functional.interpolate(outputs['instances'], size=image_ori.shape[0:2],
                                                            mode='bilinear', align_corners=True)
            
            trimap_mask = (trimap == torch.max(trimap,dim=1)[0]).type(torch.uint8)

            trimap_255 = trimap_mask[0,0] * 0 + trimap_mask[0,1] * 128 + trimap_mask[0,2] * 255


            if save_trimap:
                root = '/mnt/data/datasets/trimaps_huge_Q1_0.1/trimaps_ppm100/trimaps_click_'               
                click_index = str(self.click_index + 1)
                trimap_click_dir_path = root + click_index
                if not os.path.exists(trimap_click_dir_path):
                    os.mkdir(trimap_click_dir_path)
                trimap_name = self.image_path[0].split('/')[-1]
                save_dir = os.path.join(trimap_click_dir_path, trimap_name)
                cv2.imwrite(save_dir, trimap_255.cpu().numpy())

            # image_dict = {'image':image_ori, 'trimap':trimap_mask, 'alpha_shape':image_ori.shape[-2:]}
            if matting:

                image_dict = generator_tensor_dict(self.image_path[0], trimap_255.cpu().numpy())
                cv2.imwrite('vis/image.jpg',image_ori)             
                alpha_pred = single_inference(self.matting, image_dict)
                cv2.imwrite('vis/alpha_pred.jpg',alpha_pred)


                plt.figure(figsize=(7,5),dpi=450)

                plt.subplot(2,2,1)
                plt.title('trimap_hr')
                plt.imshow(trimap_255.cpu().numpy())

                plt.subplot(2,2,2)
                plt.title('alpha_pred')
                plt.imshow(alpha_pred)

                plt.subplot(2,2,3)
                plt.title('trimap_gt')
                plt.imshow(self.trimap_data.squeeze().cpu().numpy())

                plt.subplot(2,2,4)
                plt.title('alpha_gt')
                plt.imshow(self.alpha_data.squeeze().cpu().numpy())

                plt.suptitle('click:' + str(self.click_index + 1) )
                plt.savefig('vis/alpha')

                root = '/home/tianhuawei/zcy/click2trimap/alpha/alpha_click_'               
                click_index = str(self.click_index + 1)
                alpha_click_dir_path = root + click_index
                if not os.path.exists(alpha_click_dir_path):
                    os.mkdir(alpha_click_dir_path)
                alpha_name = self.image_path[0].split('/')[-1]
                save_dir = os.path.join(alpha_click_dir_path, alpha_name)
                cv2.imwrite(save_dir, alpha_pred)


        return outputs

    def prepare_input(self, image):
        # prev_mask = None
        # if self.with_prev_mask:
        #     prev_mask = image[:, 3:, :, :]
        #     image = image[:, :3, :, :]

        image = self.normalization(image)
        return image

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, points,rectangle):
        coord_features = self.dist_maps(image, points)
        # coord_features[:,0:1] = coord_features[:,0:1] * 0.5 + 0.5
        # coord_features[:,1:2] = (1 - coord_features[:,1:2]) * 0.5 

        # plt.figure('fore')
        # plt.imshow(coord_features[0,0].cpu().numpy())
        # plt.savefig('vis/fore.png')
        # plt.figure('back')
        # plt.imshow(coord_features[0,1].cpu().numpy())
        # plt.savefig('vis/back.png')


        # if prev_mask is not None:
        #     coord_features = torch.cat((prev_mask, coord_features), dim=1)

        coord_features = torch.cat((coord_features, rectangle), dim=1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
