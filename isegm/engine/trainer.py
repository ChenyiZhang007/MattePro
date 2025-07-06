import os
import random
import logging
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, loss_cfg,
                 trainset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 ):
        self.cfg = cfg
        # self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.train_data = trainset

        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)


        model = self._load_weights(model)



        # self.is_master = False
        if self.is_master:
            logger.info(model)
            # logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            
            if validation:
                self.validation(epoch)
            else:
                self.training(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):

            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward_its(batch_data)


            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                # if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                #     self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):

            # if '4871683890_480b80' not in batch_data['image_name'][0]:
            #     continue
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward_its(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    
    
    def batch_forward_its(self, batch_data, validation=False):
        losses_logging = dict()

        
        with torch.set_grad_enabled(not validation):
            # batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            # print(self.device)

            batch_data['trimap'] = batch_data['trimap'].to(self.device)
            image, gt_mask = batch_data['image'].to(self.device), batch_data['trimap']
            # points = batch_data['points'].to(self.device)
            alpha = batch_data['alpha'].to(self.device)


            self.net.gt_mask = gt_mask
            self.net.alpha = alpha
            self.net.validation = False

            self.net.image_name = batch_data['image_name']

            num_iters = random.randint(0, 8)
            


            rectangle = batch_data['rectangle']
            # prompt = (batch_data['click'], batch_data['bbox'])
            bbox = batch_data['bbox'].to(self.device)
            click = batch_data['click'].to(self.device)
            features, interm_features, pred_trimap = self.net.forward_image_encoder(image, (click, bbox))

            with torch.no_grad():
                for click_indx in range(num_iters):


                    

                    ## prompt encoder and mask decoder

                    interm_features, sam2_logits, trimap_logits, pred_trimap = self.net.forward_others(image, (click, bbox), features, interm_features) + (pred_trimap, )
                    trimap_logits = F.softmax(trimap_logits, dim=1)

                    # trimap_logits= self.net((image,prompt))
                    trimap_logits = F.interpolate(trimap_logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
                    next_click = get_next_click(trimap_logits,gt_mask)
                    click[:,-(click_indx+1),:] = next_click






            # features, interm_features, pred_trimap = self.net.forward_samhq_image_encoder(image, (click, bbox))
            interm_features, sam2_logits, trimap_logits, pred_trimap = self.net.forward_others(image, (click, bbox), features, interm_features) + (pred_trimap, )
            trimap_logits = F.softmax(trimap_logits, dim=1)
            trimap_logits = F.interpolate(trimap_logits, size=image.shape[-2:], mode='bilinear', align_corners=False)


            output = trimap_logits


            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (trimap_logits, gt_mask))

            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                 lambda: (trimap_logits, gt_mask))

        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['image']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0,0:1], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0
    
    

def load_weights(model, path_to_weights, use_rules=False):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    # new_state_dict = torch.load(path_to_weights, map_location='cpu')
    if use_rules:
        del_list = []
        print(len(new_state_dict))
        for k, _ in new_state_dict.items():
            if 'backbone' not in k or 'neck' not in k:
                del_list.append(k)
        for k in del_list:
            del new_state_dict[k]
        print(len(new_state_dict))

    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict, strict=False)

def ensure_three_channels(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def logits2trimap(trimap_logits):
    logits_argmax = torch.argmax(trimap_logits,dim=1)
    # trimap_tensor = torch.where(logits_argmax == 0, torch.tensor(0), torch.where(logits_argmax == 1, torch.tensor(128), torch.tensor(255)))
    trimap = logits_argmax.cpu().numpy()

    return trimap



def get_next_click(trimap_logits,gt_mask):

    pred_batch = logits2trimap(trimap_logits)
    trimap_batch = logits2trimap(gt_mask)
    next_click_batch = []

    for pred,trimap in zip(pred_batch,trimap_batch):

        fore_mask = pred == 2
        uk_mask = pred == 1
        back_mask = pred == 0

        fore_gt = trimap == 2
        uk_gt = trimap == 1
        back_gt = trimap == 0

        ######### False Positive        
        foreground_fn = np.logical_and(fore_gt, np.logical_not(fore_mask))
        foreground_fn = np.pad(foreground_fn, ((1, 1), (1, 1)), 'constant').astype(np.uint8)

        uk_fn = np.logical_and(uk_gt, np.logical_not(uk_mask))
        uk_fn = np.pad(uk_fn, ((1, 1), (1, 1)), 'constant').astype(np.uint8)

        background_fn = np.logical_and(back_gt, np.logical_not(back_mask))
        background_fn = np.pad(background_fn, ((1, 1), (1, 1)), 'constant').astype(np.uint8)

        foreground_fn_mask_dt = cv2.distanceTransform(foreground_fn, cv2.DIST_L2, 5)[1:-1, 1:-1]
        uk_fn_mask_dt = cv2.distanceTransform(uk_fn, cv2.DIST_L2, 5)[1:-1, 1:-1]
        background_fn_mask_dt = cv2.distanceTransform(background_fn, cv2.DIST_L2, 5)[1:-1, 1:-1]

        tri_fn_list = [foreground_fn_mask_dt, uk_fn_mask_dt, background_fn_mask_dt] ##三种fn map

        foreground_fn_max_dist = np.max(foreground_fn_mask_dt)
        uk_fn_max_dist = np.max(uk_fn_mask_dt)
        background_fn_max_dist = np.max(background_fn_mask_dt)



        tri_dist = [foreground_fn_max_dist, uk_fn_max_dist, background_fn_max_dist]
        dist_max = max(tri_dist)##最大距离

        
        flag = tri_dist.index(dist_max)

        fn_map = tri_fn_list[flag]

        indices = np.argwhere(fn_map == dist_max)

        if len(indices) > 0:
            coords = indices[0]

            if flag==0: #前景
                next_click = [coords[1],coords[0],1]

            if flag==1: #unknown景
                next_click = [coords[1],coords[0],4]

            if flag==2: #背景
                next_click = [coords[1],coords[0],0]

        next_click_batch.append(next_click)

    next_click_batch = torch.tensor(next_click_batch, dtype=torch.float64, device=trimap_logits.device)

    return next_click_batch

def vis_click(click, trimap):
    # 创建空白画布
    canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    trimap = (trimap.permute(1,2,0)*255).cpu().numpy().astype(np.uint8)

    data = click.cpu().numpy()

    # 绘制有效点
    for point in data:
        x, y, flag = point
        if flag == -1:
            continue
        if flag == 1:
            color = (0, 0, 255)  # Red
        elif flag == 0:
            color = (255, 0, 0)  # Blue
        elif flag == 4:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 0)  # Black for undefined
        cv2.circle(canvas, (int(x), int(y)), 5, color, -1)

    # 显示图像
    combined = np.hstack((canvas, trimap))

    cv2.imwrite('click.png', combined)

def compute_single_data(pred, alpha, trimap):

    pred_mask = pred.detach().cpu().numpy()
    trimap = trimap.detach().cpu().numpy()

    fore_mask = pred_mask.argmax(axis=1) == 2
    uk_mask = pred_mask.argmax(axis=1) == 1
    back_mask = pred_mask.argmax(axis=1) == 0

    # alpha_mask = alpha.cpu().numpy()
    # uk_alpha = np.logical_and(alpha_mask < 1 , alpha_mask > 0).squeeze(1)
    fore_gt = trimap[:,2]
    uk_gt = trimap[:,1]
    back_gt = trimap[:,0]

    # foreground_fn = get_fn()

    # union_mask = fore_mask + uk_mask

    # pred = union_mask.cpu().numpy()
    # gt = gt.cpu().numpy()[:, 0, :, :] > 0 ## alpha mask

    foreground_fp =  np.logical_and(np.logical_not(fore_gt), fore_mask)
    # foreground_fp = np.pad(foreground_fp, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    uk_fn = np.logical_and(uk_gt, np.logical_not(uk_mask))
    # uk_fn = np.pad(uk_fn, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    background_fp = np.logical_and(np.logical_not(back_gt), back_mask)
    # background_fp = np.pad(background_fp, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)

    plt.subplot(1,3,1)
    plt.title('foreground_fp')
    plt.imshow(foreground_fp[0])

    plt.subplot(1,3,2)
    plt.title('uk_fn')
    plt.imshow(uk_fn[0])

    plt.subplot(1,3,3)
    plt.title('background_fp')
    plt.imshow(background_fp[0])

    plt.savefig('vis/tri_single')

    single_error = torch.tensor(np.concatenate((foreground_fp,uk_fn,background_fp),axis=0),device='cuda:0')
    return single_error


