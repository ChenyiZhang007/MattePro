from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader, train_dataset


train.max_iter = int(282700 / 2 / 4 * 10)
train.checkpointer.period = int(282700 / 2 / 4 * 0.5)

model.backbone.use_rel_pos = True
model.backbone.topk = 0.25
model.backbone.window_block_indexes=[0,1,3,4,6,7,9,10,] # 2, 5, 8 11 for global attention
model.backbone.multi_score = True
model.distill = True

model.teacher_backbone.window_block_indexes=[0,1,3,4,6,7,9,10,] # 2, 5, 8 11 for global attention

optimizer.lr=5e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones=[int(282700 / 2 / 4 * 3), int(282700 / 2 / 4 * 9)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = '/opt/data/private/lyh/MAM/MEMatte_mix_data/pretrained/ViTMatte_S_MixData_1024_0053003_with_teacher.pth'
train.output_dir = './output_of_train/MixData_ViTMatte_S_topk0.25_1024_distill'


train_dataset.crop_size = 1024
model.backbone.img_size = 1024
model.teacher_backbone.img_size = 1024
dataloader.train.batch_size=2
dataloader.train.num_workers=16
train_dataset.data.alpha_dir = "/opt/data/private/lyh/Datasets/mix_data_com1k_d646_am2k/Train/alpha"
train_dataset.data.alpha_exts = [".jpg", ".png"]
train_dataset.data.fg_dir = "/opt/data/private/lyh/Datasets/mix_data_com1k_d646_am2k/Train/fg"
train_dataset.data.fg_exts = [".jpg", ".png"]
train_dataset.data.root = "/opt/data/private/lyh/Datasets/mix_data_com1k_d646_am2k"





"""
1. gumbel_softmax
2. 1 router
3. replace
"""
