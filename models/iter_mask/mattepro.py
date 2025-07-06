import os
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

from dataloader.data_generator import DataGenerator
from dataloader.image_file import ImageFileTrain, ImageFileTest
from dataloader.prefetcher import Prefetcher

from torch.utils.data import DataLoader
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from torch.utils.checkpoint import checkpoint

MODEL_NAME = 'mattepro_sam2'


def wrap_with_gradient_checkpointing(module):
    """
    Wraps the forward method of a module with torch.utils.checkpoint to enable gradient checkpointing.
    """
    original_forward = module.forward

    def checkpointed_forward(*inputs):
        # Use torch.utils.checkpoint to wrap the original forward method
        return checkpoint(original_forward, *inputs)

    module.forward = checkpointed_forward
    print(f"Wrapped module {module} with gradient checkpointing.")

def wrap_load_sam_model(model,state_dict):
    # 修改SAM2 ckpt的key
    prefix = 'sam_model.'
    keys_to_update = list(state_dict.keys())  # 将 keys 转为列表以避免遍历过程中修改问题
    for key in keys_to_update:
        new_key = prefix+key
        state_dict[new_key] = state_dict.pop(key)

    load_status = model.load_state_dict(state_dict, strict=False)
    return model



def load_model(model_type='SAM2'):
    config_path = './configs/MattePro_{}.py'.format(model_type)
    cfg = LazyConfig.load(config_path)

    if hasattr(cfg.model.sam_model, 'ckpt_path'):
        cfg.model.sam_model.ckpt_path = None
    else:
        cfg.model.sam_model.checkpoint = None
    model = instantiate(cfg.model)

    sam2_checkpoint_path = './pretrained/sam2.1_hiera_large.pt'
    sam2_checkpoint = torch.load(sam2_checkpoint_path)
    state_dict = sam2_checkpoint["model"]
    model = wrap_load_sam_model(model,state_dict)



    model.lora_rank = 2
    model.lora_alpha = model.lora_rank
 

    model.sam_model.image_encoder.trunk.enable_gradient_checkpointing()

    if model.lora_rank is not None:
        
        model.init_lora()
        

    # 将配置文件绑定到模型
    model._config = cfg.model


    return model



def get_dataset(cfg):

    
    DUTS_img_dir = os.path.join(cfg.DUTS_PATH, 'DUTS-TR-Image')
    DUTS_mask_dir = os.path.join(cfg.DUTS_PATH, 'DUTS-TR-Mask')

    DIS_img_dir = os.path.join(cfg.DIS_PATH, 'im')
    DIS_mask_dir = os.path.join(cfg.DIS_PATH, 'gt')

    DIM_alpha_dir = os.path.join(cfg.DIM_PATH,'alpha')
    DIM_fg_dir = os.path.join(cfg.DIM_PATH,'fg')

    D646_alpha_dir = os.path.join(cfg.D646_PATH,'GT')
    D646_fg_dir = os.path.join(cfg.D646_PATH,'FG')

    AM2K_fg_dir = os.path.join(cfg.AM2K_PATH,'original')
    AM2K_alpha_dir = os.path.join(cfg.AM2K_PATH,'mask')

    T460_fg_dir = os.path.join(cfg.T460_PATH,'fg')
    T460_alpha_dir = os.path.join(cfg.T460_PATH,'alpha')
    
    P3M_fg_dir = os.path.join(cfg.P3M_PATH,'blurred_image')
    P3M_alpha_dir = os.path.join(cfg.P3M_PATH,'mask')
    

    bg_dir = cfg.BG_PATH
   
    train_image_file = ImageFileTrain(
                                    dim_alpha_dir=DIM_alpha_dir,
                                    dim_fg_dir=DIM_fg_dir,

                                    d646_alpha_dir=D646_alpha_dir,
                                    d646_fg_dir=D646_fg_dir,

                                    duts_img_dir=DUTS_img_dir,
                                    duts_mask_dir=DUTS_mask_dir,

                                    dis_img_dir=DIS_img_dir,
                                    dis_mask_dir=DIS_mask_dir,

                                    am2k_fg_dir=AM2K_fg_dir,
                                    am2k_alpha_dir=AM2K_alpha_dir,  

                                    t460_fg_dir=T460_fg_dir,
                                    t460_alpha_dir=T460_alpha_dir,

                                    p3m_fg_dir=P3M_fg_dir,
                                    p3m_alpha_dir=P3M_alpha_dir,
                            
                                    bg_dir=bg_dir)
    
    train_dataset = DataGenerator(train_image_file, phase='train')

    return train_dataset
 

def train(model, cfg, model_cfg, train_dataloader):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0





    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8,
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[15, 20], gamma=0.1)
    trainer = ISTrainer(model, cfg, loss_cfg,
                        train_dataloader, 
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=1,
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=25, validation=False)
