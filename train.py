import os
import argparse
import importlib.util
import numpy as np
import torch
from isegm.utils.exp import init_experiment
from torch.utils.data import DataLoader
from dataloader.prefetcher import Prefetcher
import warnings
from isegm.utils.exp_imports.default import *

def main_worker(args):
    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.distributed = False
    args.local_rank = 0

    cfg = init_experiment(args, model_base_name)

    model_cfg = edict()
    model_cfg.crop_size = (512, 512)
    model_cfg.num_max_points = 24

    # _, model_cfg = model_script.init_model(cfg)

    torch.backends.cudnn.benchmark = True
    

    # 初始化模型
    device = torch.device("cuda:"+args.gpus if torch.cuda.is_available() else "cpu")
    model = model_script.load_model().to(device)
    cfg.device = device

    # 切分数据
    train_dataset = model_script.get_dataset(cfg)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.workers,
                                  pin_memory=True,
                                  drop_last=True)
    train_dataloader = Prefetcher(train_dataloader)



    warnings.filterwarnings("ignore", message="Expected to have finished reduction in the prior iteration.*")

    # 调用训练逻辑，传递模型和数据
    model_script.train(model=model, cfg=cfg, model_cfg=model_cfg, train_dataloader=train_dataloader)


def parse_args():

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='models/iter_mask/mattepro.py', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='rank2',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4, metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default=None,
                        help='The prefix of the name of the checkpoint to be loaded.')
    
    parser.add_argument('--gpus', type=str, default='3', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')
    
    parser.add_argument('--ngpus', type=str, default=1, required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument('--layerwise-decay', action='store_true',
                        help='layer wise decay for transformer blocks.')

    parser.add_argument('--upsample', type=str, default='x1',
                        help='upsample the output.')

    parser.add_argument('--random-split', action='store_true',
                        help='random split the patch instead of window split.')

    return parser.parse_args()


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    args = parse_args()
    # 启动单卡训练
    main_worker(args)
