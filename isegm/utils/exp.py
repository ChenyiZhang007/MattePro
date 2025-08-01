import os
import sys
import shutil
import pprint
from pathlib import Path
from datetime import datetime
 
import yaml
import torch
from easydict import EasyDict as edict

from .log import logger, add_logging
from .distributed import synchronize, get_world_size


def init_experiment(args, model_name):
    model_path = Path(args.model_path)
    ftree = get_model_family_tree(model_path, model_name=model_name)

    if ftree is None:
        print('Models can only be located in the "models" directory in the root of the repository')
        sys.exit(1)

    cfg = load_config(model_path)
    update_config(cfg, args)

    cfg.distributed = args.distributed
    cfg.local_rank = args.local_rank
    

    # if cfg.distributed:
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #     if args.workers > 0:
    #         torch.multiprocessing.set_start_method('forkserver', force=True)

    experiments_path = Path(cfg.EXPS_PATH)
    exp_parent_path = experiments_path / '/'.join(ftree)
    exp_parent_path.mkdir(parents=True, exist_ok=True)

    if cfg.resume_exp:
        exp_path = find_resume_exp(exp_parent_path, cfg.resume_exp)
    else:
        last_exp_indx = find_last_exp_indx(exp_parent_path)
        exp_name = f'{last_exp_indx:03d}'
        if cfg.exp_name:
            exp_name += '_' + cfg.exp_name
        exp_path = exp_parent_path / exp_name
        synchronize()
        if cfg.local_rank == 0:
            exp_path.mkdir(parents=True)

    cfg.EXP_PATH = exp_path
    cfg.CHECKPOINTS_PATH = exp_path / 'checkpoints'
    cfg.VIS_PATH = exp_path / 'vis'
    cfg.LOGS_PATH = exp_path / 'logs'

    if cfg.local_rank == 0:
        cfg.LOGS_PATH.mkdir(exist_ok=True)
        cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)
        cfg.VIS_PATH.mkdir(exist_ok=True)

        dst_script_path = exp_path / (model_path.stem + datetime.strftime(datetime.today(), '_%Y-%m-%d-%H-%M-%S.py'))
        if args.temp_model_path:
            shutil.copy(args.temp_model_path, dst_script_path)
            os.remove(args.temp_model_path)
        else:
            shutil.copy(model_path, dst_script_path)

    synchronize()

    if cfg.gpus != '':
        gpu_ids = [int(id) for id in cfg.gpus.split(',')]
    else:
        gpu_ids = list(range(max(cfg.ngpus, get_world_size())))
        cfg.gpus = ','.join([str(id) for id in gpu_ids])

    cfg.gpu_ids = gpu_ids
    cfg.ngpus = len(gpu_ids)
    cfg.multi_gpu = cfg.ngpus > 1

    if cfg.distributed:
        cfg.device = torch.device('cuda')
        cfg.gpu_ids = [cfg.gpu_ids[cfg.local_rank]]
        torch.cuda.set_device(cfg.gpu_ids[0])
    else:
        if cfg.multi_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
            ngpus = torch.cuda.device_count()
            assert ngpus >= cfg.ngpus
        cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')

    if cfg.local_rank == 0:
        add_logging(cfg.LOGS_PATH, prefix='train_')
        logger.info(f'Number of GPUs: {cfg.ngpus}')
        if cfg.distributed:
            logger.info(f'Multi-Process Multi-GPU Distributed Training')

        logger.info('Run experiment with config:')
        logger.info(pprint.pformat(cfg, indent=4))

    return cfg


def get_model_family_tree(model_path, terminate_name='models', model_name=None):
    if model_name is None:
        model_name = model_path.stem
    family_tree = [model_name]
    for x in model_path.parents:
        if x.stem == terminate_name:
            break
        family_tree.append(x.stem)
    else:
        return None

    return family_tree[::-1]


def find_last_exp_indx(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx


def find_resume_exp(exp_parent_path, exp_pattern):
    candidates = sorted(exp_parent_path.glob(f'{exp_pattern}*'))
    if len(candidates) == 0:
        print(f'No experiments could be found that satisfies the pattern = "*{exp_pattern}"')
        sys.exit(1)
    elif len(candidates) > 1:
        print('More than one experiment found:')
        for x in candidates:
            print(x)
        sys.exit(1)
    else:
        exp_path = candidates[0]
        print(f'Continue with experiment "{exp_path}"')

    return exp_path


def update_config(cfg, args):
    for param_name, value in vars(args).items():
        if param_name.lower() in cfg or param_name.upper() in cfg:
            continue
        cfg[param_name] = value


def load_config(model_path):
    model_name = model_path.stem
    config_path = model_path.parent / (model_name + '.yml')

    if config_path.exists():
        cfg = load_config_file(config_path)
    else:
        cfg = dict()

    cwd = Path.cwd()
    config_parent = config_path.parent.absolute()
    while len(config_parent.parents) > 0:
        config_path = config_parent / 'config.yml'

        if config_path.exists():
            local_config = load_config_file(config_path, model_name=model_name)
            cfg.update({k: v for k, v in local_config.items() if k not in cfg})

        if config_parent.absolute() == cwd:
            break
        config_parent = config_parent.parent

    return edict(cfg)


def load_config_file(config_path, model_name=None, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'SUBCONFIGS' in cfg:
        if model_name is not None and model_name in cfg['SUBCONFIGS']:
            cfg.update(cfg['SUBCONFIGS'][model_name])
        del cfg['SUBCONFIGS']

    return edict(cfg) if return_edict else cfg
