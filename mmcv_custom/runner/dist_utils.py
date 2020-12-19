import logging
import os
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info
from mmdet.utils import gpu_indices, ompi_size, ompi_rank, ompi_local_size, set_environment_variables_for_nccl_backend, get_philly_master_ip, get_aml_master_ip


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'infimpi':
        set_environment_variables_for_nccl_backend(ompi_size() == ompi_local_size())
        _init_dist_infimpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    os.environ['MASTER_ADDR'] = '127.0.0.10'
    os.environ['MASTER_PORT'] = '29999'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    # TODO: find a better way to support both philly and AML enviroment
    print(get_aml_master_ip())
    if get_aml_master_ip() is not None:
        dist_url = 'tcp://' + get_aml_master_ip() + ':23456'
    else:
        dist_url = 'tcp://' + get_philly_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is {}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))


def _init_dist_infimpi(backend, **kwargs):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    # TODO: find a better way to support both philly and AML enviroment
    if get_aml_master_ip() is not None:
        # master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')
        # os.environ['MASTER_ADDR'] = master_node_params[0]
        # os.environ['MASTER_PORT'] = master_node_params[1]
        # dist_url = 'tcp://' + get_aml_master_ip() + ':23456'
        dist_url = 'tcp://' + os.environ['AZ_BATCH_MASTER_NODE']
    else:
        dist_url = 'tcp://' + get_philly_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is{}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger