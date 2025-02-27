# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import wandb
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from tqdm import tqdm
from utils import utilis_func as uf

# from models import DiT_models
# from diffusion import create_diffusion
# from diffusers.models import AutoencoderKL

# QM9
from mol_data.get_datasets import get_data_loader
from utils.parse_args import parse_args
from models.get_models import get_optim, get_goat
from utils.datasets_config import get_dataset_info
from train_epoch import train_epoch, analyze_and_save, test
from qm9.utils import compute_mean_mad

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dtype = torch.float32
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        if args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if args.online else 'offline'
        kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'molecule', 'config': args,
                  'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        model_string_name = args.exp_name
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Load dataset {args.dataset}")
    else:
        logger = create_logger(None)

    # Retrieve QM9 dataloaders
    dataloaders, samplers = get_data_loader(args, dist, rank)
    property_norms = None
    args.context_node_nf = 0
    args.property = 'none'
    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        args.property = args.conditioning[0]
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    model, nodes_dist, prop_dist = get_goat(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    model = model.to(device)
    optim = get_optim(args, model)
    gradnorm_queue = uf.Queue()
    gradnorm_queue.add(3000)

    model_ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    ema = uf.EMA(args.ema_decay)

    begin_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model_check_point = checkpoint["model"]
        model_ema_cp = checkpoint["model_ema"]
        optim_cp = checkpoint["opt"]
        model.load_state_dict(model_check_point)
        model_ema.load_state_dict(model_ema_cp)
        optim.load_state_dict(optim_cp)
        begin_epoch = checkpoint["args"].current_epoch

    if args.reflow_model_path is not None:
        if rank == 0:
            logger.info(f"Load from reflow {args.reflow_model_path}")
        checkpoint = torch.load(args.reflow_model_path, map_location=lambda storage, loc: storage)
        model_check_point = checkpoint["model"]
        model_ema_cp = checkpoint["model_ema"]
        optim_cp = checkpoint["opt"]
        model.load_state_dict(model_check_point)
        model_ema.load_state_dict(model_ema_cp)
        optim.load_state_dict(optim_cp)

    requires_grad(model_ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    # Variables for monitoring/logging purposes:
    best_nll_val = 1e8
    best_nll_test = 1e8
    best_valid = 0
    if rank == 0:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(begin_epoch, args.epochs):
        samplers['train'].set_epoch(epoch)
        if rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        start_epoch = time()
        loss = train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, rank=rank)
        dist.barrier()
        if rank == 0:
            logger.info(f"Epoch {epoch}, loss = {loss:.2f}, took {time() - start_epoch:.1f} seconds.")
            wandb.log({'Train Loss': loss}, commit=True)
        if epoch % args.test_epochs == 0:
            if not args.break_train_epoch and epoch % args.sample_epochs == 0 and args.probabilistic_model != 'vae':
                gen_time = time()
                metrics = analyze_and_save(args=args, epoch=epoch, model_sample=model_ema,
                                                             nodes_dist=nodes_dist, property_norms=property_norms,
                                                             dataset_info=dataset_info, device=device,
                                                             prop_dist=prop_dist, n_samples=args.n_stability_samples)
                metric_list = [torch.zeros(metrics.size(0)).to(device) for _ in range(dist.get_world_size())]
                dist.barrier()
                dist.all_gather(metric_list, metrics)
                if rank == 0:
                    # Calculate the mean along the specified dimension (0 for one-dimensional tensors)
                    metrics = torch.mean(torch.stack(metric_list), dim=0) * 100
                    logger.info(f'AS\tMS\tValidity\tV & U\tNovelty ({args.n_stability_samples}/{time() - gen_time:.1f}s)')
                    logger.info('%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (
                    metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5] / 100))
                    m_text = ['atom stability', 'mol stability', 'validity', 'V & U', 'Novelty', args.property]
                    for text, metric in zip(m_text, metrics):
                        wandb.log({text: metric / 100.0}, commit=True)
                    valid = metrics[1]
                    if valid > best_valid:
                        best_valid = valid
                        if args.save_model:
                            args.current_epoch = epoch + 1
                            checkpoint = {
                                "model": model.module.state_dict(),
                                "model_ema": model_ema.state_dict(),
                                "opt": optim.state_dict(),
                                "args": args
                            }
                            checkpoint_path = f"{checkpoint_dir}/best_valid.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved best valid checkpoint to {checkpoint_path}")
            dist.barrier()
            samplers['valid'].set_epoch(epoch)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema,
                           partition='valid', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, rank=rank)
            samplers['test'].set_epoch(epoch)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, rank=rank)
            dist.barrier()
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "model_ema": model_ema.state_dict(),
                            "opt": optim.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/smg.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        if args.probabilistic_model == 'vae':
                            checkpoint_path = args.vae_path
                            torch.save(model_ema.state_dict(), checkpoint_path)
            if rank == 0:
                logger.info('E: %d Val loss: %.4f \t Test loss:  %.4f' % (epoch, nll_val, nll_test))
                logger.info('E: %d Best val loss: %.4f \t Best test loss:  %.4f' % (epoch, best_nll_val, best_nll_test))
                wandb.log({"Val loss ": nll_val}, commit=True)
                wandb.log({"Test loss ": nll_test}, commit=True)
                wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)
            dist.barrier()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
