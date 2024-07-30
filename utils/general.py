#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import random
from copy import deepcopy

import numpy as np
import torch

import lightning as L
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm


@torch.no_grad()
def get_representations(args, net, data_loader, t, get_pair=False   ):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        args: arguments
        net: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        tuple of data with the first one being image representations. Other data depends on the dataset.
    """
    net.eval()
    gathered_data = [[], [], [], [], [], [], [], [], [], []]
    strt_idx = 0 if not get_pair else 1

    for data in tqdm(data_loader):
        rep = net(t(data[0][0]))
        gathered_data[0].append(rep)
        if get_pair:
            gathered_data[1].append(net(t(data[0][1])))
        for i in range(1, len(data)):
            gathered_data[i+strt_idx].append(data[i])

        if "supervised" in args.modules:
            gathered_data[len(data)+strt_idx].append(net.sup_projector(rep))

        if args.name == "test3":
            break
    tensor_data = [torch.cat(data, dim=0) for data in gathered_data if data]
    return tensor_data


def prepare_device(args):
    np.set_printoptions(linewidth=np.nan, precision=2)
    torch.set_printoptions(precision=3, linewidth=150)
    torch.set_float32_matmul_precision('medium')


    fabric = L.Fabric(accelerator=args.device, devices=args.num_devices, strategy=DDPStrategy(broadcast_buffers=False), num_nodes=1)
    if args.seed != -1:
        fabric.seed_everything(args.seed)

    fabric.launch()
    return fabric



def run_forward(args, x, net):
    x1, x2 = x.split(x.shape[0] // 2)
    rep1 = net(x1)
    rep2 = net(x2)
    return torch.cat((rep1, rep2), dim=0)



def init_target_net(net, net_target):
    # initialize target network
    for param_online, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(param_online.data)  # initialize
        param_target.requires_grad = False  # not update by gradient


def get_dataset_kwargs(d):
    d_new = deepcopy(d)
    del d_new["class"]
    return d_new


def is_target_needed(args):
    if args.main_loss in ['BYOL'] or "byol" in args.modules:
        return True
    return False


def update_target_net(net, net_target, tau):
    for param, target_param in zip(net.parameters(), net_target.parameters()):
        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)


def save_model(fabric, net, log_dir, epoch, optimizer=None, scheduler=None):
    path = os.path.join(log_dir, 'models')
    if fabric.global_rank == 0:
        if not os.path.exists(path):
            os.mkdir(path)
    obj = {}
    obj["model"] = net
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()
    fabric.save(os.path.join(path, f'epoch_{epoch}.pt'), obj)


def load_model(fabric, net, args, optimizer=None, scheduler=None,strict=True):
    checkpoint = fabric.load(os.path.join(args.path_load_model, "epoch_" + str(args.epoch_load_model)) + ".pt")
    net.load_state_dict(checkpoint["model"],strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if args.cosine_decay and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


#@torch.no_grad()
def normalize(x):
    return (x - x.mean(0, keepdim=True)) / (x.var(dim=0, keepdim=True) + 1e-5).sqrt()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise Exception('Boolean value expected.')

def str2table(v):
    return v.split(',')

