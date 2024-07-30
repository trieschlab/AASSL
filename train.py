#!/usr/bin/python
# _____________________________________________________________________________

import datetime
import json
# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import time
from argparse import Namespace

from tqdm import tqdm
import torch
import config
from utils.augmentations import  get_transformations
from utils.constants import DATASETS
from utils.finetuning import finetune
from utils.finetuning_all import finetune_all
# configuration module
# -----


# custom libraries
# -----
from utils.general import init_target_net, prepare_device, save_model, is_target_needed, update_target_net, load_model, \
    run_forward

from utils.getters import get_datasets, get_train_iterator, get_scheduler, get_network, get_modules, get_networks, \
    get_arguments, get_optimizer, apply_transform


# custom function
# -----


def train():

    # Initialization

    run_name = f'{datetime.datetime.now().strftime("%d-%m-%y_%H-%M")}_{args.name}_{args.seed}'
    fabric = prepare_device(args)
    dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test = get_datasets(args, run_name, fabric)
    train_t, val_t = get_transformations(args, crop_size=DATASETS[args.dataset]['img_size'])
    net, net_target, method_modules = get_networks(args, fabric, dataset_train)

    optimizer = get_optimizer(args, net)
    scheduler = get_scheduler(args, optimizer, len(dataloader_train)*args.n_epochs)
    net, optimizer= fabric.setup(net, optimizer)

    dataloader_train, dataloader_train_eval, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_train_eval, dataloader_test, move_to_device=True)

    if args.path_load_model:
        load_model(fabric, net,args, optimizer=optimizer, scheduler=scheduler)

    epoch_loop = tqdm(range(args.n_epochs), ncols=80)

    if args.name != "test":
        dataset_test.eval(net, dataloader_train_eval, dataloader_test, modules=method_modules, tf= train_t, tv=val_t)

    # Training
    for epoch in epoch_loop:
        epoch_loop.set_description(f"Method: {run_name.split('~')[0]}, Epoch: {epoch + 1}")
        training_loop = get_train_iterator(args, dataloader_train)
        net.train()
        for i, data in enumerate(training_loop):
            optimizer.zero_grad()
            (x_pair1, x_pair, a), labels = data[0], data[1]


            with torch.no_grad():
                x = apply_transform(args, train_t, x_pair1, x_pair)
            rep = run_forward(args, x, net)
            rep_target=None
            if is_target_needed(args):
                with torch.no_grad():
                    rep_target = run_forward(args, x, net_target)

            loss = 0
            for m in method_modules:
                loss = loss+ m.apply(net, net_target=net_target, action=a, rep=rep, rep_target=rep_target, data=data, epoch=epoch)

            fabric.backward(loss)
            optimizer.step()
            scheduler.step()

            if is_target_needed(args):
                update_target_net(net, net_target, args.tau)
            training_loop.set_description(f'Loss: {loss.item():>8.4f}')


        if (epoch+1) % args.test_every == 0:
            dataset_test.eval(net, dataloader_train_eval, dataloader_test, epoch=epoch +1, modules=method_modules, tf= train_t, tv=val_t)
        if args.save_model and (epoch+1) % args.save_every == 0:
            save_model(fabric, net, dataset_test.get_log_dir(), epoch, optimizer=optimizer, scheduler=scheduler)
            fabric.barrier()

    if args.finetune:
        finetune_all(args, dataloader_train, dataset_train, dataloader_test, net, os.path.join(args.log_dir, run_name), fabric, train_t, val_t)


def end_finetune(all=False):
    if args.path_load_model != "":
        with open(os.path.join(args.path_load_model,'../config.json'), 'r') as f:
            config = json.load(f)
            config = config[next(iter(config))]
            new_args = Namespace(**config)
            args.seed = new_args.seed

    fabric = prepare_device(args)
    run_name = f'{datetime.datetime.now().strftime("%d-%m-%y_%H-%M")}_{args.name}_{args.seed}'
    dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test = get_datasets(args, run_name, fabric)
    net, _, method_modules = get_networks(args, fabric, dataset_train)
    net = fabric.setup(net)
    # dataloader_train, dataset_train_eval, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_train_eval, dataloader_test, move_to_device=True)
    dataloader_train, dataloader_test = fabric.setup_dataloaders(dataloader_train, dataloader_test, move_to_device=True)

    if args.path_load_model != "": load_model(fabric, net, args, strict=False)
    train_t, val_t = get_transformations(args, crop_size=DATASETS[args.dataset]['img_size'])
    if all:
        finetune_all(args, dataloader_train, dataset_train, dataloader_test, net, os.path.join(args.log_dir, run_name),fabric, train_t, val_t, name=args.name_eval)
    else:
        finetune(args, dataloader_train, dataset_train, dataloader_test, net, os.path.join(args.log_dir, run_name), fabric, train_t, val_t, name=args.name_eval)

# ----------------
# main program
# ----------------

if __name__ == '__main__':
    args = get_arguments(config.parser).parse_args()
    if args.mode == "train":
        train()
    if args.mode == "finetune_all":
        end_finetune(all=True)
    if args.mode == "finetune":
        end_finetune()

    # for i in range(args.n_repeat):
    # config.RUN_NAME = config.RUN_NAME
    # train()

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
