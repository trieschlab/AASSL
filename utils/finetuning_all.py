import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from networks.heads import MLPHead
from utils.augmentations import get_action_size
from utils.getters import get_scheduler, get_optimizer
from utils.logger import EpochLogger


# Taken in https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/simclr_lin.py

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def run_epoch(args, dataloader, network, lin_model, optimizers=None, fabric=None, t=None, epoch=0):

    num_models = len(args.fine_path)*len(args.fine_proj)
    if "0" in args.fine_proj:
        num_models = num_models - len(args.fine_path) +1
    acc_meters = [AverageMeter(f'acc{i}') for i in range(num_models)]


    for i_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch " + str(epoch)):
        # print(i_batch)
        with torch.no_grad():
            labels = batch[1]
            e = network(t(batch[0][0])).detach()


        index = 0
        done_0 = False
        for p in args.fine_path:
            proj = getattr(network, p)
            if p in ["action_projector","action_predictor"]:
                ep = torch.cat((e, e), dim=1)
            else:
                ep = e
            for fp in range(len(proj.net)):
                if str(fp) in args.fine_proj and (fp != 0 or not done_0):
                    if fp == 0:
                        done_0=True
                    # logits = lin_model[index](ep)
                    logits = lin_model[index](ep.detach())
                    acc = (logits.argmax(dim=1) == labels).float().mean()
                    loss = F.cross_entropy(logits, labels)

                    if fabric is not None:
                        fabric.backward(loss)
                        optimizers[index].step()
                        optimizers[index].zero_grad()
                    acc_meters[index].update(acc.detach(), labels.size(0))
                    index += 1

                ep = proj.net[fp](ep)
        if args.name == "test3" and i_batch > 2:
            break
    return [(acc_meter.sum, acc_meter.count) for acc_meter in  acc_meters ]


def finetune_all(args, train_dataloader, train_set, test_dataloader, net, save_dir, fabric, tf, tv, epoch="", name=""):
    class_number = train_set.n_classes if not hasattr(train_set, "all_classes") else train_set.all_classes[
        args.fine_label]

    modelsT = []
    loggers = []
    found_0 = False
    for p in args.fine_path:
        for fp in args.fine_proj:
            if fp == "0":
                if found_0:
                    continue
                found_0 = True
                num_outputs = net.num_output
            else:
                proj = getattr(net, p)
                if fp == len(proj.net)-1:
                    num_outputs = args.feature_dim
                else:
                    num_outputs = args.hidden_dim
            modelsT.append(nn.Linear(num_outputs, class_number))
            if fabric.global_rank == 0:
                fl = "" if args.finetune_labels == -1 else str(args.finetune_labels)
                output_fname = f'linear_progress{p}{fp}{fl}{name}.txt'
                loggers.append(EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname=output_fname))
    n_epochs = args.n_epochs //2 if args.finetune_labels == -1 else args.n_epochs
    optimizersT = [get_optimizer(args, modelsT[i]) for i in range(len(modelsT))]
    models, optimizers, schedulers = [], [], []
    for i in range(len(optimizersT)):
        modelstt, optimizerstt = fabric.setup(modelsT[i], optimizersT[i])
        models.append(modelstt)
        optimizers.append(optimizerstt)
        schedulers.append(get_scheduler(args, optimizerstt, n_epochs))


    net.eval()
    for epoch in range(1, n_epochs + 1):
        for m in models:
            m.train()
        run_epoch(args, train_dataloader, net, models, optimizers, fabric=fabric, t=tf, epoch=epoch)
        if epoch%5 == 0 or epoch == 1:
            for m in models:
                m.eval()
            acc_cpt = run_epoch(args, test_dataloader, net, models, t=tv)

            all_acc_cpt = fabric.all_reduce(acc_cpt, reduce_op="sum")

            for index in range(len(loggers)):
                logger = loggers[index]
                if fabric.global_rank == 0:
                    logger.log_tabular(f"test_acc", (all_acc_cpt[index][0]/all_acc_cpt[index][1]).item())
                logger.log_tabular("epoch", epoch)
                logger.dump_tabular()
        for i in range(len(schedulers)):
            schedulers[i].step()

    net.train()