import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

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


def run_epoch(args, dataloader, network, lin_model, optimizer=None, fabric=None, t=None):
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    for i_batch, batch in tqdm(enumerate(dataloader)):
        # print(i_batch)
        with torch.no_grad():
            labels = batch[1]
            e = network(t(batch[0][0])).detach()
        # print(e[0])

        logits = lin_model(e)
        loss = F.cross_entropy(logits, labels)
        if fabric is not None:
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        acc = (logits.argmax(dim=1) == labels).float().mean()
        loss_meter.update(loss.detach(), labels.size(0))
        acc_meter.update(acc.detach(), labels.size(0))
        if args.name == "test3" and i_batch > 2:
            break
    return loss_meter.sum, acc_meter.sum, loss_meter.count


def finetune(args, train_dataloader, train_set, test_dataloader, net, save_dir, fabric, tf, tv):
    # Prepare model
    lin_model = nn.Linear(net.num_output, train_set.n_classes)

    optimizer = get_optimizer(args, lin_model)
    n_epochs = min(100,max(20,args.n_epochs))
    lin_model, optimizer = fabric.setup( lin_model, optimizer)
    scheduler = get_scheduler(args, optimizer, n_epochs)

    if fabric.global_rank == 0:
        logger = EpochLogger(output_dir=save_dir, exp_name="Seed-" + str(args.seed), output_fname='linear_progress.txt')
    net.eval()
    for epoch in range(1, n_epochs + 1):
        lin_model.train()
        if not args.fine_eval:
            net.train()
        run_epoch(args, train_dataloader, net, lin_model, optimizer, fabric=fabric, t=tf)
        if not args.fine_eval:
            net.eval()
        if epoch%5 == 0 or epoch == 1:
            lin_model.eval()
            loss, acc, cpt = run_epoch(args, test_dataloader, net, lin_model, t=tv)

            all_loss = fabric.all_reduce(loss, reduce_op="sum")
            all_acc = fabric.all_reduce(acc, reduce_op="sum")
            all_cpt = fabric.all_reduce(cpt, reduce_op="sum")
            if fabric.global_rank == 0:
                logger.log_tabular("test_acc", (all_acc/all_cpt).item())
                logger.log_tabular("loss", (all_loss/all_cpt).item())
                logger.log_tabular("epoch", epoch)
                logger.dump_tabular()
        scheduler.step()

    net.train()


