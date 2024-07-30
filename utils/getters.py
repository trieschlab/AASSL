import copy

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, SequentialLR, LambdaLR
from torchvision import transforms as T

from modules.action_ssl import ActionSSL
from modules.ciper import Ciper
from modules.equivariant import EquivariantSSL
from modules.linear_eval import OnlineLinearEval
from modules.ssl import Ssl
from modules.supervised import Supervised
from networks.resnets import resnet18, resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.constants import DATASETS, LOSS
from utils.general import get_dataset_kwargs, is_target_needed



def get_datasets(args, run_name, fabric):
    dataloader_train = None
    dataset_train = None
    transform_train = []
    transform_test_eval=[]
    if args.crop_first:
        transform_train.append(T.RandomResizedCrop(size=DATASETS[args.dataset]['img_size'], scale=(args.min_crop, args.max_crop)))
        transform_test_eval.append(T.Resize(int(DATASETS[args.dataset]['img_size'][0]*8/7), interpolation=T.InterpolationMode.BICUBIC))
        transform_test_eval.append(T.CenterCrop(DATASETS[args.dataset]['img_size'][0]))

    if not args.crop_first and not args.kornia and args.contrast != 'time':
        if args.min_crop != 1 and not args.one_crop:
            transform_train.append(T.RandomResizedCrop(size=DATASETS[args.dataset]['img_size'], scale=(args.min_crop, args.max_crop)))
        if args.flip:
            transform_train.append(T.RandomHorizontalFlip(p=0.5))
        if args.jitter != 0 and not args.unijit:
            s = args.jitter_strength
            transform_train.append(T.RandomApply([T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=args.jitter))
        if args.grayscale and not args.unijit:
            transform_train.append(T.RandomGrayscale(p=0.2))

    transform_train.append(T.ToTensor())
    transform_test_eval.append(T.ToTensor())


    dataset_train_eval = DATASETS[args.dataset]['class'](
        args=args,
        run_name=None,
        split='train',
        transform=T.Compose(transform_test_eval),
        contrastive=False,
        fabric=fabric,
        eval = True,
        **get_dataset_kwargs(DATASETS[args.dataset])
    )
    dataloader_train_eval = DataLoader(dataset_train_eval, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=False, pin_memory=True)
    dataset_test = DATASETS[args.dataset]['class'](
        args=args,
        run_name=run_name,
        split='test',
        transform=T.Compose(transform_test_eval),
        contrastive=False,
        fabric=fabric,
        **get_dataset_kwargs(DATASETS[args.dataset])
    )
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,num_workers=args.n_workers, shuffle=False, drop_last=False, pin_memory=True)
    dataset_train = DATASETS[args.dataset]['class'](
        args=args,
        run_name=None,
        split='train',
        transform=T.Compose(transform_train),
        contrastive=True if (args.contrast in ['time','combined']) else False,
        fabric=fabric,
        **get_dataset_kwargs(DATASETS[args.dataset])
    )

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,num_workers=args.n_workers, shuffle=True, drop_last=True, pin_memory=True)

    return dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test


def get_train_iterator(args, dataloader_train):
    return tqdm(dataloader_train)


def apply_transform(args, transform, x, x_pair):
    return transform(torch.cat([x, x_pair], 0))

def get_network(args):
    kwargs = {"input_channels": 3}

    if args.model == "resnet18":
        model = resnet18(**kwargs)
    elif args.model == "resnet50":
        model = resnet50(**kwargs)
    return model

def get_networks(args, fabric, dataset_train):
    net = get_network(args)
    method_modules = get_modules(args, fabric, net=net, train_dataset=dataset_train)
    if args.num_devices > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if fabric.global_rank == 0:
        print(net)

    net_target = None
    if is_target_needed(args):
        net_target = copy.deepcopy(net)
        if args.compile:
            net_target = torch.compile(net_target)
        net_target.train()
        net_target = fabric.setup(net_target)

    if args.compile:
        net = torch.compile(net)

    return net, net_target, method_modules

def get_optimizer(args, net):

    if args.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=args.lrate, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)


def get_scheduler(args, optimizer, n_epochs):
    if args.cosine_decay:
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.eta_min, last_epoch=-1)
    else:
        scheduler = ExponentialLR(optimizer, 1.0)

    if args.warmup:
        def warmup(current_epoch):
            return (1+current_epoch) / args.warmup
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], [args.warmup])
    return scheduler


MODULES = {
    "classic": Ssl,
    "supervised": Supervised,
    "ciper": Ciper,
    "action": ActionSSL,
    "equivariant":EquivariantSSL,
    "linear_eval": OnlineLinearEval
}

def get_modules(args, fabric,  **kwargs):
    modules = []
    for m in args.modules:
        modules.append(MODULES[m](args, fabric, **kwargs))
    return modules

def get_arguments(parser):
    for _, m in MODULES.items():
        parser = m.get_args(parser)
    for _,d in DATASETS.items():
        parser = d["class"].get_args(parser)
    return parser

