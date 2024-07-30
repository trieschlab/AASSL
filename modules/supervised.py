import os

import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from torch.nn import functional as F

from utils.general import run_forward


class Supervised(LossModule):
    def __init__(self,args, fabric, net=None, net_target=None, n_classes=0, train_dataset=None, **kwargs):
        self.args = args
        self.parameters = []
        net.add_module("sup_projector",MLPHead(args,net.num_output, args.sup_hidden_dim, train_dataset.n_classes, args.sup_layers))

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--sup_hidden_dim', default=1024, type=int)
        parser.add_argument('--sup_layers', default=2, type=int)
        return parser

    def apply(self, net, rep=None, net_target=None, data=None, **kwargs):
        y = run_forward(self.args, rep, net.sup_projector)
        y1, y2 = y.split(y.shape[0]//2)
        loss = 0.5*F.cross_entropy(y1, data[1]).mean() + 0.5*F.cross_entropy(y2, data[1]).mean()
        return loss

    @torch.no_grad()
    def eval(self, network, f_l_test = None):
        dict = {}
        # sup_net = network.get_submodule("sup_projector")

        if f_l_test is not None:
            v = f_l_test["0"]
            # sups = []
            # for f in v["features"].split(self.args.batch_size):
            #     sups.append(sup_net(f))
            # output = torch.cat(sups, dim=0)
            output = v["supervised"]
            corrects = output.argmax(dim=-1) == v["labels"]
            dict["test_sup"] = corrects.float().mean().item()
        return dict


