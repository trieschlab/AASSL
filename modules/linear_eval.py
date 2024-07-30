

import torch

from modules.loss_module import LossModule
from torch.nn import functional as F
from torch import nn

from utils.general import str2table


class OnlineLinearEval(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, train_dataset=None, **kwargs):
        self.args = args
        self.fabric=fabric
        self.parameters = []
        self.losses = [self.fabric.to_device(torch.zeros((1,)))]*len(args.eval_labels)
        self.train_accs = [self.fabric.to_device(torch.zeros((1,)))]*len(args.eval_labels)
        self.cpt = 0.00001
        if not hasattr(train_dataset, "all_classes"):
            self.classes = [train_dataset.n_classes]*len(args.eval_labels)
        else:
            self.classes = train_dataset.all_classes

        for k in range(len(args.eval_labels)):
            label_type = int(self.args.eval_labels[k])
            net.add_module("sup_lin_projector"+str(label_type), nn.Linear(net.num_output, self.classes[label_type]))
            # net.register_buffer("sup_lin"+str(label_type), torch.empty((2*args.batch_size, self.classes[label_type])), persistent=False)


    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--eval_labels', default="0", type=str2table)
        parser.add_argument('--linear_wait', default=0, type=int)
        return parser

    def apply(self, net, rep=None, net_target=None, data=None, epoch=None, **kwargs):
        if epoch < self.args.linear_wait:
            return 0
        loss = 0
        for k in range(len(self.args.eval_labels)):
            label_type = int(self.args.eval_labels[k])

            label = data[1+label_type]
            # output = network.get_submodule("sup_lin_projector"+l)(rep.detach())
            output = net.get_submodule("sup_lin_projector" + str(label_type))(rep.detach())

            y1, y2 = output.split(output.shape[0]//2)
            l = 0.5*F.cross_entropy(y1, label).mean() + 0.5*F.cross_entropy(y2, label).mean()
            self.losses[k] += l.detach()
            self.train_accs[k] += (0.5*(y1.argmax(dim=1) == label).float().mean() + 0.5*(y2.argmax(dim=1) == label).float().mean()).detach()
            loss += l
        self.cpt += 1
        return loss

    @torch.no_grad()
    def eval(self, network, f_l_test = None):
        dict = {}
        for k in range(len(self.args.eval_labels)):
            dict["lin_loss"+self.args.eval_labels[k]] = self.losses[k].item()/self.cpt
            dict["lin_acc"+self.args.eval_labels[k]] = self.train_accs[k].item()/self.cpt
            self.losses[k][:]=0
            self.train_accs[k][:]=0
        self.cpt = 0.00001

        if f_l_test is not None:
            for k, v in f_l_test.items():
                lin_net = network.get_submodule("sup_lin_projector0")
                output = lin_net(v["features"])
                corrects = output.argmax(dim=-1) == v["labels"]
                dict["test_acc"+k] = corrects.float().mean().item()
        return dict