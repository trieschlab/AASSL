import torch
from torch import nn

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_action_size
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from utils.general import normalize, str2bool


class EquivariantSSL(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        action_size = get_action_size(args)

        net.add_module("equivariant_predictor", torch.nn.Sequential(
                nn.Linear(args.feature_dim + args.action_feature_dim, args.feature_dim),
                nn.BatchNorm1d(args.feature_dim)
            ))
        net.add_module("equivariant_projector", MLPHead(args,net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        net.register_buffer("equivariant_proj_output", torch.empty((args.batch_size, args.feature_dim)), persistent=False)

        self.loss = LOSS[self.args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature=self.args.equivariant_temperature)
        net.add_module("equi_action_head", nn.Sequential(MLPHead(args,action_size, self.args.hidden_dim,  args.feature_dim), nn.BatchNorm1d(args.feature_dim)))


        self.loss = LOSS[self.args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature=self.args.equivariant_temperature)
        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        self.loss_cpt = 1e-5

    @classmethod
    def get_args(cls,parser):
        parser.add_argument('--equivariant_temperature', default=0.1, type=float)

        return parser
    def apply(self, net, rep=None, net_target=None, action=None, data=None, **kwargs):

        # Compute action representation
        action = net.equi_action_head(action)

        #Compute action prediction
        half_size = rep.shape[0] // 2
        net.equivariant_proj_output = net.equivariant_projector(rep[:half_size])
        input = torch.cat((net.equivariant_proj_output, action), dim=1)

        #Compute loss
        loss_mean = self.args.action_weight * self.loss(net.equivariant_predictor(input), net.proj_output[half_size:]).mean()
        self.loss_store += loss_mean.detach()
        self.loss_cpt += 1
        return loss_mean


    @torch.no_grad()
    def eval(self, *args):
        dict = {"equivariant_loss": self.loss_store.item()/self.loss_cpt}
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict

