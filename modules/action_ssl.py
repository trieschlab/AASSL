import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_action_size
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from utils.general import normalize, is_target_needed, str2bool
import numpy as np

class ActionSSL(LossModule):
    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        net.add_module("action_projector", MLPHead(args, 2*net.num_output, args.action_dim, args.action_feature_dim, self.args.action_layers))
        net.register_buffer("action_proj_output", torch.empty((args.batch_size, args.feature_dim)), persistent=False)

        action_size = get_action_size(args)
        net.add_module("action_head", torch.nn.Sequential(
            MLPHead(args, action_size, self.args.hidden_dim,  args.action_feature_dim),torch.nn.BatchNorm1d(args.action_feature_dim, affine=False)))
        # self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity])
        self.loss = LOSS[self.args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric, temperature=self.args.action_temperature, lambda_vicreg=self.args.lambda_vicreg_a, mu_vicreg=self.args.mu_vicreg_a, v_vicreg=self.args.v_vicreg_a)

        if is_target_needed(args):
            net.add_module("action_predictor",MLPHead(args,args.action_feature_dim, args.action_dim, args.action_feature_dim,args.hidden_layers))
            net.add_module("action_head_predictor",MLPHead(args,args.action_feature_dim, args.action_dim, args.action_feature_dim,args.hidden_layers))

        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        self.action_mean = self.fabric.to_device(torch.zeros((action_size,)))
        self.action_std = self.fabric.to_device(torch.zeros((action_size,)))
        self.loss_cpt = 1e-5



    def apply(self, net, rep=None, net_target=None, rep_target=None, action=None, data=None, **kwargs):
        # action=action.to(self.args.device)
        # Compute action representation

        action_rep = net.action_head(action)

        #Compute action prediction
        output = torch.cat(rep.split(rep.shape[0] // 2), dim=1)
        net.action_proj_output = net.action_projector(output)
        if self.args.main_loss in ['BYOL']:
            rep_target = torch.cat(rep_target.split(rep.shape[0] // 2), dim=1)
            y1 = net_target.action_projector(torch.cat(rep_target.split(rep.shape[0] // 2), dim=1)).detach()
            y2 = net_target.action_head(action).detach()

            y11 = net.action_predictor(net.action_proj_output)
            y22 = net.action_head_predictor(action_rep)

            loss_mean = self.loss(torch.cat((y1,y2),dim=0), torch.cat((y22,y11),dim=0)).mean()
        else:
            # Compute loss
            loss_mean = self.args.action_weight * self.loss(action_rep, net.action_proj_output).mean()

        #Compute loss
        self.loss_store += loss_mean.detach()
        self.action_mean += action.mean(dim=0)
        self.action_std += torch.pow(action,2).mean(dim=0)

        self.loss_cpt += 1
        return loss_mean

    @torch.no_grad()
    def eval(self, net, *args):
        dict = {"action_loss": self.loss_store.item()/self.loss_cpt}
        for a in range(self.action_mean.shape[0]):
            dict[f"a{a}_mean"] = self.action_mean[a].cpu().item()/self.loss_cpt
            dict[f"a{a}_std"] = np.sqrt(self.action_std[a].cpu().item()/self.loss_cpt - dict[f"a{a}_mean"]**2)
        self.action_mean.zero_()
        self.action_std.zero_()
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--lambda_vicreg_a', default=25, type=float)
        parser.add_argument('--mu_vicreg_a', default=25, type=float)
        parser.add_argument('--v_vicreg_a', default=1, type=float)
        parser.add_argument('--action_temperature', default=0.1, type=float)
        parser.add_argument('--action_dim', default=1024, type=int)
        parser.add_argument('--action_feature_dim', default=128, type=int)
        parser.add_argument('--action_layers', default=1, type=int)
        return parser



