import torch

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_action_size
from utils.constants import LOSS, SIMILARITY_FUNCTIONS
from utils.general import normalize, str2bool


class Ciper(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        self.action_head = None
        action_size = get_action_size(args)
        net.add_module("action_predictor", MLPHead(args,2*net.num_output, args.ciper_dim, action_size, self.args.ciper_layers))
        net.register_buffer("action_prediction", torch.empty((args.batch_size, action_size)), persistent=False)
        net.add_module("ciper_action_bn", torch.nn.BatchNorm1d(action_size, affine=False))
        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        self.loss_cpt = 1e-5

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--action_weight', default=1., type=float)
        parser.add_argument('--ciper_dim', default=1024, type=int)
        parser.add_argument('--ciper_layers', default=1, type=int)

        return parser

    def apply(self, net, rep=None, net_target=None, action=None, data=None, **kwargs):
        # t = normalize(action.to(self.args.device))
        t = net.ciper_action_bn(action)

        net.action_prediction = net.action_predictor(torch.cat(rep.split(rep.shape[0]//2), dim=1))
        pred_loss = torch.nn.functional.mse_loss(net.action_prediction, t, reduction="none")
        loss_mean = pred_loss.mean()
        self.loss_store += loss_mean.detach()
        self.loss_cpt += 1
        return self.args.action_weight * loss_mean

    @torch.no_grad()
    def eval(self, *args):
        dict = {"ciper_loss": self.loss_store.item()/self.loss_cpt}
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict