import functorch
import torch
import torchvision
from torchvision import transforms

from modules.loss_module import LossModule
from networks.heads import MLPHead
from utils.augmentations import get_transformations, get_resized_crop, get_flip, get_jitter, get_grayscale, \
    get_transform_list
from utils.constants import LOSS, SIMILARITY_FUNCTIONS, SIMILARITY_FUNCTIONS_SIMPLE, DATASETS
from utils.general import is_target_needed, run_forward
from utils.losses import BYOL


class Ssl(LossModule):

    def __init__(self,args, fabric, net=None, net_target=None, **kwargs):
        self.args = args
        self.fabric=fabric
        net.add_module("projector", MLPHead(args,net.num_output, args.hidden_dim, args.feature_dim, args.hidden_layers))
        net.register_buffer("proj_output", torch.empty((2 * args.batch_size, args.feature_dim)), persistent=False)

        if is_target_needed(args):
            net.register_buffer("prediction", torch.empty((2 * args.batch_size, args.feature_dim)), persistent=False)
            net.add_module("predictor",MLPHead(args,args.feature_dim, args.hidden_dim, args.feature_dim, args.hidden_layers))

        self.loss = LOSS[args.main_loss](args, SIMILARITY_FUNCTIONS[args.similarity], fabric)
        self.k=0
        self.loss_store = self.fabric.to_device(torch.zeros((1,)))
        self.loss_cpt = 1e-5

    @classmethod
    def get_args(cls, parser):
        parser.add_argument('--lambda_vicreg', default=25, type=float)
        parser.add_argument('--mu_vicreg', default=25, type=float)
        parser.add_argument('--v_vicreg', default=1, type=float)
        parser.add_argument('--temperature', default=0.1, type=float)
        parser.add_argument('--classic_weight', default=1, type=float)

        parser.add_argument('--feature_dim', default=128, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--hidden_layers', default=1, type=int)

        return parser

    @torch.no_grad()
    def eval(self, net, *args):
        dict = {"ssl_loss": self.loss_store.item()/self.loss_cpt}
        self.loss_store[:]=0
        self.loss_cpt = 1e-5
        return dict

    def apply(self, net, rep=None, rep_target=None, net_target=None, data=None, **kwargs):
        net.proj_output = run_forward(self.args, rep, net.projector)

        if self.args.main_loss in ['BYOL']:
            net.prediction = net.predictor(net.proj_output)
            net_target.proj_output = run_forward(self.args, rep, net_target.projector).detach()
            y1, y2 = BYOL.get_byol_output(net_target.proj_output, net.prediction)
            loss = self.loss(y1, y2)
        else:
            y1, y2 = net.proj_output.split(net.proj_output.shape[0]//2)
            loss = self.loss(y1, y2)

        loss_mean = loss.mean()
        self.loss_store = self.loss_store + loss_mean.detach()
        self.loss_cpt += 1
        return self.args.classic_weight*loss_mean
