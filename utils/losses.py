#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# -----
import torch.nn.functional as F
import torch.nn as nn
import torch



class BYOL(nn.Module):
    """
        BYOL loss that maximizes cosine similarity between the online projection (x) and the target projection(x_pair)
    """

    @classmethod
    def get_byol_output(cls, proj_target_output, pred_output):
        mid_size = pred_output.shape[0] // 2

        x_mix = torch.cat((pred_output[:mid_size], proj_target_output[:mid_size]), dim=0)
        y_mix = torch.cat((proj_target_output[mid_size:], pred_output[mid_size:]), dim=0)
        return x_mix, y_mix

    def __init__(self, args, sim_func, fabric, **kwargs):
        """Initialize the SimCLR_TT_Loss class"""
        super(BYOL, self).__init__()
        self.args = args
        self.fabric = fabric

    def forward(self, x, x_target):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of BYOL-TT (Tensor)
        """
        loss = 2-2*F.cosine_similarity(x, x_target, dim=1).mean()
        return loss




class SimCLR(nn.Module):
    def __init__(self, args, sim_func, fabric, batch_size=None, temperature=None, **kwargs):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR, self).__init__()

        self.args = args
        self.fabric = fabric
        self.batch_size = args.batch_size if batch_size is None else batch_size
        self.temperature = args.temperature if temperature is None else temperature
        self.sim_func = sim_func

    def forward(self, x, x_pair):

        N = 2 * self.batch_size * self.args.num_devices

        z_pos, z_pos2= self.fabric.all_gather((x, x_pair), sync_grads=True)
        z_pos = z_pos.view(-1, z_pos.shape[2])
        z_pos2 = z_pos2.view(-1, z_pos2.shape[2])


        z = torch.cat((z_pos, z_pos2), dim=0)
        sim = self.sim_func(z, z) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size*self.args.num_devices)
        sim_j_i = torch.diag(sim, -self.batch_size*self.args.num_devices)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        logits = sim.flatten()[1:].view(N-1, N+1)[:,:-1].reshape(N, N-1)
        negative_loss = torch.logsumexp(logits, dim=1,keepdim=True)
        return (-positive_samples.mean() + negative_loss.mean())

class VicReg(nn.Module):
    def __init__(self, argss, sim_func, fabric, *args, lambda_vicreg=None, mu_vicreg=None, v_vicreg=None, **kwargs):
        super().__init__()
        self.fabric = fabric
        self.args = argss
        self.lambda_vicreg = self.args.lambda_vicreg if lambda_vicreg is None else lambda_vicreg
        self.mu_vicreg = self.args.mu_vicreg if mu_vicreg is None else mu_vicreg
        self.v_vicreg = self.args.v_vicreg if v_vicreg is None else v_vicreg

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x_local, x_pair_local, *args, **kwargs):
        x = self.fabric.all_gather(x_local, sync_grads=True).view(-1, x_local.shape[1])
        x_pair = self.fabric.all_gather(x_pair_local, sync_grads=True).view(-1, x_pair_local.shape[1])
        repr_loss = F.mse_loss(x, x_pair)

        x = x - x.mean(dim=0)
        x_pair = x_pair - x_pair.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(x_pair.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (x_pair.T @ x_pair) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(x.shape[1]) + self.off_diagonal(cov_y).pow_(
            2).sum().div(x.shape[1])
        return self.v_vicreg*cov_loss + self.mu_vicreg*std_loss + self.lambda_vicreg*repr_loss

