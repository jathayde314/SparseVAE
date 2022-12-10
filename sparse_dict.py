import torch
from torch.autograd import Function
from ista import ista
import torch.nn as nn


class SparseDict(nn.Module):
    def __init__(self, K, D, alpha = 1.0):
        super().__init__()
        self.dictionary = torch.nn.Parameter(torch.randn(D, K, requires_grad=True))
        self.alpha = alpha

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous().flatten(start_dim = 1)
        latents = SparseDictFunction.apply(z_e_x_, self.dictionary, self.alpha)
        sparsity = torch.sum(torch.abs(latents) > 1e-6) / z_e_x.shape[0]
        outputs = self.dictionary.mm(latents.T).T
        return outputs.view(*z_e_x.shape), sparsity

class SparseDictFunction(Function):
    @staticmethod
    def forward(ctx, input, Wd, alpha=0.1, tolerance=0.01, max_steps=-1, sample_gradients=0):
        with torch.no_grad():
            Z = ista(input, torch.zeros((input.shape[0], Wd.shape[1])).cuda(), Wd)
        ctx.save_for_backward(input, Wd, Z)
        ctx.sample_gradients = sample_gradients

        return Z

    @staticmethod
    def backward(ctx, grad_output):
        X, Wd, Z = ctx.saved_tensors

        betas = torch.zeros(grad_output.size()).type(grad_output.type()).t()

        for index in list(range(Z.size()[1])):
            nonzero_inds = Z[:,index].nonzero().squeeze(1)
            sampled_dict = Wd[:, nonzero_inds]
            dict_grad_weights = sampled_dict.t().mm(sampled_dict)
            dict_grad_weights += 0.1 * torch.eye(dict_grad_weights.shape[1]).cuda()
            beta = torch.inverse(dict_grad_weights) @ (grad_output[nonzero_inds, index])

            betas[index, nonzero_inds] += beta

        dict_grad = -Wd.mm(betas).mm(Z) + (X.T - Wd.mm(Z.T)).mm(betas.t())

        return grad_output.mm(Wd.t()), dict_grad.detach(), None

