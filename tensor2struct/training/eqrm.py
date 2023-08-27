# pylint: disable=unused-argument
import attr
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import collections

import copy
import higher
import logging

from tensor2struct.modules.distributions import Nonparametric

logger = logging.getLogger("tensor2struct")


class EQRM(nn.Module):
    def __init__(self, device=None, 
                quantile=0.75, 
                burnin_iters=2500, 
                dist=None):
        super().__init__()
        self.device = device
        self.quantile = quantile
        self.burnin_iters = burnin_iters
        
        # self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.quantile, dtype=torch.float64))

        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist
    
    def train(self, model, batch, n_domains):
        assert model.training
        
        losses = []
        # Calculate loss for each env
        for env_batch in batch:
            losses.append(model(env_batch)['loss'])
        
        return losses
    
    def transform(self, losses, step, unlabeled=None):
        losses = torch.cat([loss.reshape(1) for loss in losses])
        
        if step < self.burnin_iters:
            # Burn-in/anneanlig period uses ERM like penalty methods
            loss = torch.mean(losses)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(losses)
            loss = self.dist.icdf(self.alpha)
        
        reset_opt = True if step == self.burnin_iters else False  # Reset optimizer at burnin_iters step
        return loss, reset_opt
    