import torch
import torch.nn as nn

import logging

logger = logging.getLogger("tensor2struct")

class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, condition_dim, eps=1e-5):
        """
        Args:
            hidden_size (int): The size of the feature dimension to normalize.
            condition_dim (int): The dimensionality of the condition input.
            eps (float): Small value to avoid division by zero in normalization.
        """
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        
        # Linear layers to generate scale (gamma) and shift (beta) from the condition
        self.gammar_transform = nn.Linear(condition_dim, hidden_size)
        self.beta_transform = nn.Linear(condition_dim, hidden_size)
        
        # Initialize the transformations
        self.reset_params()
        
    def reset_params(self):
        nn.init.constant_(self.gammar_transform.weight, 0.0)
        nn.init.constant_(self.gammar_transform.bias, 1.0)
        nn.init.constant_(self.beta_transform.weight, 0.0)
        nn.init.constant_(self.beta_transform.bias, 0.0)
        for param in self.gammar_transform.parameters():
            param.requires_grad = False
        for param in self.beta_transform.parameters():
            param.requires_grad = False
    def forward(self, x, condition):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            condition (Tensor): Condition tensor of shape (batch_size, condition_dim)
        Returns:
            Tensor: Normalized output with the same shape as x
        To do: remember to check the sizes of all inputs, outputs
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x-mean)**2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        x_out = (x - mean) / std
        gammar = self.gammar_transform(condition)
        beta = self.beta_transform(condition)
        x_out *= gammar.unsqueeze(1)
        x_out += beta.unsqueeze(1)
        
        return x_out