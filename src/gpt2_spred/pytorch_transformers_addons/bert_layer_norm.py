import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        #===MOD===
        # Added ``.cuda()`` to end of both statements. 
        self.weight = nn.Parameter(torch.ones(hidden_size)).cuda()
        self.bias = nn.Parameter(torch.zeros(hidden_size)).cuda()
        #===MOD===
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        #===DEBUG===
        # print("bert_layer_norm: self.weight.data type:", type(self.weight.data))
        # print("bert_layer_norm: self.bias.data type:", type(self.bias.data))
        #===DEBUG===
        return self.weight * x + self.bias
