import torch.nn as nn
import torch
class Integration(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.integration_lager=nn.GRU(input_size=embed_dim,hidden_size=)
torch.nn.GRU