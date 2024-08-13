import torch
from torch import nn

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1

def getNoise(n_samples: int, z_dim: int = 100, device="cuda"):
    return torch.randn(n_samples, z_dim, device=device)

class Generator(nn.Module):
    def __init__(self, z_dim: int=100, im_channels: int=3, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(z_dim, 4*4*(hidden_dim*8))
        self.norm = nn.BatchNorm2d(hidden_dim*8)
        self.relu = nn.ReLU(inplace=True)
        self.gen = nn.Sequential(
            self.generatorBlock(hidden_dim*8, hidden_dim*4),
            self.generatorBlock(hidden_dim*4, hidden_dim*2),
            self.generatorBlock(hidden_dim*2, hidden_dim),
            self.generatorBlock(hidden_dim, im_channels, last_layer=True),
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = x.view(len(x), -1, 4, 4)
        x = self.norm(x)
        x = self.relu(x)
        x = self.gen(x)
        return x

    def generatorBlock(self, in_channels, out_channels, kernel_size: int = KERNEL_SIZE, stride: int=STRIDE, padding: int=PADDING, last_layer: bool = False) -> nn.Sequential:
        if not last_layer:
           return nn.Sequential(
               nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
           )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.Tanh()
            )


def get_gen_loss(crit_fake_pred):
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss
