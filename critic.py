import torch
from torch import nn

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1

class Critic(nn.Module):
    def __init__(self, im_channels: int=3, hidden_dim: int = 128):
        super().__init__()
        self.disc = nn.Sequential(
            self.critic_block(im_channels, hidden_dim, first_layer=True),
            self.critic_block(hidden_dim, hidden_dim*2),
            self.critic_block(hidden_dim*2, hidden_dim*4),
            self.critic_block(hidden_dim*4, hidden_dim*8),
        )
        self.proj = nn.Linear(hidden_dim*8*4*4, 1)

    def critic_block(self, input_channels: int, output_channels: int, kernel_size: int = KERNEL_SIZE, stride: int = STRIDE, padding: int = PADDING, first_layer: bool = False):
        if not first_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
    def forward(self, x):
        x = self.disc(x)
        x = x.view(len(x), -1)
        x = self.proj(x)
        return x

def get_gradient(crit, real, fake, epsilon):
    mixed_image = real*epsilon + fake*(1-epsilon)
    mixed_scores = crit(mixed_image)
    gradient = torch.autograd.grad(
        mixed_scores, 
        mixed_image, 
        torch.ones_like(mixed_scores), 
        True,
        True
        )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()
    return penalty

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda*gp
    return crit_loss